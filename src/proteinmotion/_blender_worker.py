"""Private worker run by Blender's Python, not imported by ProteinMotion."""

import json
import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCALE = 0.0012  # Display meters per ångström, for convenient macroscopic lens settings.


def linear(rgb):
    rgb = np.asarray(rgb)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def done(folder, name, payload):
    temp = folder / (name + ".tmp")
    temp.write_text(json.dumps(payload))
    temp.replace(folder / name)


def point(obj, target):
    # Match ProteinMotion's Y-up view matrix, including its pole fallback.
    forward = (Vector(target) - obj.location).normalized()
    world_up = Vector((0, 0, 1)) if abs(forward.y) > 0.999 else Vector((0, 1, 0))
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()
    obj.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()


class Worker:
    def __init__(self):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        self.scene = scene = bpy.context.scene
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.use_compositing = True
        scene.render.compositor_device = "GPU"
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.eevee.use_bokeh_jittered = False
        scene.eevee.bokeh_threshold = 100
        scene.eevee.use_raytracing = False
        scene.eevee.use_shadows = False
        if hasattr(scene.eevee, "use_fast_gi"):
            scene.eevee.use_fast_gi = False
        self.camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        scene.collection.objects.link(self.camera)
        scene.camera = self.camera
        self.camera.data.clip_start = 0.00001
        self.camera.data.clip_end = 1000
        self.camera.data.sensor_fit = "VERTICAL"
        self.camera.data.sensor_height = 24
        self.camera.data.dof.aperture_blades = 9
        self.focus = bpy.data.objects.new("Residue focus", None)
        scene.collection.objects.link(self.focus)
        self.camera.data.dof.focus_object = self.focus
        self.unlit_material = unlit = bpy.data.materials.new("Density slice")
        unlit.use_nodes = True
        nodes = unlit.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        attribute = nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = "color"
        unlit.node_tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
        unlit.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
        self.material = mat = bpy.data.materials.new("Protein")
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Roughness"].default_value = 0.45
        shader.inputs["Specular IOR Level"].default_value = 0.25
        attr = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = "color"
        mat.node_tree.links.new(attr.outputs["Color"], shader.inputs["Base Color"])
        self.lights = []
        for name, energy, location, size, color in (
            ("Key", 2.0, (-0.055, -0.065, 0.10), 0.075, (1, 0.91, 0.82)),
            ("Fill", 0.7, (0.08, -0.03, 0.04), 0.07, (0.70, 0.85, 1)),
            ("Rim", 1.4, (0.025, 0.065, 0.065), 0.055, (0.69, 0.85, 1)),
        ):
            light = bpy.data.lights.new(name, "SUN")
            light.energy, light.color, light.use_shadow = energy, color, False
            obj = bpy.data.objects.new(name, light)
            scene.collection.objects.link(obj)
            self.lights.append((obj, np.array(location), energy, size))
        scene.world.use_nodes = True
        self.world = scene.world.node_tree.nodes.get("Background")
        self.world.inputs["Strength"].default_value = 0.15
        self.groups, self.objects = [], {}
        self.layer_count = 0

    def compositor(self, count):
        scene = self.scene
        for layer in list(scene.view_layers)[1:]:
            scene.view_layers.remove(layer)
        scene.view_layers[0].name = "Opacity 0"
        for i in range(1, count):
            scene.view_layers.new(f"Opacity {i}")
        if hasattr(scene, "compositing_node_group"):
            old = scene.compositing_node_group
            tree = bpy.data.node_groups.new("Opacity composition", "CompositorNodeTree")
            scene.compositing_node_group = tree
            if old:
                bpy.data.node_groups.remove(old)
            tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            output = tree.nodes.new("NodeGroupOutput")
        else:
            scene.use_nodes = True
            tree = scene.node_tree
            tree.nodes.clear()
            output = tree.nodes.new("CompositorNodeComposite")
        self.rgb = tree.nodes.new("CompositorNodeRGB")
        current = self.rgb.outputs[0]
        self.mixes = []
        for i in range(count):
            render = tree.nodes.new("CompositorNodeRLayers")
            render.layer = f"Opacity {i}"
            mix = (
                tree.nodes.new("ShaderNodeMix")
                if bpy.app.version >= (5, 0, 0)
                else tree.nodes.new("CompositorNodeMixRGB")
            )
            if hasattr(mix, "data_type"):
                mix.data_type = "RGBA"
                a = next(s for s in mix.inputs if s.name == "A" and s.type == "RGBA")
                b = next(s for s in mix.inputs if s.name == "B" and s.type == "RGBA")
                result = next(s for s in mix.outputs if s.type == "RGBA")
            else:
                a, b, result = mix.inputs[1], mix.inputs[2], mix.outputs[0]
            tree.links.new(current, a)
            tree.links.new(render.outputs["Image"], b)
            current = result
            self.mixes.append(mix)
        tree.links.new(current, output.inputs["Image"])
        self.layer_count = count

    def render(self, request, folder):
        scene = self.scene
        opts = request["options"]
        width, height = request["width"], request["height"]
        scene.render.resolution_x = round(width * opts["supersampling"])
        scene.render.resolution_y = round(height * opts["supersampling"])
        scene.eevee.taa_render_samples = opts["samples"]
        scene.eevee.bokeh_max_size = opts["max_blur"] * opts["supersampling"]
        self.camera.location = np.array(request["eye"]) * SCALE
        target = np.array(request["target"]) * SCALE
        point(self.camera, target)
        self.camera.data.lens = 24 / (2 * math.tan(request["fov"] / 2))
        self.camera.data.dof.use_dof = request["dof"]
        self.camera.data.dof.aperture_fstop = request["fstop"]
        self.focus.location = np.array(request["focus"]) * SCALE
        # Lights follow the camera's orientation and scale, keeping large structures lit.
        distance = np.linalg.norm(np.array(request["eye"]) - request["target"]) * SCALE
        scale = max(distance / 0.186, 0.01)
        rotation = self.camera.rotation_euler.to_matrix()
        for light, position, energy, size in self.lights:
            light.location = target + rotation @ Vector(position * scale)
            point(light, target)
            light.data.energy = energy
        bg = linear(request["background"])
        # Match the requested background while retaining dim environment illumination.
        self.world.inputs["Color"].default_value = (*bg / 0.15, 1)
        with np.load(folder / "frame.npz") as data:
            alphas = [np.clip(np.round(data[f"{i}_opacity"], 6), 0, 1) for i in range(request["meshes"])]
            levels = np.unique(np.concatenate(alphas)) if alphas else np.array([])
            levels = levels[levels > 0]
            if len(levels) > opts["max_opacity_layers"]:
                raise ValueError(
                    f"This frame needs {len(levels)} opacity layers; increase EEVEEOptions(max_opacity_layers=...) "
                    "or use fewer simultaneous residue opacity values."
                )
            count = max(1, len(levels))
            while len(self.groups) < count:
                collection = bpy.data.collections.new(f"Opacity group {len(self.groups)}")
                scene.collection.children.link(collection)
                self.groups.append(collection)
            active = set()
            for i, alpha in enumerate(alphas):
                vertices = data[f"{i}_vertices"] * SCALE
                faces = data[f"{i}_faces"]
                colors = np.c_[linear(data[f"{i}_colors"]), np.ones(len(vertices))].astype(np.float32)
                normals = data[f"{i}_normals"]
                for value in np.unique(alpha[alpha > 0]):
                    group = int(np.searchsorted(levels, value))
                    key = (i, group)
                    active.add(key)
                    subset = faces[alpha == value]
                    cached = self.objects.get(key)
                    rebuild = (
                        cached is None or cached[1] != len(vertices) or not np.array_equal(cached[2], subset)
                    )
                    if rebuild:
                        if cached:
                            old = cached[0].data
                            bpy.data.objects.remove(cached[0], do_unlink=True)
                            bpy.data.meshes.remove(old)
                        mesh = bpy.data.meshes.new(f"Mesh {i}:{group}")
                        mesh.from_pydata(vertices.tolist(), [], subset.tolist())
                        mesh.update()
                        mesh.polygons.foreach_set("use_smooth", np.ones(len(subset), bool))
                        mesh.color_attributes.new(name="color", type="FLOAT_COLOR", domain="POINT")
                        mesh.materials.append(
                            self.unlit_material if bool(data.get(f"{i}_unlit", False)) else self.material
                        )
                        obj = bpy.data.objects.new(mesh.name, mesh)
                        self.groups[group].objects.link(obj)
                    else:
                        obj = cached[0]
                    coordinates_changed = rebuild or not np.array_equal(cached[3], vertices)
                    colors_changed = rebuild or not np.array_equal(cached[4], colors)
                    normals_changed = rebuild or not np.array_equal(cached[5], normals)
                    if coordinates_changed and not rebuild:
                        obj.data.vertices.foreach_set("co", vertices.ravel())
                    obj.hide_render = False
                    if colors_changed:
                        obj.data.color_attributes["color"].data.foreach_set("color", colors.ravel())
                    if normals_changed or coordinates_changed:
                        obj.data.normals_split_custom_set_from_vertices(normals.tolist())
                    if coordinates_changed or colors_changed or normals_changed:
                        obj.data.update()
                    self.objects[key] = obj, len(vertices), subset, vertices, colors, normals
            # Drop unused meshes promptly; topology changes must not accumulate memory.
            for key in list(self.objects):
                if key not in active:
                    obj = self.objects.pop(key)[0]
                    mesh = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    bpy.data.meshes.remove(mesh)
        if self.layer_count != count:
            self.compositor(count)
        for i, layer in enumerate(scene.view_layers):
            for j, group in enumerate(self.groups):
                layer.layer_collection.children[group.name].exclude = j < i or j >= len(levels)
        # Integral of opaque renders above each opacity threshold. With two levels,
        # this is exactly the prototype's isolated-helix / complete-protein crossfade.
        self.rgb.outputs[0].default_value = (*bg, 1)
        weights = np.diff(np.r_[0, levels]) if len(levels) else np.array([0])
        total = 1 - float(levels[-1]) if len(levels) else 1
        for mix, weight in zip(self.mixes, weights):
            total += float(weight)
            mix.inputs[0].default_value = float(weight) / max(total, 1e-8)
        scene.render.filepath = str(folder / "frame.png")
        bpy.ops.render.render(write_still=True)


def main():
    folder = Path(sys.argv[sys.argv.index("--") + 1])
    try:
        if bpy.app.version < (4, 5, 0):
            raise RuntimeError("Blender 4.5 or later is required")
        worker = Worker()
        done(
            folder,
            "ready.json",
            dict(
                renderer="EEVEE",
                blender=bpy.app.version_string,
                backend=bpy.context.preferences.system.gpu_backend,
            ),
        )
    except Exception:
        done(folder, "ready.json", {"error": traceback.format_exc()})
        return
    for line in sys.stdin:
        try:
            worker.render(json.loads(line), folder)
            done(folder, "done.json", {"ok": True})
        except Exception:
            done(folder, "done.json", {"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
