"""Blender side of the EEVEE backend, run by Blender's own Python. ProteinMotion never imports it.

``eevee.EEVEE`` starts one background Blender process per export, in essence::

    blender --background --factory-startup --python _blender_worker.py -- <folder>

(plus ``--python-exit-code 1``, and ``--gpu-backend metal`` on macOS).

Blender's bundled Python has ``bpy``, ``mathutils`` and numpy, but not ProteinMotion, so this
file imports nothing from the package and keeps its own small helpers (``linear``, ``main``).
The two processes talk through ``<folder>`` and the worker's stdin:

- At start, the worker builds a scene it reuses for every frame (camera, three sun lamps,
  world, materials) and writes ``ready.json`` with the Blender version and GPU backend, or
  ``{"error": traceback}``.
- Each stdin line is one JSON frame request: output size, ``EEVEEOptions`` as a dict, the
  camera (eye, target and focus point in world Å, vertical fov in radians, depth of field,
  f-stop), the background as display RGB in [0, 1], and fog. The frame's meshes are already
  in ``frame.npz`` as ``{i}_vertices``, ``{i}_faces``, ``{i}_colors``, ``{i}_normals`` and
  per-face ``{i}_opacity``, with optional ``{i}_strength``, ``{i}_kind`` and ``{i}_unlit``
  (see ``_eevee_geometry``).
- The worker renders ``frame.png`` and writes ``done.json``: ``{"ok": true}`` or an error.
  An error in one frame does not stop the worker. When ``eevee.py`` closes stdin, the loop
  ends and Blender exits.

Transparency is not blended per fragment. Faces are grouped by opacity, rounded to
1/``OPACITY_STEPS``, and the frame is a weighted sum of opaque renders combined in the
compositor (see ``Worker._set_layer_weights``). A faded group therefore never shows its
own hidden back faces; it crossfades as a whole.
"""

import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCALE = 0.0012  # Display meters per ångström, for convenient macroscopic lens settings.
OPACITY_STEPS = 48  # Opacities are rounded to 1/48, so a frame never needs more than 48 layers.
SENSOR_HEIGHT = 24  # Camera sensor height in mm. The focal length is derived from it and the fov.
# Strength of the world light. The world color is divided by it, so the ambient light that
# reaches the molecules equals the scene-linear background color.
AMBIENT_STRENGTH = 0.15


def linear(rgb):
    """Decode sRGB display values in [0, 1] to scene-linear values (the sRGB transfer curve)."""
    rgb = np.asarray(rgb)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def done(folder, name, payload):
    """Write a status file for ``eevee.py``, which polls for it."""
    # Write under a temporary name, then rename: the poller never reads a half-written file.
    temp = folder / (name + ".tmp")
    temp.write_text(json.dumps(payload))
    temp.replace(folder / name)


def point(obj, target):
    """Turn a camera or sun lamp so its local -Z axis points at ``target``, with +Y up."""
    # Match ProteinMotion's Y-up view matrix, including its pole fallback.
    forward = (Vector(target) - obj.location).normalized()
    world_up = Vector((0, 0, 1)) if abs(forward.y) > 0.999 else Vector((0, 1, 0))
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()
    # The rotation's columns are the object's local X, Y and Z axes in world space.
    obj.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()


class Worker:
    """One Blender scene, reused for every frame. Mesh objects are cached between frames.

    World coordinates are only scaled by ``SCALE``; no axes are swapped. ProteinMotion is +Y
    up and Blender's convention is +Z up, but the camera is aimed explicitly, so the image
    is the same.
    """

    def __init__(self):
        # Factory startup has a cube, a camera and a lamp. Start from an empty scene.
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        self.scene = bpy.context.scene
        self._configure_render()
        self._add_camera()
        self.unlit_material = self._unlit_material()
        self.material = self._lit_material()
        self.glow_material = self._glow_material()
        self.backgrounds = {}  # Background radiance solved per requested background color.
        self.lights = self._add_lights()
        self._configure_world()
        # Collections, one per opacity level, and cached objects keyed by (mesh index, level).
        self.groups, self.objects = [], {}
        self.layer_count = 0  # View layers the compositor is currently built for.

    def _configure_render(self):
        scene = self.scene
        # Some Blender versions call the engine BLENDER_EEVEE_NEXT; an unknown name raises TypeError.
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        # The compositor combines the opacity layers into one image.
        scene.render.use_compositing = True
        scene.render.compositor_device = "GPU"
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        # Depth of field is EEVEE's post-process blur, not jittered camera samples. Sprite
        # bokeh is used only for pixels brighter than bokeh_threshold, set very high here.
        scene.eevee.use_bokeh_jittered = False
        scene.eevee.bokeh_threshold = 100
        # No ray tracing, shadows or fast GI. The native renderer has no cast shadows either.
        scene.eevee.use_raytracing = False
        scene.eevee.use_shadows = False
        if hasattr(scene.eevee, "use_fast_gi"):
            scene.eevee.use_fast_gi = False

    def _add_camera(self):
        scene = self.scene
        self.camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        scene.collection.objects.link(self.camera)
        scene.camera = self.camera
        # Near clip 1e-5 m (under 0.01 Å at SCALE) and far clip 1 km: no molecule is ever clipped.
        self.camera.data.clip_start = 0.00001
        self.camera.data.clip_end = 1000
        # With a vertical sensor fit, the focal length set per frame fixes the vertical field
        # of view, which is what ProteinMotion's camera.fov means.
        self.camera.data.sensor_fit = "VERTICAL"
        self.camera.data.sensor_height = SENSOR_HEIGHT
        self.camera.data.dof.aperture_blades = 9  # Nine blades give nearly round bokeh.
        # Depth of field focuses on this empty, which render() moves to the requested focus point.
        self.focus = bpy.data.objects.new("Residue focus", None)
        scene.collection.objects.link(self.focus)
        self.camera.data.dof.focus_object = self.focus

    def _unlit_material(self):
        """Emission-only material for meshes flagged unlit, such as density slices and tunnel rings."""
        unlit = bpy.data.materials.new("Density slice")
        unlit.use_nodes = True
        nodes = unlit.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        attribute = nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = "color"
        unlit.node_tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
        unlit.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
        return unlit

    def _lit_material(self):
        """Principled material for molecules, colored per vertex, with distance fog.

        Sets ``fog_range``, ``fog_strength`` and ``fog_color``, the nodes render() updates per frame.
        """
        mat = bpy.data.materials.new("Protein")
        mat.use_nodes = True
        tree = mat.node_tree
        shader = tree.nodes.get("Principled BSDF")
        shader.inputs["Roughness"].default_value = 0.45
        shader.inputs["Specular IOR Level"].default_value = 0.25
        attr = tree.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = "color"
        tree.links.new(attr.outputs["Color"], shader.inputs["Base Color"])
        # Distance fog, as in the native renderer: blend toward the background color by
        # smoothstep(start, end, camera distance) * depth_cue.
        camera_data = tree.nodes.new("ShaderNodeCameraData")
        self.fog_range = tree.nodes.new("ShaderNodeMapRange")
        self.fog_range.interpolation_type = "SMOOTHSTEP"
        self.fog_range.clamp = True
        tree.links.new(camera_data.outputs["View Distance"], self.fog_range.inputs["Value"])
        self.fog_strength = tree.nodes.new("ShaderNodeMath")
        self.fog_strength.operation = "MULTIPLY"
        tree.links.new(self.fog_range.outputs["Result"], self.fog_strength.inputs[0])
        # The native renderer fogs display colors; the equivalent scene-linear blend is
        # 1 - (1 - f)^2.2, so the same depth_cue dims distant geometry equally.
        keep = tree.nodes.new("ShaderNodeMath")
        keep.operation = "SUBTRACT"
        keep.inputs[0].default_value = 1.0
        tree.links.new(self.fog_strength.outputs[0], keep.inputs[1])
        power = tree.nodes.new("ShaderNodeMath")
        power.operation = "POWER"
        tree.links.new(keep.outputs[0], power.inputs[0])
        power.inputs[1].default_value = 2.2
        fog = tree.nodes.new("ShaderNodeMath")
        fog.operation = "SUBTRACT"
        fog.inputs[0].default_value = 1.0
        tree.links.new(power.outputs[0], fog.inputs[1])
        # Mix the lit surface with an emission of the fog color by that factor.
        self.fog_color = tree.nodes.new("ShaderNodeEmission")
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(fog.outputs[0], mix.inputs["Fac"])
        tree.links.new(shader.outputs[0], mix.inputs[1])
        tree.links.new(self.fog_color.outputs[0], mix.inputs[2])
        tree.links.new(mix.outputs[0], tree.nodes.get("Material Output").inputs["Surface"])
        return mat

    def _glow_material(self):
        """Additive halo: emission over a transparent surface, brightest at the center."""
        mat = bpy.data.materials.new("Glow")
        mat.use_nodes = True
        # Only the near hemisphere of each halo sphere draws, so a pixel gets one layer of glow.
        mat.use_backface_culling = True
        # Blended transparency; the property name depends on the Blender version.
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
        tree = mat.node_tree
        nodes, links = tree.nodes, tree.links
        nodes.clear()

        def math_node(op, a=None, b=None):
            # A Math node; each input is a constant or another node's output socket.
            node = nodes.new("ShaderNodeMath")
            node.operation = op
            for socket, value in ((0, a), (1, b)):
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    node.inputs[socket].default_value = value
                else:
                    links.new(value, node.inputs[socket])
            return node.outputs[0]

        # |normal · view direction| is 1 at the center of the sphere's disc and 0 at its rim.
        geometry = nodes.new("ShaderNodeNewGeometry")
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        links.new(geometry.outputs["Normal"], dot.inputs[0])
        links.new(geometry.outputs["Incoming"], dot.inputs[1])
        facing = math_node("ABSOLUTE", dot.outputs["Value"])
        # Projected radius squared across the halo disc, then the native halo and core.
        r2 = math_node("SUBTRACT", 1.0, math_node("MULTIPLY", facing, facing))
        # Tight soft halo plus a hot core; AgX compresses highlights, hence the larger weights.
        halo = math_node("EXPONENT", math_node("MULTIPLY", r2, -6.0))
        core = math_node("EXPONENT", math_node("MULTIPLY", r2, -38.0))
        strength = math_node("ADD", math_node("MULTIPLY", halo, 0.9), math_node("MULTIPLY", core, 9.0))
        attr = nodes.new("ShaderNodeVertexColor")
        attr.layer_name = "color"
        emission = nodes.new("ShaderNodeEmission")
        links.new(attr.outputs["Color"], emission.inputs["Color"])
        links.new(strength, emission.inputs["Strength"])
        # Transparent plus emission adds light without hiding what is behind the halo.
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        add = nodes.new("ShaderNodeAddShader")
        links.new(transparent.outputs[0], add.inputs[0])
        links.new(emission.outputs[0], add.inputs[1])
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(add.outputs[0], output.inputs["Surface"])
        return mat

    def _add_lights(self):
        """Key, fill and rim sun lamps, as (object, offset, strength, size) tuples."""
        lights = []
        # Offsets are in meters in the camera frame (right, up, toward the camera); render()
        # rotates them with the camera. The size is stored but not applied to the lamp.
        for name, energy, location, size, color in (
            ("Key", 2.0, (-0.055, -0.065, 0.10), 0.075, (1, 0.91, 0.82)),
            ("Fill", 0.7, (0.08, -0.03, 0.04), 0.07, (0.70, 0.85, 1)),
            ("Rim", 1.4, (0.025, 0.065, 0.065), 0.055, (0.69, 0.85, 1)),
        ):
            light = bpy.data.lights.new(name, "SUN")
            light.energy, light.color, light.use_shadow = energy, color, False
            obj = bpy.data.objects.new(name, light)
            self.scene.collection.objects.link(obj)
            lights.append((obj, np.array(location), energy, size))
        return lights

    def _configure_world(self):
        scene = self.scene
        scene.world.use_nodes = True
        world = scene.world.node_tree
        self.world = world.nodes.get("Background")
        self.world.inputs["Strength"].default_value = AMBIENT_STRENGTH
        # Camera rays see the requested background exactly; lighting keeps a dim ambient.
        self.backdrop = world.nodes.new("ShaderNodeBackground")
        path = world.nodes.new("ShaderNodeLightPath")
        mix = world.nodes.new("ShaderNodeMixShader")
        world.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
        world.links.new(self.world.outputs[0], mix.inputs[1])
        world.links.new(self.backdrop.outputs[0], mix.inputs[2])
        world.links.new(mix.outputs[0], world.nodes.get("World Output").inputs["Surface"])

    def _display(self, colors):
        """Scene-linear colors as the view transform and look display them, in [0, 1]."""
        # Save a strip of pixels with save_render, which applies the scene's color management
        # (AgX and its look), then read the displayed sRGB values back.
        count = len(colors)
        image = bpy.data.images.new("Color probe", count, 1, alpha=True, float_buffer=True)
        pixels = np.ones((count, 4), np.float32)
        pixels[:, :3] = colors
        image.pixels.foreach_set(pixels.ravel())
        path = os.path.join(tempfile.gettempdir(), f"proteinmotion-probe-{os.getpid()}.png")
        image.save_render(path, scene=self.scene)
        loaded = bpy.data.images.load(path)
        result = np.array(loaded.pixels[:]).reshape(count, 4)[:, :3]
        bpy.data.images.remove(image)
        bpy.data.images.remove(loaded)
        os.remove(path)
        return result

    def background_radiance(self, background):
        """Scene-linear color that the view transform displays as the requested background."""
        # Each solve renders probe images, so results are cached per background color.
        key = tuple(np.round(background, 6))
        if key not in self.backgrounds:
            # AgX has no simple inverse. Fixed-point iteration: scale each channel by
            # target / shown until the displayed color matches. The per-step factor is
            # clamped to [0.2, 5] to keep the iteration stable, the value to [0, 20].
            target = linear(np.asarray(background, float))
            value = target.copy()
            for _ in range(16):
                shown = linear(self._display(np.array([value]))[0])
                value = np.clip(value * np.clip(target / np.maximum(shown, 1e-6), 0.2, 5.0), 0.0, 20.0)
            self.backgrounds[key] = value
        return self.backgrounds[key]

    def compositor(self, count):
        """Build ``count`` view layers and a compositor chain that mixes their renders in order.

        The chain starts from a flat background color (``self.rgb``) and mixes in view layer 0,
        then 1, and so on, through the nodes in ``self.mixes``. render() sets the mix factors.
        """
        scene = self.scene
        for layer in list(scene.view_layers)[1:]:
            scene.view_layers.remove(layer)
        scene.view_layers[0].name = "Opacity 0"
        for i in range(1, count):
            scene.view_layers.new(f"Opacity {i}")
        if hasattr(scene, "compositing_node_group"):
            # Newer Blender: the compositor is a node group with an explicit Image output.
            old = scene.compositing_node_group
            tree = bpy.data.node_groups.new("Opacity composition", "CompositorNodeTree")
            scene.compositing_node_group = tree
            if old:
                bpy.data.node_groups.remove(old)
            tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            output = tree.nodes.new("NodeGroupOutput")
        else:
            # Older Blender: the scene owns the compositor tree and ends in a Composite node.
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
            # Blender 5 uses the generic Mix node in the compositor; earlier versions MixRGB.
            mix = (
                tree.nodes.new("ShaderNodeMix")
                if bpy.app.version >= (5, 0, 0)
                else tree.nodes.new("CompositorNodeMixRGB")
            )
            if hasattr(mix, "data_type"):
                # The generic Mix node has float, vector and color sockets of the same names.
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
        """Render one frame request to ``folder/frame.png`` from the meshes in ``folder/frame.npz``."""
        scene = self.scene
        opts = request["options"]
        width, height = request["width"], request["height"]
        # Render larger than the output; eevee.py resizes the PNG to the output size.
        scene.render.resolution_x = round(width * opts["supersampling"])
        scene.render.resolution_y = round(height * opts["supersampling"])
        scene.eevee.taa_render_samples = opts["samples"]
        # max_blur is in output pixels; Blender's bokeh limit is in render pixels.
        scene.eevee.bokeh_max_size = opts["max_blur"] * opts["supersampling"]
        target = self._place_camera(request)
        self._place_lights(request, target)
        radiance = self._set_background_and_fog(request)
        with np.load(folder / "frame.npz") as data:
            alphas, levels = self._opacity_levels(data, request["meshes"], opts["max_opacity_layers"])
            count = max(1, len(levels))
            self._sync_meshes(data, alphas, levels, count)
        if self.layer_count != count:
            self.compositor(count)
        self._show_groups(levels)
        self._set_layer_weights(levels, radiance)
        scene.render.filepath = str(folder / "frame.png")
        bpy.ops.render.render(write_still=True)

    def _place_camera(self, request):
        """Pose the camera, lens and focus from the request; return the view target in meters."""
        self.camera.location = np.array(request["eye"]) * SCALE
        target = np.array(request["target"]) * SCALE
        point(self.camera, target)
        # Pinhole lens: focal length = sensor height / (2 tan(fov / 2)), fov vertical in radians.
        self.camera.data.lens = SENSOR_HEIGHT / (2 * math.tan(request["fov"] / 2))
        self.camera.data.dof.use_dof = request["dof"]
        self.camera.data.dof.aperture_fstop = request["fstop"]
        self.focus.location = np.array(request["focus"]) * SCALE
        return target

    def _place_lights(self, request, target):
        # The lamps keep fixed offsets in the camera frame, so key, fill and rim turn with the
        # view. Offsets were set for a camera 0.186 m (155 Å) from the target and scale with
        # the distance. A sun lamp lights from its direction only, so the scaling moves the
        # lamp objects without changing the lighting.
        distance = np.linalg.norm(np.array(request["eye"]) - request["target"]) * SCALE
        scale = max(distance / 0.186, 0.01)
        rotation = self.camera.rotation_euler.to_matrix()
        for light, position, energy, size in self.lights:
            light.location = target + rotation @ Vector(position * scale)
            point(light, target)
            light.data.energy = energy

    def _set_background_and_fog(self, request):
        """Set ambient light, backdrop and fog for the request; return the background radiance."""
        bg = linear(request["background"])
        # Dim environment illumination in the background color.
        self.world.inputs["Color"].default_value = (*bg / AMBIENT_STRENGTH, 1)
        radiance = self.background_radiance(request["background"])
        self.backdrop.inputs["Color"].default_value = (*radiance, 1)
        # Fog as [start, end] camera distance in Å and strength; strength 0 when absent.
        start, end, strength = request.get("fog", (0.0, 1.0, 0.0))
        self.fog_range.inputs["From Min"].default_value = start * SCALE
        # Keep the range at least 1e-3 Å wide so the map range never divides by zero.
        self.fog_range.inputs["From Max"].default_value = max(end, start + 1e-3) * SCALE
        self.fog_strength.inputs[1].default_value = strength
        # Fog fades toward the same radiance the camera sees as background.
        self.fog_color.inputs["Color"].default_value = (*radiance, 1)
        return radiance

    @staticmethod
    def _opacity_levels(data, meshes, max_layers):
        """Per-face opacities of each mesh, rounded to 1/OPACITY_STEPS, and their distinct positive values."""
        alphas = [
            np.clip(np.round(data[f"{i}_opacity"] * OPACITY_STEPS) / OPACITY_STEPS, 0, 1)
            for i in range(meshes)
        ]
        levels = np.unique(np.concatenate(alphas)) if alphas else np.array([])
        levels = levels[levels > 0]
        # Each level costs one view layer, so refuse rather than silently merge levels.
        if len(levels) > max_layers:
            raise ValueError(
                f"This frame needs {len(levels)} opacity layers; increase EEVEEOptions(max_opacity_layers=...) "
                "or use fewer simultaneous residue opacity values."
            )
        return alphas, levels

    def _sync_meshes(self, data, alphas, levels, count):
        """Make the cached Blender objects match this frame's meshes, split by opacity level."""
        scene = self.scene
        # One collection per opacity level. Collections are kept for later frames.
        while len(self.groups) < count:
            collection = bpy.data.collections.new(f"Opacity group {len(self.groups)}")
            scene.collection.children.link(collection)
            self.groups.append(collection)
        active = set()
        for i, alpha in enumerate(alphas):
            vertices, faces, colors, normals, material = self._mesh_arrays(data, i)
            # Faces that share one rounded opacity become one object in that level's collection.
            for value in np.unique(alpha[alpha > 0]):
                group = int(np.searchsorted(levels, value))
                key = (i, group)
                active.add(key)
                self._sync_object(key, vertices, faces[alpha == value], colors, normals, material)
        # Drop unused meshes promptly; topology changes must not accumulate memory.
        for key in list(self.objects):
            if key not in active:
                obj = self.objects.pop(key)[0]
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.meshes.remove(mesh)

    def _mesh_arrays(self, data, i):
        """Arrays and material for mesh ``i`` of frame.npz: vertices in meters, colors as linear RGBA."""
        vertices = data[f"{i}_vertices"] * SCALE
        faces = data[f"{i}_faces"]
        # Colors arrive as display sRGB; the color attribute holds scene-linear values.
        colors = linear(data[f"{i}_colors"])
        if f"{i}_strength" in data:
            # Glow halos scale their emission color by a per-vertex intensity.
            colors = colors * data[f"{i}_strength"][:, None]
        colors = np.c_[colors, np.ones(len(vertices))].astype(np.float32)
        kind = str(data[f"{i}_kind"]) if f"{i}_kind" in data else "lit"
        if bool(data.get(f"{i}_unlit", False)):
            kind = "unlit"
        material = {"unlit": self.unlit_material, "glow": self.glow_material}.get(kind, self.material)
        normals = data[f"{i}_normals"]
        return vertices, faces, colors, normals, material

    def _sync_object(self, key, vertices, subset, colors, normals, material):
        """Create or update the object that draws the faces ``subset`` of one mesh at one opacity."""
        i, group = key
        cached = self.objects.get(key)
        # Cache entries are (object, vertex count, faces, vertices, colors, normals). The Blender
        # mesh is rebuilt only when its topology or material changed; otherwise its coordinates,
        # colors and normals are updated in place, which is cheaper.
        rebuild = (
            cached is None
            or cached[1] != len(vertices)
            or not np.array_equal(cached[2], subset)
            or cached[0].data.materials[0].name != material.name
        )
        if rebuild:
            if cached:
                old = cached[0].data
                bpy.data.objects.remove(cached[0], do_unlink=True)
                bpy.data.meshes.remove(old)
            # Every level's object holds all of the mesh's vertices but only its own faces.
            mesh = bpy.data.meshes.new(f"Mesh {i}:{group}")
            mesh.from_pydata(vertices.tolist(), [], subset.tolist())
            mesh.update()
            # Smooth shading: face corners use vertex normals, not the flat face normal.
            mesh.polygons.foreach_set("use_smooth", np.ones(len(subset), bool))
            mesh.color_attributes.new(name="color", type="FLOAT_COLOR", domain="POINT")
            mesh.materials.append(material)
            obj = bpy.data.objects.new(mesh.name, mesh)
            self.groups[group].objects.link(obj)
        else:
            obj = cached[0]
        # A rebuilt mesh has its coordinates from from_pydata; its colors and normals are not set yet.
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

    def _show_groups(self, levels):
        # View layer i renders opacity groups i and above, opaque: every face at least as
        # opaque as levels[i]. Spare groups from frames with more levels stay excluded.
        for i, layer in enumerate(self.scene.view_layers):
            for j, group in enumerate(self.groups):
                layer.layer_collection.children[group.name].exclude = j < i or j >= len(levels)

    def _set_layer_weights(self, levels, radiance):
        # Integral of opaque renders above each opacity threshold. With two levels,
        # this is exactly the prototype's isolated-helix / complete-protein crossfade.
        # Layer k gets weight levels[k] - levels[k - 1] (levels[0] for the first layer), and the
        # background keeps 1 - levels[-1].
        # A face at levels[k] appears in layers 0..k, so when nothing covers it, it adds
        # exactly levels[k] of its color. Each mix node blends the running result with the
        # next layer by weight / (weights so far), a running weighted mean whose weights
        # sum to 1 after the last layer.
        self.rgb.outputs[0].default_value = (*radiance, 1)
        weights = np.diff(np.r_[0, levels]) if len(levels) else np.array([0])
        total = 1 - float(levels[-1]) if len(levels) else 1
        for mix, weight in zip(self.mixes, weights):
            total += float(weight)
            mix.inputs[0].default_value = float(weight) / max(total, 1e-8)


def main():
    # Blender ignores arguments after "--" and leaves them for the script: the exchange folder.
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
    # One JSON request per line. The loop ends when eevee.py closes stdin.
    for line in sys.stdin:
        try:
            worker.render(json.loads(line), folder)
            done(folder, "done.json", {"ok": True})
        except Exception:
            done(folder, "done.json", {"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
