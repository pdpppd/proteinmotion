"""Molecular surface definitions and their Studio GPU integration."""

import numpy as np
import pytest

from proteinmotion import Protein, ProteinScene, Representation, Reveal, SetOpacity, StudioLook
from proteinmotion.camera import Camera
from proteinmotion.renderer import Renderer
from proteinmotion.structure import Atom, Residue, Topology
from proteinmotion.studio import StudioRenderer


def atom(element="C"):
    # A hetero-only object also checks the old density path's untraced-atom omission.
    return Protein(
        Topology(
            (Atom("A", 1, "", "LIG", element, element, 0),),
            (Residue("A", 1, "", "LIG", 0, -1),),
            np.empty((0, 2), np.uint32),
            (),
        ),
        [[0, 0, 0]],
    ).set_color("#8FE3C0")


@pytest.mark.gpu
@pytest.mark.parametrize("element,radius", [("C", 1.7), ("O", 1.52), ("S", 1.8)])
def test_studio_vdw_radii_probe_and_surface_switches(element, radius):
    p = atom(element)
    camera = Camera()
    camera.distance, camera.radius, camera.depth_cue = 16, 5, 0
    look = StudioLook(effects="clean", ambient_occlusion=0)
    with StudioRenderer(192, 192, look=look) as renderer:
        images = {}
        for kind, probe, expected in [
            ("vdw", 0.0, radius),
            ("vdw", 2.0, radius),
            ("sas", 0.5, radius + 0.5),
            ("sas", 1.4, radius + 1.4),
            ("ses", 1.4, radius),
        ]:
            p.surface(kind=kind, probe_radius=probe, resolution=0.2)
            image = renderer.render([p], camera, (0, 0, 0))
            mesh = renderer._molecules[p].surface.mesh
            assert np.linalg.norm(mesh.vertices, axis=1).mean() == pytest.approx(expected, abs=0.16)
            images[kind, probe] = image
        np.testing.assert_array_equal(images["vdw", 0.0], images["vdw", 2.0])
        # Compare rendered silhouettes, independent of Studio lighting values.
        area = {key: np.any(im[:, :, :3] > 10, axis=2).sum() for key, im in images.items()}
        assert area["vdw", 0.0] > 100
        assert area["sas", 1.4] > area["sas", 0.5] > area["vdw", 0.0]
        assert not renderer._grids
        p.surface(kind="vdw", resolution=0.05, max_voxels=1000)
        with pytest.raises(ValueError, match="max_voxels"):
            renderer.render([p], camera, (0, 0, 0))


@pytest.mark.gpu
def test_studio_shares_native_geometry_and_honors_resolution(protein):
    camera = Camera().frame(protein, aspect=1)
    with Renderer(192, 192) as native, StudioRenderer(192, 192) as studio:
        counts = []
        for kind, spacing in [("vdw", 0.9), ("sas", 0.9), ("ses", 0.9), ("ses", 0.45)]:
            protein.surface(kind=kind, resolution=spacing)
            native.render([protein], camera, (0, 0, 0))
            studio.render([protein], camera, (0, 0, 0))
            a, b = [r._molecules[protein].surface.mesh for r in (native, studio)]
            for field in ("vertices", "normals", "faces", "owners", "weights"):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
            counts.append(len(b.vertices))
        assert counts[-1] > 2 * counts[-2]


@pytest.mark.gpu
@pytest.mark.parametrize("msaa", [1, 4])
def test_studio_surface_cutaway_transparency_and_seek(protein, msaa):
    protein.surface(resolution=0.9)
    scene = ProteinScene(width=256, height=256, msaa=msaa)
    scene.add(protein)
    scene.camera.frame(protein, aspect=1)
    scene.play(Reveal(scene.camera, protein.select(residues=(23, 30))), run_time=1)
    scene.play(SetOpacity(protein.select(residues=(1, 76)), 0.65), run_time=1)
    scene.play(Representation(protein, "cartoon"), run_time=1)
    with StudioRenderer(256, 256, msaa=msaa, look=StudioLook(effects="clean")) as renderer:
        images = [scene.render_frame(t, renderer=renderer) for t in (0, 1, 2, 2.5, 3)]
        assert renderer.front_mesh is not None  # Molecular mesh participates in transparent AO.
        for a, b in zip(images, images[1:]):
            assert np.abs(a.astype(int) - b).sum() > 1000
        for t, expected in zip((0, 1, 2, 2.5, 3), images):
            np.testing.assert_array_equal(scene.render_frame(t, renderer=renderer), expected)


@pytest.mark.gpu
def test_original_density_surface_remains_opt_in(protein):
    protein.surface(resolution=0.9)
    camera = Camera().frame(protein, aspect=1)
    with StudioRenderer(192, 192, look=StudioLook(surface_mode="density")) as renderer:
        image = renderer.render([protein], camera, (0, 0, 0))
        assert renderer._molecules[protein].surface is None
        assert protein in renderer._grids
        assert np.any(image[:, :, :3] > 10, axis=2).sum() > 100
