import numpy as np
import pytest

from proteinmotion import Colorize, PlayTrajectory, Protein, Representation, Scene, SetOpacity, Trajectory
from proteinmotion.structure import Atom, Residue, Topology
from proteinmotion.styling import current_tints
from proteinmotion.surface import build_surface, surface_options


def single_atom():
    return Protein(
        Topology(
            (Atom("A", 1, "", "ALA", "CA", "C", 0),),
            (Residue("A", 1, "", "ALA", 0, -1),),
            np.empty((0, 2), np.uint32),
            (),
        ),
        [[0, 0, 0]],
    )


def test_residue_colors_opacity_stagger_channels_and_seek(protein):
    a, b, c = [protein.select(residues=i) for i in (8, 44, 70)]
    a.set_color("#ff0000")
    b.set_opacity(0.5)
    scene = Scene()
    scene.add(protein)
    scene.wait(1)
    scene.play(
        Colorize(a | b, "#00ff00", residue_delay=0.5, easing="linear"),
        Colorize(c, "#0000ff"),
        SetOpacity(b, 0, residue_delay=0.1, easing="linear"),
        run_time=2,
    )
    scene.seek(1.5)
    colors = current_tints(protein)
    np.testing.assert_allclose(
        colors[a.atom_indices, :3], np.tile([2 / 3, 1 / 3, 0], (len(a.atom_indices), 1)), atol=1e-6
    )
    assert not colors[b.atom_indices, 3].any()  # Its delay has just ended.
    np.testing.assert_allclose(protein.atom_opacities[b.atom_indices], 0.375)
    scene.seek(3)
    expected = current_tints(protein).copy()
    np.testing.assert_allclose(
        current_tints(protein)[c.atom_indices, :3], np.tile([0, 0, 1], (len(c.atom_indices), 1))
    )
    scene.play(Colorize(a, None), run_time=1)
    scene.seek(4)
    assert not current_tints(protein)[a.atom_indices].any()
    assert (current_tints(protein)[c.atom_indices, 2] == 1).all()
    scene.seek(1.8)
    scene.seek(3)
    np.testing.assert_array_equal(current_tints(protein), expected)
    with pytest.raises(ValueError, match="color"):
        Scene().play(Colorize(a, "#ff0000"), Colorize(a, "#00ff00"))
    with pytest.raises(ValueError, match="delay"):
        Scene().play(Colorize(a | b | c, "#ffffff", residue_delay=1), run_time=1)


def test_style_opacity_composes_with_morph_visibility_and_global_fade(protein):
    protein.set_opacity(0.5).set_residue_opacity(0.4, residues=8)
    controls = protein._controls.copy()
    controls[:, 4:6] = 0.3
    controls.flags.writeable = False
    protein._controls = controls
    np.testing.assert_allclose(
        protein.atom_opacities[protein.select(residues=8).atom_indices], 0.06, atol=1e-7
    )
    np.testing.assert_allclose(
        protein.atom_opacities[protein.select(residues=9).atom_indices], 0.15, atol=1e-7
    )
    with pytest.raises(ValueError):
        protein.select(residues=8).set_color("wrong")
    with pytest.raises(ValueError):
        protein.set_residue_opacity(float("nan"))


@pytest.mark.parametrize("kind,expected", [("vdw", 1.7), ("sas", 3.1), ("ses", 1.7)])
def test_single_atom_surface_radius_normals_and_closed_mesh(kind, expected):
    mesh = build_surface(single_atom(), surface_options(kind=kind, resolution=0.25))
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.mean(radii) == pytest.approx(expected, abs=0.16)
    assert np.mean(np.sum(mesh.normals * mesh.vertices, axis=1) > 0) > 0.99
    edges = np.sort(
        np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    np.testing.assert_allclose(mesh.weights.sum(1), 1)
    assert (mesh.owners == 0).all()


def test_surface_limits_and_representation_seek(protein):
    with pytest.raises(ValueError, match="max_voxels"):
        build_surface(protein, surface_options(resolution=0.05, max_voxels=1000))
    with pytest.raises(ValueError):
        surface_options(kind="wrong")
    scene = Scene()
    scene.add(protein)
    scene.play(Representation(protein, "surface"), run_time=2)
    scene.play(Representation(protein, "ribbon"), run_time=2)
    scene.seek(1)
    assert protein.surface_opacity == 0.5
    assert protein._surface_options.reference is not None
    scene.seek(3)
    assert protein.surface_opacity == 0.5
    np.testing.assert_allclose(protein.representation, [0, 0.5, 0])
    scene.seek(0)
    assert protein.surface_opacity == 0


@pytest.mark.gpu
@pytest.mark.parametrize("representation", ["cartoon", "ribbon", "ball_and_stick", "surface"])
def test_gpu_residue_color_opacity_all_representations(protein, representation):
    from proteinmotion.renderer import Renderer

    getattr(protein, representation)()
    region = protein.select(residues=(1, 76))
    scene = Scene(width=384, height=256)
    scene.add(protein)
    scene.camera.frame(protein, aspect=1.5)
    scene.play(Colorize(region, "#ff0000", easing="linear"), run_time=1)
    scene.play(SetOpacity(region, 0.2), run_time=1)
    # Single-sample pixels isolate color interpolation from MSAA edge resolve
    # (whose individual samples can clip even when the resolved pixel does not).
    with Renderer(384, 256, msaa=1) as renderer:
        before = scene.render_frame(0, renderer=renderer)
        halfway = scene.render_frame(0.5, renderer=renderer)
        red = scene.render_frame(1, renderer=renderer)
        faded = scene.render_frame(2, renderer=renderer)
        assert np.abs(red.astype(int) - before.astype(int)).sum() > 10000
        assert red[:, :, 0].sum() > before[:, :, 0].sum()
        # Away from output clipping, a linear tint must be halfway between
        # palette and override, even when fading from no existing override.
        a, b, mid = [im[:, :, :3].astype(float) for im in (before, red, halfway)]
        unclipped = (a < 250) & (b < 250) & (np.abs(a - b) > 10)
        assert unclipped.sum() > 100
        assert np.max(np.abs(mid - (a + b) / 2)[unclipped]) <= 1
        assert np.abs(faded.astype(int) - red.astype(int)).sum() > 10000
        for t, image in [(1, red), (0, before), (2, faded)]:
            np.testing.assert_array_equal(scene.render_frame(t, renderer=renderer), image)


@pytest.mark.gpu
def test_gpu_surface_deformation_cached_mesh_and_resurfacing(protein):
    from proteinmotion.renderer import Renderer

    protein.surface(resolution=1, update="deform")
    scene = Scene(width=256, height=256)
    scene.add(protein)
    scene.camera.frame(protein, aspect=1)
    frames = np.array([protein.positions, protein.positions * np.array([1.15, 0.85, 1.1])])
    scene.play(PlayTrajectory(protein, Trajectory(frames)), run_time=2)
    with Renderer(256, 256) as renderer:
        first = scene.render_frame(0, renderer=renderer)
        mesh = renderer._molecules[protein].surface
        buffer = mesh.vertices
        last = scene.render_frame(2, renderer=renderer)
        assert mesh.vertices is buffer
        assert np.abs(first.astype(int) - last.astype(int)).sum() > 5000
        np.testing.assert_array_equal(scene.render_frame(0, renderer=renderer), first)
    protein.surface(resolution=1, update="rebuild")
    with Renderer(256, 256) as renderer:
        renderer.render([protein], scene.camera, scene.background)
        original = renderer._molecules[protein].surface.mesh
        protein.set_positions(frames[1])
        renderer.render([protein], scene.camera, scene.background)
        assert renderer._molecules[protein].surface.mesh is not original
