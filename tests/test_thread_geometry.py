"""Real geometry threading: rigid atom groups, eased travel, seeks and GPU drawing."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from proteinmotion import (
    Colorize,
    Deform,
    Protein,
    ProteinScene,
    Representation,
    StudioLook,
    Thread,
    Unthread,
    smooth,
)
from proteinmotion._thread_geometry import TransportPath


def bound(protein, cls=Thread, **kwargs):
    scene = ProteinScene(width=320, height=240)
    scene.camera.frame(protein, aspect=4 / 3)
    animation = cls(protein, scene.camera, aspect=4 / 3, **kwargs)
    animation.run_time, animation.start_time = 8, 0
    animation.bind()
    return animation


def test_transport_frames_are_orthonormal_and_extend_upstream():
    t = np.linspace(0, 3 * np.pi, 200)
    path = TransportPath(np.column_stack([5 * np.cos(t), 5 * np.sin(t), t]))
    points, frames = path.sample(np.linspace(-20, path.s[-1] + 5, 100))
    np.testing.assert_allclose(
        frames.transpose(0, 2, 1) @ frames, np.tile(np.eye(3), (100, 1, 1)), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(frames), 1, atol=1e-12)
    assert np.linalg.norm(points[0] - path.points[0]) == pytest.approx(20)


def test_actual_atoms_move_with_rigid_residues_and_ligands(structure_path):
    p = Protein.from_file(structure_path.parent / "2dri.cif", chains="A").center()
    xyz = p.positions.copy()
    animation = bound(p, seed=4)
    assert animation.easing is smooth
    animation.apply(0.45)
    assert p.opacity == 1  # Geometry is visible throughout, with no wire crossfade.
    assert not animation.wire.representation.any()
    assert not animation.wire._appearance[:, 16:18].any()
    assert np.linalg.norm(p.positions - xyz, axis=1).max() > 20
    for selection in (p.select(residues=13), p.select(resname="RIP")):
        ids = selection.atom_indices
        np.testing.assert_allclose(pdist(p.positions[ids]), pdist(xyz[ids]), atol=3e-5)
    for t in (0.1, 0.7, 0.94):
        animation.apply(t)
        expected = p.positions.copy()
        animation.apply(0.9)
        animation.apply(t)
        np.testing.assert_array_equal(p.positions, expected)
    animation.apply(1)
    np.testing.assert_array_equal(p.positions, xyz)


def test_eased_landing_slows_to_rest_without_overshoot(protein):
    animation = bound(protein)
    xyz = protein.positions.copy()
    path = animation.geometry.paths[0]
    direction = path.tangent[-1]
    head = animation.geometry.trace[animation.geometry.slices[0].stop - 1]
    values = []
    for t in np.linspace(0.97, 1.0, 31):
        animation.apply(t)
        values.append(np.dot(protein.positions[head] - xyz[head], direction))
    # The leading terminus approaches from behind and never bounces past rest.
    assert values[0] < -0.01 and max(values) <= 1e-5
    steps = np.diff(values)
    assert min(steps) >= -1e-5
    assert steps[-1] < steps[0] * 0.05
    np.testing.assert_array_equal(protein.positions, xyz)


def test_reverse_and_staggered_chains_are_deterministic(structure_path):
    p = Protein.from_file(structure_path.parent / "4hhb.cif").center()
    forward, backward = bound(p, seed=2, stagger=0.15), bound(p, Unthread, seed=2, stagger=0.15)
    xyz = p.positions.copy()
    for alpha in (0.0, 0.15, 0.45, 0.8, 1.0):
        forward.apply(alpha)
        positions, opacity = p.positions.copy(), p.opacity
        backward.apply(1 - alpha)
        np.testing.assert_allclose(p.positions, positions, atol=2e-5)
        assert p.opacity == opacity
    forward.apply(0.65)
    first = forward.geometry.slices[0]
    ids = np.flatnonzero(forward.geometry.owners < first.stop)
    np.testing.assert_array_equal(p.positions[ids], xyz[ids])
    assert not np.allclose(p.positions, xyz)
    backward.apply(1)
    assert p.opacity == 0


def test_thread_preserves_bound_visibility_and_restores_coordinate_state(protein):
    original = protein.positions.copy()
    protein._pair(original, original + [1, 0, 0], 0.4)
    controls = protein._controls.copy()
    controls[:, 4:6] = [0.2, 0.8]
    controls.flags.writeable = False
    protein._controls = controls
    state = protein.snapshot()
    alpha = protein.atom_opacities.copy()
    animation = bound(protein)
    animation.apply(0.5)
    np.testing.assert_allclose(protein.atom_opacities, alpha)
    animation.apply(1)
    assert protein._a is state["a"] and protein._b is state["b"]
    assert protein._mix == state["mix"] and protein._controls is controls


def test_surface_and_conflicting_animation_errors(protein):
    protein.surface()
    for mode in ("geometry", "wire"):
        with pytest.raises(ValueError, match="surface representation"):
            Thread(protein, mode=mode)
    protein.cartoon()
    animation = Thread(protein)
    protein.surface()
    with pytest.raises(ValueError, match="surface representation"):
        animation.bind()
    protein.cartoon()
    for other in (Deform(protein, lambda p: p + 1), Representation(protein, "surface")):
        with pytest.raises(ValueError, match="Concurrent animations"):
            ProteinScene().play(Thread(protein), other)
    for options in ({"mode": "bad"}, {"aspect": 0}):
        with pytest.raises(ValueError):
            Thread(protein, **options)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["native", "studio"])
@pytest.mark.parametrize("representation", ["cartoon", "ball_and_stick"])
def test_geometry_thread_gpu_seek_and_final_representation(protein, backend, representation):
    from proteinmotion.renderer import Renderer
    from proteinmotion.studio import StudioRenderer

    getattr(protein, representation)()
    scene = ProteinScene(width=320, height=240)
    scene.camera.frame(protein, aspect=4 / 3)
    scene.play(
        Thread(protein, scene.camera, aspect=4 / 3, glow=0),
        Colorize(protein.select(residues=(1, 76)), "#e8cda5"),
        run_time=8,
    )
    cls = Renderer if backend == "native" else StudioRenderer
    options = {} if backend == "native" else {"look": StudioLook(effects="clean")}
    with cls(320, 240, **options) as renderer:
        images = [scene.render_frame(t, renderer=renderer) for t in (0, 3.2, 5.6, 6.4, 8)]
        for a, b in zip(images, images[1:]):
            assert np.abs(a.astype(int) - b).sum() > 1000
        for t, expected in zip((0, 3.2, 5.6, 6.4, 8), images):
            np.testing.assert_array_equal(scene.render_frame(t, renderer=renderer), expected)
