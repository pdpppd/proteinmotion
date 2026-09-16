"""Live selections, camera focus, 3D annotations and an actual deposited NMR ensemble."""

from pathlib import Path

import numpy as np
import pytest

from proteinmotion import (
    FadeIn,
    Focus,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Region,
    Trajectory,
    linear,
    smooth,
)
from proteinmotion.camera import Camera


def test_selection_pdb_numbers_atom_names_and_union(protein):
    helix = protein.select(chain="A", residues=(23, 34))
    assert [protein.topology.residues[i].resid for i in helix.residue_indices] == list(range(23, 35))
    patch = protein.select(residues=[8, 44, 70], atoms="CA")
    assert len(patch.atom_indices) == 3
    assert all(protein.topology.atoms[i].name == "CA" for i in patch.atom_indices)
    combined = helix | patch
    np.testing.assert_array_equal(combined.atom_indices, np.union1d(helix.atom_indices, patch.atom_indices))
    assert not combined.atom_indices.flags.writeable
    with pytest.raises(ValueError, match="no atoms"):
        protein.select(chain="missing")
    with pytest.raises(ValueError, match="inclusive"):
        protein.select(residues=(34, 23))
    with pytest.raises(ValueError, match="integer"):
        Region(protein, [1.5])
    with pytest.raises(ValueError, match="same Protein"):
        helix | protein.copy().select(residues=23)
    before = helix.world_positions.copy()
    protein.shift([2, -3, 4])
    np.testing.assert_allclose(helix.world_positions, before + [2, -3, 4], atol=1e-6)


@pytest.mark.parametrize("style", ["sphere", "box", "atoms"])
def test_highlight_follows_deformation_transform_and_seek(protein, style):
    region = protein.select(residues=(23, 34))
    highlight = region.highlight(style=style, padding=0.8)
    scene = ProteinScene()
    scene.add(protein)
    scene.camera.frame(protein)
    scene.play(FadeIn(highlight), run_time=0.5)
    frames = np.array([protein.positions, protein.positions * np.array([1.2, 0.8, 1.1])])
    scene.play(
        PlayTrajectory(protein, Trajectory(frames)), protein.animate.rotate(0.8).shift([1, 3, -2]), run_time=2
    )
    scene.seek(1.25)
    np.testing.assert_array_equal(highlight.model_matrix, protein.model_matrix)
    positions = region.positions
    if style == "sphere":
        center = highlight.positions[0]
        radius = highlight._metadata_override[0, 3]
        assert radius >= np.linalg.norm(positions - center, axis=1).max() + 0.8 - 1e-5
    elif style == "box":
        np.testing.assert_allclose(highlight.positions.min(0), positions.min(0) - 0.8, atol=1e-5)
        np.testing.assert_allclose(highlight.positions.max(0), positions.max(0) + 0.8, atol=1e-5)
        assert len(highlight.topology.bonds) == 12
    else:
        np.testing.assert_array_equal(highlight.positions, positions)
    expected = highlight.positions.copy(), highlight.model_matrix.copy(), highlight._metadata_override.copy()
    scene.seek(2.5)
    scene.seek(0)
    scene.seek(1.25)
    for actual, reference in zip(
        (highlight.positions, highlight.model_matrix, highlight._metadata_override), expected
    ):
        np.testing.assert_array_equal(actual, reference)


def test_focus_evaluates_after_motion_and_tracks_later_clips(protein):
    region = protein.select(residues=(23, 34))
    scene = ProteinScene(width=400, height=600)
    scene.add(protein)
    scene.camera.frame(protein, aspect=2 / 3)
    initial = scene.camera.target.copy()
    scene.play(Focus(scene.camera, region, aspect=2 / 3), protein.animate.shift([4, 2, -1]), run_time=2)
    scene.play(protein.animate.shift([1, 3, 2]), run_time=2)
    scene.seek(1)
    points = region.world_positions
    np.testing.assert_allclose(scene.camera.target, 0.5 * initial + 0.5 * (points.min(0) + points.max(0)) / 2)
    scene.seek(3)
    points = region.world_positions
    np.testing.assert_allclose(scene.camera.target, (points.min(0) + points.max(0)) / 2)
    expected = scene.camera.target.copy(), scene.camera.distance
    scene.seek(0.4)
    scene.seek(4)
    scene.seek(3)
    np.testing.assert_array_equal(scene.camera.target, expected[0])
    assert scene.camera.distance == expected[1]
    assert scene.camera.distance < Camera().frame(protein, aspect=2 / 3).distance


def test_focus_conveniences_static_option_and_conflicts(protein):
    region = protein.select(residues=(45, 55))
    camera = Camera().focus(region, follow=False)
    old = camera.target.copy()
    protein.shift([5, 0, 0])
    camera.update_tracking()
    np.testing.assert_array_equal(camera.target, old)
    scene = ProteinScene()
    scene.add(protein)
    scene.camera.fov = np.deg2rad(55)
    scene.camera.frame(protein)
    scene.focus(region, run_time=1)
    fitted = Camera()
    fitted.fov = scene.camera.fov
    fitted.frame(region)
    assert scene.camera.distance == pytest.approx(fitted.distance)
    scene.play(scene.camera.animate.focus(protein), run_time=1)
    scene.seek(2)
    points = protein.select().world_positions
    np.testing.assert_allclose(scene.camera.target, (points.min(0) + points.max(0)) / 2)
    with pytest.raises(ValueError, match="camera"):
        scene.play(Focus(scene.camera, region), scene.camera.animate.orbit(0.3))


def test_fractional_focus_boundary_and_easing_stay_in_range(protein):
    for t in [np.nextafter(1.0, 0.0), 0.999999, 0.7, 0.5, 0.0]:
        assert 0 <= smooth(t) <= 1
    scene = ProteinScene()
    scene.add(protein)
    scene.camera.frame(protein)
    scene.wait(5.3)
    scene.focus(protein.select(residues=(23, 34)), run_time=1.8)
    scene.seek(7.1)  # This decimal boundary evaluates to alpha=0.9999999999999999.


def test_state_easing_preserves_models_and_eases_between_them(protein):
    frames = np.array([protein.positions + [i * 4, 0, 0] for i in range(3)])
    scene = ProteinScene()
    scene.add(protein)
    scene.play(PlayTrajectory(protein, Trajectory(frames), state_easing=smooth), run_time=2)
    scene.seek(0.25)
    np.testing.assert_allclose(protein.positions, frames[0] + [4 * smooth(0.25), 0, 0], atol=1e-5)
    for model in range(3):
        scene.seek(model)
        np.testing.assert_array_equal(protein.positions, frames[model].astype(np.float32))
    bad = PlayTrajectory(protein, Trajectory(frames), state_easing=lambda t: 2)
    with pytest.raises(ValueError, match="state_easing"):
        bad.apply(0.25)


@pytest.fixture(scope="module")
def nmr():
    path = Path(__file__).resolve().parents[1] / "examples/data/2k39.cif"
    return Protein.from_file(path, chains="A")


def test_real_nmr_ensemble_identity_alignment_and_endpoints(nmr):
    assert len(nmr.trajectory) == 116
    assert len(nmr.topology.atoms) == 602
    assert len(nmr.topology.residues) == 76
    assert {r.secondary for r in nmr.topology.residues} == {"H", "E", "C"}
    indices = nmr.select(residues=(1, 70), atoms="CA").atom_indices
    aligned = nmr.trajectory.aligned(indices=indices)
    assert len(aligned) == 116
    assert np.isfinite(np.array([aligned.frame(i) for i in range(116)])).all()
    p = Protein.from_trajectory(aligned)
    scene = ProteinScene()
    scene.add(p)
    anim = PlayTrajectory(p, state_easing=smooth)
    scene.play(anim, run_time=115, rate_func=linear)
    for i in [0, 13, 78, 115, 13]:
        scene.seek(i)
        np.testing.assert_array_equal(p.positions, aligned.frame(i))
    assert len(anim.cache.cache) <= 3
    assert np.sqrt(np.mean((aligned.frame(115) - aligned.frame(0)) ** 2)) > 0.1


@pytest.mark.gpu
@pytest.mark.parametrize("style", ["sphere", "box", "atoms"])
def test_gpu_region_highlight_is_visible_and_reproducible(protein, style):
    import wgpu

    from proteinmotion.renderer import Renderer

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter")
    scene = ProteinScene(width=384, height=256)
    scene.add(protein)
    scene.camera.frame(protein, aspect=1.5)
    region = protein.select(residues=(23, 34))
    highlight = region.highlight(style=style)
    scene.wait(0.5)
    scene.play(FadeIn(highlight), run_time=0.5)
    frames = np.array([protein.positions, protein.positions * [1.25, 0.8, 1.15]])
    scene.play(protein.animate.rotate(0.5), PlayTrajectory(protein, Trajectory(frames)), run_time=1)
    with Renderer(384, 256) as renderer:
        original = scene.render_frame(0, renderer=renderer)
        annotated = scene.render_frame(1, renderer=renderer)
        assert np.abs(original.astype(int) - annotated).sum() > 5000
        scene.render_frame(2, renderer=renderer)
        np.testing.assert_array_equal(scene.render_frame(1, renderer=renderer), annotated)
        times = [1.1, 1.3, 1.5, 1.7, 1.9]
        images = {t: scene.render_frame(t, renderer=renderer) for t in times}
        for t in times[::-1] + times:
            np.testing.assert_array_equal(scene.render_frame(t, renderer=renderer), images[t])
