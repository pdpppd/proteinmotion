"""Independent correspondence oracle, delayed motion, visibility and GPU regression tests."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from proteinmotion import (
    BackboneMorph,
    ContactMatch,
    Morph,
    Protein,
    ProteinScene,
    Rotate,
    match_backbones,
    smooth,
)
from proteinmotion.matching import _CliqueSearch, contact_map, fit_transform
from proteinmotion.structure import Atom, Residue, Topology


def ca_protein(xyz):
    n = len(xyz)
    atoms = tuple(Atom("A", i + 1, "", "ALA", "CA", "C", i) for i in range(n))
    residues = tuple(Residue("A", i + 1, "", "ALA", i, -1, "H") for i in range(n))
    topology = Topology(atoms, residues, np.empty((0, 2), np.uint32), (np.arange(n, dtype=np.uint32),))
    return Protein(topology, xyz).ribbon(color="#62dab6")


def world(protein):
    m = protein.model_matrix
    return protein.positions @ m[:3, :3].T + m[:3, 3]


@pytest.mark.parametrize("seed", [2, 7, 19])
def test_clique_matches_independent_exhaustive_oracle(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=(5, 3)) * 5, rng.normal(size=(6, 3)) * 5
    ca, cb = contact_map(x), contact_map(y)
    tolerance = 0.28
    best_count, best_error = 0, np.inf
    # Enumerate every monotone matching; this does not use the solver's graph or pruning.
    for count in range(1, 6):
        for aa in combinations(range(5), count):
            for bb in combinations(range(6), count):
                diff = ca[np.ix_(aa, aa)] - cb[np.ix_(bb, bb)]
                if np.max(np.abs(diff)) > tolerance + 1e-10:
                    continue
                error = np.sum(diff[np.triu_indices(count, 1)] ** 2)
                if count > best_count or (count == best_count and error < best_error):
                    best_count, best_error = count, error
    pairs = np.array([(i, j) for i in range(5) for j in range(6)])
    solver = _CliqueSearch(pairs, ca, cb, tolerance, [], 10)
    selected = solver.solve()
    assert solver.completed
    assert len(selected) == best_count
    assert solver.best_error == pytest.approx(best_error, abs=1e-12)
    assert (np.diff(selected, axis=0) > 0).all()


def test_matching_insertion_rigid_invariance_and_save(tmp_path):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(6, 3)) * 7
    y = np.insert(x, 3, [30, -20, 40], axis=0)
    y = y @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + [3, 8, -2]
    a, b = ca_protein(x), ca_protein(y)
    result = match_backbones(a, b, max_contact_error=1e-6, max_candidates=None, search_seconds=2)
    np.testing.assert_array_equal(result.source_indices, np.arange(6))
    np.testing.assert_array_equal(result.target_indices, [0, 1, 2, 4, 5, 6])
    assert result.report["global_optimal"]
    assert result.report["aligned_ca_rmsd_angstrom"] < 1e-5
    assert result.report["contact_max_error"] < 1e-6
    path = tmp_path / "match.json"
    result.save(path)
    restored = ContactMatch.load(path)
    np.testing.assert_array_equal(restored.target_indices, result.target_indices)
    assert restored.report == result.report
    assert not restored.source_indices.flags.writeable
    r, _ = fit_transform(x * [-1, 1, 1], x)
    assert np.linalg.det(r) == pytest.approx(1)


def example_morph():
    t, u = np.arange(7), np.arange(9)
    a = ca_protein(np.column_stack((3 * t, 2 * np.sin(t), np.cos(t))))
    b = ca_protein(np.column_stack((2.5 * u, 4 * np.sin(u * 0.6), 3 * np.cos(u * 0.4))))
    a.rotate(0.3).scale(1.2).shift([0, 1, 3])
    b.rotate(-0.5).scale(0.8).shift([2, 4, -1])
    pairs = ContactMatch.from_pairs(a, b, [0, 2, 4, 6], [1, 3, 5, 7])
    animation = BackboneMorph(a, b, match=pairs, residue_delay=0.3, align=False)
    scene = ProteinScene(width=384, height=256)
    scene.add(a)
    scene.camera.frame(a, b, aspect=1.5)
    scene.wait(0.5)
    scene.play(animation, run_time=2)
    return scene, a, b, animation


def test_delay_seconds_fades_world_positions_and_backward_seek():
    scene, a, b, anim = example_morph()
    ia, ib = anim.match.source_indices, anim.match.target_indices
    assert anim.move_time == pytest.approx(1.1)
    np.testing.assert_allclose(anim.delay_schedule, [0, 0.3, 0.6, 0.9])
    assert scene.seek(0.25) == [a]
    scene.seek(0.6)  # 0.1 seconds into the morph: only first selected residue has started.
    assert a.atom_progress[ia[0]] > 0
    np.testing.assert_array_equal(a.atom_progress[ia[1:]], 0)
    for elapsed in [0.1, 0.7, 1.0, 1.9]:
        scene.seek(0.5 + elapsed)
        expected_progress = np.array(
            [smooth(np.clip((elapsed - delay) / 1.1, 0, 1)) for delay in anim.delay_schedule]
        )
        expected_world = anim.source_world + expected_progress[:, None] * (
            anim.target_world - anim.source_world
        )
        np.testing.assert_allclose(world(a)[ia], expected_world, atol=5e-5)
        np.testing.assert_allclose(world(a)[ia], world(b)[ib], atol=5e-5)
        np.testing.assert_allclose(a.atom_opacities[ia] + b.atom_opacities[ib], 1, atol=1e-6)
    scene.seek(1.5)
    assert a.atom_opacities[1] == 0  # Unmatched source faded by 35% of clip.
    assert b.atom_opacities[0] == 0  # Unmatched destination waits until 65%.
    midpoint = world(a).copy(), world(b).copy(), a.atom_opacities.copy()
    assert scene.seek(2.5) == [b]
    np.testing.assert_allclose(world(b)[ib], anim.target_world, atol=2e-5)
    np.testing.assert_array_equal(b.atom_opacities, 1)
    scene.seek(0.5)
    scene.seek(1.5)
    for actual, expected in zip((world(a), world(b), a.atom_opacities), midpoint):
        np.testing.assert_array_equal(actual, expected)


def test_next_morph_uses_normal_timing_and_visibility():
    scene, a, b, anim = example_morph()
    original = b.positions.copy()
    scene.play(Morph(b, original + [0, 6, 0], align=False), run_time=2)
    scene.seek(3.5)
    np.testing.assert_allclose(b.positions, original + [0, 3, 0], atol=2e-5)
    np.testing.assert_array_equal(b.atom_opacities, 1)


def test_invalid_correspondences_timing_and_channels():
    _, a, b, anim = example_morph()
    with pytest.raises(ValueError, match="different"):
        BackboneMorph(a, a)
    with pytest.raises(ValueError, match="increase strictly"):
        BackboneMorph(a, b, match=ContactMatch.from_pairs(a, b, [0, 2, 4], [3, 2, 1])).bind()
    with pytest.raises(ValueError, match="run_time must exceed"):
        ProteinScene().play(BackboneMorph(a, b, match=anim.match, residue_delay=1), run_time=2)
    with pytest.raises(ValueError, match="linear clip clock"):
        ProteinScene().play(BackboneMorph(a, b, match=anim.match), rate_func=smooth)
    with pytest.raises(ValueError, match="transform"):
        ProteinScene().play(BackboneMorph(a, b, match=anim.match), Rotate(b, 1))
    with pytest.raises(ValueError, match="max_contact_error"):
        match_backbones(a, b, max_contact_error=-0.1)


@pytest.mark.gpu
def test_gpu_backbone_endpoints_seek_and_static_buffers():
    import wgpu

    from proteinmotion.renderer import Renderer

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter")
    scene, a, b, anim = example_morph()
    with Renderer(384, 256) as renderer:
        source = scene.render_frame(0.25, renderer=renderer)
        at_start = scene.render_frame(0.5, renderer=renderer)
        np.testing.assert_array_equal(source, at_start)
        middle = scene.render_frame(1.5, renderer=renderer)
        cached_keys = [(tuple(m.keys), m.controls_key) for m in renderer._molecules.values()]
        scene.render_frame(1.9, renderer=renderer)
        assert [(tuple(m.keys), m.controls_key) for m in renderer._molecules.values()] == cached_keys
        end = scene.render_frame(2.5, renderer=renderer)
        assert np.abs(source.astype(int) - middle).sum() > 5000
        assert np.abs(middle.astype(int) - end).sum() > 5000
        np.testing.assert_array_equal(scene.render_frame(1.5, renderer=renderer), middle)
        fresh_target = b.copy().set_positions(anim.destination_end)
        fresh_target._controls = anim.destination_final_controls
        fresh_target.opacity = 1
        reference = renderer.render([fresh_target], scene.camera, scene.background)
        # Sweep-frame reference orientation can differ, but final coverage must agree.
        fg_end = np.max(np.abs(end[:, :, :3].astype(int) - end[0, 0, :3]), axis=2) > 15
        fg_ref = np.max(np.abs(reference[:, :, :3].astype(int) - reference[0, 0, :3]), axis=2) > 15
        assert (fg_end & fg_ref).sum() / (fg_end | fg_ref).sum() > 0.97


@pytest.fixture
def atom_morph_scene():
    data = Path(__file__).resolve().parents[1] / "examples/data"
    a = Protein.from_file(data / "1cll.cif", chains="A").ball_and_stick().center()
    b = Protein.from_file(data / "1ncx.cif", chains="A").ball_and_stick().center()
    match = ContactMatch.load(data / "calmodulin-troponin-match.json")
    original = a.copy()
    scene = ProteinScene(width=384, height=256)
    scene.add(a)
    scene.camera.frame(a, aspect=1.5)
    animation = BackboneMorph(a, b, match=match, residue_delay=0.025)
    scene.play(animation, run_time=6)
    return scene, a, b, animation, original


def test_ball_and_stick_translates_whole_residues_and_fades(atom_morph_scene):
    scene, a, b, anim, original = atom_morph_scene
    scene.seek(3)
    for protein, ids, start in [
        (a, anim.match.source_indices, anim.source_start),
        (b, anim.match.target_indices, anim.destination_start),
    ]:
        atom_residues = np.array([atom.residue_index for atom in protein.topology.atoms])
        for residue in ids:
            ca = protein.topology.residues[residue].ca
            atoms = np.flatnonzero(atom_residues == residue)
            np.testing.assert_allclose(
                protein.positions[atoms] - protein.positions[ca], start[atoms] - start[ca], atol=2e-5
            )
            np.testing.assert_allclose(protein.atom_progress[atoms], protein.atom_progress[ca])
            np.testing.assert_allclose(protein.atom_opacities[atoms], protein.atom_opacities[ca])
        unmatched = ~np.isin(atom_residues, ids)
        np.testing.assert_array_equal(protein.atom_opacities[unmatched], 0)
    source_ca = [a.topology.residues[i].ca for i in anim.match.source_indices]
    target_ca = [b.topology.residues[i].ca for i in anim.match.target_indices]
    np.testing.assert_allclose(world(a)[source_ca], world(b)[target_ca], atol=3e-5)
    assert scene.seek(6) == [b]
    np.testing.assert_allclose(b.positions, anim.destination_end, atol=1e-6)
    np.testing.assert_array_equal(b.atom_opacities, 1)
    assert scene.seek(0) == [a]
    np.testing.assert_array_equal(a.positions, original.positions)


@pytest.mark.gpu
def test_ball_and_stick_morph_gpu_endpoints_and_replay(atom_morph_scene):
    import wgpu

    from proteinmotion.renderer import Renderer

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter")
    scene, a, b, anim, original = atom_morph_scene
    with Renderer(384, 256) as renderer:
        start = scene.render_frame(0, renderer=renderer)
        reference = renderer.render([original], scene.camera, scene.background)
        np.testing.assert_array_equal(start, reference)
        middle = scene.render_frame(3, renderer=renderer)
        end = scene.render_frame(6, renderer=renderer)
        target = b.copy().set_positions(anim.destination_end)
        expected = renderer.render([target], scene.camera, scene.background)
        np.testing.assert_array_equal(end, expected)
        np.testing.assert_array_equal(scene.render_frame(3, renderer=renderer), middle)
        assert np.abs(start.astype(int) - middle).sum() > 10000
        assert np.abs(end.astype(int) - middle).sum() > 10000


@pytest.mark.gpu
def test_bond_is_hidden_when_either_endpoint_is_hidden():
    import wgpu

    from proteinmotion.camera import Camera
    from proteinmotion.renderer import Renderer

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter")
    p = ca_protein([[-4, 0, 0], [4, 0, 0]]).ball_and_stick()
    t = p.topology
    p.topology = Topology(t.atoms, t.residues, np.array([[0, 1]], np.uint32), t.chains)
    controls = p._controls.copy()
    controls[1, 4:6] = 0
    controls.flags.writeable = False
    p._controls = controls
    camera = Camera().frame(p, aspect=1.5)
    reference = p.copy()
    reference.topology = Topology(t.atoms, t.residues, np.empty((0, 2), np.uint32), t.chains)
    with Renderer(384, 256) as renderer:
        actual = renderer.render([p], camera, np.array([0.03, 0.05, 0.08]))
        expected = renderer.render([reference], camera, np.array([0.03, 0.05, 0.08]))
        np.testing.assert_array_equal(actual, expected)
