"""C1′ contact matching, staggered whole-nucleotide motion, and rendered replay."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from proteinmotion import BackboneMorph, ContactMatch, Morph, NucleicAcid, Protein, Scene, match_backbones
from proteinmotion.matching import contact_map, fit_transform
from proteinmotion.nucleic import BaseGeometry
from proteinmotion.structure import Atom, Residue, Topology

DATA = Path(__file__).resolve().parents[1] / "examples/data"


def anchors(p, residues=None, name="C1'"):
    ids = range(len(p.topology.residues)) if residues is None else residues
    lookup = {(a.residue_index, a.name.replace("*", "'")): i for i, a in enumerate(p.topology.atoms)}
    return np.array([lookup[ri, name] for ri in ids if (ri, name) in lookup])


def world(p):
    m = p.model_matrix
    return p.positions @ m[:3, :3].T + m[:3, 3]


def artificial_dna(c1, seed):
    # C4′ positions deliberately contradict the C1′ correspondence.
    rng = np.random.default_rng(seed)
    atoms, residues, xyz = [], [], []
    for i, point in enumerate(c1):
        atoms.extend(Atom("A", i + 1, "", "DA", n, "C", i) for n in ("C1'", "C4'"))
        residues.append(
            Residue("A", i + 1, "", "DA", -1, -1, kind="dna", guide=2 * i, backbone=2 * i + 1, base="A")
        )
        xyz.extend((point, rng.normal(size=3) * 30))
    return NucleicAcid(Topology(tuple(atoms), tuple(residues), np.empty((0, 2), np.uint32), ()), xyz)


def test_c1_matching_ignores_c4_and_preserves_report(tmp_path):
    x = np.random.default_rng(23).normal(size=(6, 3)) * 7
    y = np.insert(x, 3, [30, -20, 40], axis=0)
    y = y @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + [3, 8, -2]
    a, b = artificial_dna(x, 3), artificial_dna(y, 29)
    match = match_backbones(a, b, max_contact_error=1e-6, max_candidates=None, search_seconds=2)
    np.testing.assert_array_equal(match.source_indices, np.arange(6))
    np.testing.assert_array_equal(match.target_indices, [0, 1, 2, 4, 5, 6])
    assert match.report["source_anchor_atom"] == match.report["target_anchor_atom"] == "C1'"
    assert match.report["source_anchor_count"] == 6 and match.report["target_anchor_count"] == 7
    assert match.report["aligned_anchor_rmsd_angstrom"] < 1e-5
    assert "source_ca_count" not in match.report
    assert match.report["global_optimal"]
    ca = contact_map(a.positions[anchors(a, match.source_indices)])
    cb = contact_map(b.positions[anchors(b, match.target_indices)])
    assert np.abs(ca - cb).max() <= 1e-6
    c4a = contact_map(a.positions[anchors(a, match.source_indices, "C4'")])
    c4b = contact_map(b.positions[anchors(b, match.target_indices, "C4'")])
    assert np.abs(c4a - c4b).max() > 0.01
    path = tmp_path / "dna-match.json"
    match.save(path)
    loaded = ContactMatch.load(path)
    assert loaded.report == match.report
    np.testing.assert_array_equal(loaded.target_indices, match.target_indices)
    # Legacy manual correspondences have no anchor metadata; infer from the input.
    legacy = replace(match, report={})
    BackboneMorph(a, b, match=legacy).bind()


@pytest.fixture
def dna_morph():
    a = NucleicAcid.from_file(DATA / "1bna.cif", chains="A").cartoon(bases="rings").center()
    b = NucleicAcid.from_file(DATA / "2dcg.cif", chains="A").cartoon(bases="slabs").center()
    a.rotate(0.3).scale(1.2).shift([2, 0, 3])
    b.rotate(-0.5).scale(0.8).shift([1, 4, -1])
    match = match_backbones(a, b, max_contact_error=0.2, max_candidates=None, search_seconds=2)
    anim = BackboneMorph(a, b, match=match, residue_delay=0.18)
    scene = Scene(width=384, height=256)
    scene.add(a)
    scene.camera.frame(a, b, aspect=1.5)
    scene.play(anim, run_time=4)
    return scene, a, b, anim


def test_c1_alignment_delays_fades_and_whole_residue_translation(dna_morph):
    scene, a, b, anim = dna_morph
    ai, bi = anim.match.source_indices, anim.match.target_indices
    ac, bc = anchors(a, ai), anchors(b, bi)
    np.testing.assert_array_equal(anim.source_anchor_indices, ac)
    np.testing.assert_array_equal(anim.target_anchor_indices, bc)
    scene.seek(0.1)
    assert a.atom_progress[ac[0]] > 0
    np.testing.assert_array_equal(a.atom_progress[ac[1:]], 0)
    scene.seek(2)
    np.testing.assert_allclose(world(a)[ac], world(b)[bc], atol=3e-5)
    assert np.linalg.norm(world(a)[anchors(a, ai, "C4'")] - world(b)[anchors(b, bi, "C4'")]) > 0.1
    for p, selected, initial in ((a, ai, anim.source_start), (b, bi, anim.destination_start)):
        residue_ids = np.array([atom.residue_index for atom in p.topology.atoms])
        for ri, anchor in zip(selected, anchors(p, selected)):
            ids = np.flatnonzero(residue_ids == ri)
            np.testing.assert_allclose(
                p.positions[ids] - p.positions[anchor], initial[ids] - initial[anchor], atol=2e-5
            )
            np.testing.assert_allclose(p.atom_progress[ids], p.atom_progress[anchor])
            np.testing.assert_allclose(p.atom_opacities[ids], p.atom_opacities[anchor])
        np.testing.assert_array_equal(p.atom_opacities[~np.isin(residue_ids, selected)], 0)
    np.testing.assert_allclose(a.atom_opacities[ac] + b.atom_opacities[bc], 1, atol=1e-6)
    snapshot = a.positions.copy(), b.positions.copy(), a.atom_opacities.copy()
    assert scene.seek(4) == [b]
    np.testing.assert_allclose(world(b)[bc], anim.target_world, atol=3e-5)
    np.testing.assert_array_equal(b.atom_opacities, 1)
    scene.seek(0)
    scene.seek(2)
    for actual, expected in zip((a.positions, b.positions, a.atom_opacities), snapshot):
        np.testing.assert_array_equal(actual, expected)
    # Base meshes also receive the delayed coordinates and per-residue visibility.
    for p in (a, b):
        g = BaseGeometry(p)
        i = int(np.argmax(p.base_style))
        vertices, faces, _, alpha, _ = g.evaluate(p, i)
        owners = g.parts[i][0][:, 9].copy().view(np.uint32)
        np.testing.assert_allclose(alpha, p.atom_opacities[owners][faces].min(axis=1), atol=1e-6)
        assert np.isfinite(vertices).all()


def test_same_topology_alignment_uses_c1(dna_morph):
    _, p, _, _ = dna_morph
    c1 = anchors(p)
    target = p.positions.copy() + [9, 4, -3]
    other = np.setdiff1d(np.arange(len(target)), c1)
    target[other] += np.random.default_rng(10).normal(size=(len(other), 3)) * 4
    original = p.positions.copy()
    r, t = fit_transform(target[c1], original[c1])
    scene = Scene().play(Morph(p, target), run_time=1)
    scene.seek(1)
    np.testing.assert_allclose(p.positions, target @ r + t, atol=2e-5)
    np.testing.assert_allclose(p.positions[c1], original[c1], atol=2e-5)


def test_missing_c1_and_incompatible_correspondence(dna_morph):
    _, a, b, anim = dna_morph
    # Keep C4′ present: it must never silently substitute for a missing C1′.
    ids = anim.match.source_indices
    residues = list(a.topology.residues)
    residues[ids[0]] = replace(residues[ids[0]], guide=-1)
    a.topology = replace(a.topology, residues=tuple(residues))
    with pytest.raises(ValueError, match="C1'"):
        BackboneMorph(a, b, match=anim.match).bind()
    match = match_backbones(a, b, max_contact_error=0.3, search_seconds=1)
    assert ids[0] not in match.source_indices
    assert match.report["source_anchor_count"] == 11
    protein = Protein.from_file(DATA / "1ubq.cif")
    with pytest.raises(ValueError, match="same morph anchor"):
        match_backbones(a, protein)
    with pytest.raises(ValueError, match="same morph anchor"):
        ContactMatch.from_pairs(a, protein, [0, 1, 2], [0, 1, 2])
    invalid = replace(match, report={**match.report, "source_anchor_atom": "CA"})
    with pytest.raises(ValueError, match="Saved source morph anchor"):
        BackboneMorph(a, b, match=invalid).bind()
    a.topology = replace(a.topology, residues=tuple(replace(r, guide=-1) for r in residues))
    with pytest.raises(ValueError, match="C1'"):
        match_backbones(a, b)
    with pytest.raises(ValueError, match="C1'"):
        Scene().play(Morph(a, a.positions))
    Scene().play(Morph(a, a.positions, align=False))


@pytest.mark.gpu
def test_dna_morph_gpu_replay_and_cached_base_meshes(dna_morph):
    import wgpu

    from proteinmotion.renderer import Renderer

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter")
    scene, a, b, anim = dna_morph
    with Renderer(384, 256) as renderer:
        first = scene.render_frame(0, renderer=renderer)
        middle = scene.render_frame(2, renderer=renderer)
        meshes = [m.bases.geometry for m in renderer._molecules.values()]
        scene.render_frame(2.5, renderer=renderer)
        assert [m.bases.geometry for m in renderer._molecules.values()] == meshes
        end = scene.render_frame(4, renderer=renderer)
        np.testing.assert_array_equal(scene.render_frame(2, renderer=renderer), middle)
        assert np.abs(first.astype(int) - middle).sum() > 5000
        assert np.abs(end.astype(int) - middle).sum() > 5000
        scene.seek(4)
        target = b.copy().set_positions(anim.destination_end)
        reference = renderer.render([target], scene.camera, scene.background)
        np.testing.assert_array_equal(end, reference)
