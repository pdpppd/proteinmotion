"""Morphs between experimental structures, assemblies, downloads, and look_along."""

from pathlib import Path

import gemmi
import numpy as np
import pytest

from proteinmotion import Protein, ProteinScene, StructureMorph, domain_motion, match_structures
from proteinmotion.camera import Camera
from proteinmotion.math3d import align_coordinates, rotation

DATA = Path(__file__).resolve().parents[1] / "examples/data"


def hinged_ubiquitin(path, tmp_path, angle=40):
    """1UBQ with residues 40–76 turned rigidly about an axis through Cα 39, minus residue 76."""
    st = gemmi.read_structure(str(path))
    st.remove_ligands_and_waters()
    chain = st[0]["A"]
    hinge = np.array(chain[38]["CA"][0].pos.tolist())
    turn = rotation(np.radians(angle), (0.3, 1.0, 0.2))
    for residue in chain:
        if residue.seqid.num >= 40:
            for atom in residue:
                x = turn @ (np.array(atom.pos.tolist()) - hinge) + hinge
                atom.pos = gemmi.Position(*x)
    del chain[75]  # residue 76
    out = tmp_path / "hinged.pdb"
    st.setup_entities()
    st.write_pdb(str(out))
    return out


def test_matching_pairs_residues_atoms_and_reports(structure_path, tmp_path):
    source = Protein.from_file(structure_path)
    target = Protein.from_file(hinged_ubiquitin(structure_path, tmp_path))
    match = match_structures(source, target)
    assert match.report["chains"] == [("A", "A", 1.0)]
    assert match.report["matched_residues"] == 75
    names = [source.topology.atoms[a].name for a in match.source_atoms]
    targets = [target.topology.atoms[b].name for b in match.target_atoms]
    assert names == targets
    motion = domain_motion(
        source, target, moving=source.select(residues=(45, 70)), fixed=source.select(residues=(1, 38))
    )
    assert motion.angle == pytest.approx(40, abs=0.1)
    assert abs(
        np.dot(motion.axis, np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2]))
    ) == pytest.approx(1, abs=1e-3)


def test_structure_morph_moves_rigid_domains_along_arcs(structure_path, tmp_path):
    source = Protein.from_file(structure_path).cartoon()
    target = Protein.from_file(hinged_ubiquitin(structure_path, tmp_path)).cartoon()
    domain = source.select(residues=(45, 70), atoms="CA").atom_indices
    start = source.positions[domain].astype(float)
    scene = ProteinScene(width=64, height=36).add(source)
    morph = StructureMorph(source, target, align=source.select(residues=(1, 38)))
    scene.play(morph, run_time=2)
    scene.seek(1)
    middle = source.positions[domain].astype(float)
    # The domain stays rigid halfway, turned by half the angle.
    np.testing.assert_allclose(
        np.linalg.norm(middle[:, None] - middle[None], axis=2),
        np.linalg.norm(start[:, None] - start[None], axis=2),
        atol=0.02,
    )
    fitted = align_coordinates(start, middle)
    assert np.sqrt(((fitted - middle) ** 2).sum(1).mean()) < 0.02
    halfway = Protein.from_file(structure_path)
    halfway.set_positions(source.positions)
    turned = domain_motion(
        Protein.from_file(structure_path),
        halfway,
        moving=halfway.select(residues=(45, 70)),
        fixed=halfway.select(residues=(1, 38)),
    )
    assert turned.angle == pytest.approx(20, abs=0.5)  # half of 40° with smooth easing at the midpoint
    half = domain_motion(source, source.copy(), moving=source.select(residues=(45, 70)))
    assert half.angle < 1e-3  # sanity: no motion between identical copies
    scene.seek(2)
    match = morph.match_result
    np.testing.assert_allclose(
        source.positions[match.source_atoms], target.positions[match.target_atoms], atol=1e-4
    )
    assert source.opacity == 0 and target.opacity == 1
    assert morph.result is target
    # Residue 76 exists only in the source and has faded out by the end.
    gone = source.select(residues=76).atom_indices
    scene.seek(1.9)
    assert np.all(source.atom_opacities[gone] < 0.01)
    scene.seek(0)
    np.testing.assert_allclose(source.positions[domain], start, atol=1e-5)


def test_side_chains_turn_and_bonds_hold_through_the_morph():
    source = Protein.from_file(DATA / "4ake.cif", chains="A").cartoon()
    target = Protein.from_file(DATA / "1ake.cif", chains="A").cartoon()
    bonds = source.topology.bonds
    before = np.linalg.norm(source.positions[bonds[:, 0]] - source.positions[bonds[:, 1]], axis=1)
    scene = ProteinScene(width=64, height=36).add(source)
    morph = StructureMorph(source, target)
    scene.play(morph, run_time=2)
    scene.seek(1)
    middle = np.linalg.norm(source.positions[bonds[:, 0]] - source.positions[bonds[:, 1]], axis=1)
    assert np.abs(middle - before).max() < 0.2
    assert np.percentile(np.abs(middle - before), 99) < 0.06
    scene.seek(2)
    assert target.opacity == 1
    ap5a = target.select(resname="AP5").atom_indices
    assert np.all(target.atom_opacities[ap5a] == 1)  # present only in 1AKE: faded in
    scene.seek(1.2)
    assert np.all(target.atom_opacities[ap5a] < 1)


def test_assemblies_identical_copies_and_secondary_structure():
    deoxy = Protein.from_file(DATA / "2dn2.cif")
    oxy = Protein.from_file(DATA / "2dn1.cif", assembly="1")
    assert sorted({a.chain for a in oxy.topology.atoms}) == ["A", "B", "C", "D"]
    states = oxy.secondary_structure
    by_chain = {c: "".join(s for s, r in zip(states, oxy.topology.residues) if r.chain == c) for c in "AC"}
    assert by_chain["A"] == by_chain["C"]  # copied helix records
    with pytest.raises(ValueError, match="Assembly"):
        Protein.from_file(DATA / "2dn1.cif", assembly="7")
    match = match_structures(deoxy, oxy)
    assert [(a, b) for a, b, _ in match.chain_pairs] == [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")]
    hemes = [i for i, j in match.residue_pairs if deoxy.topology.residues[i].name == "HEM"]
    assert len(hemes) == 4
    motion = domain_motion(
        deoxy, oxy, moving=deoxy.select(chain=["C", "D"]), fixed=deoxy.select(chain=["A", "B"])
    )
    assert 12 < motion.angle < 16  # the α2β2 rotation of the T → R transition


def test_fetch_uses_the_cache_and_parses_identifiers(tmp_path, monkeypatch, structure_path):
    import proteinmotion.download as download

    calls = []

    def fake(url, destination):
        calls.append(url)
        destination.write_bytes(Path(structure_path).read_bytes())
        return destination

    monkeypatch.setattr(download, "_download", fake)
    first = Protein.fetch("1ubq", cache=tmp_path, chains="A")
    again = Protein.fetch("1UBQ", cache=tmp_path)
    assert len(first.topology.residues) == len(again.topology.residues) and len(calls) == 1
    assert calls[0].endswith("/1UBQ.cif")
    with pytest.raises(ValueError, match="PDB ID"):
        download.fetch_structure("not-an-id", cache=tmp_path)


def test_look_along_points_the_axis_at_the_viewer(protein):
    camera = Camera()
    axis = np.array([0.3, -0.5, 0.8])
    camera.look_along(axis)
    view = camera.eye - camera.target
    assert np.dot(view / np.linalg.norm(view), axis / np.linalg.norm(axis)) == pytest.approx(1, abs=1e-6)
    protein.rotate(0.7, (0, 1, 0))
    camera.look_along((0, 0, 1), protein)
    view = camera.eye - camera.target
    assert np.dot(view / np.linalg.norm(view), protein.orientation @ [0, 0, 1]) == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError):
        camera.look_along((0, 0, 0))


@pytest.mark.gpu
def test_structure_morph_renders():
    source = Protein.from_file(DATA / "4ake.cif", chains="A").cartoon()
    target = Protein.from_file(DATA / "1ake.cif", chains="A").cartoon()
    scene = ProteinScene(width=160, height=90)
    scene.add(source)
    scene.camera.frame(source, aspect=160 / 90)
    scene.play(StructureMorph(source, target), run_time=1)
    first, middle, last = (scene.render_frame(t) for t in (0, 0.5, 1))
    assert np.abs(first.astype(int) - middle.astype(int)).sum() > 0
    assert np.abs(middle.astype(int) - last.astype(int)).sum() > 0


def test_cartoon_frames_change_continuously_through_a_morph():
    from proteinmotion.geometry import secondary_weights, state_data

    source = Protein.from_file(DATA / "4ake.cif", chains="A").cartoon()
    target = Protein.from_file(DATA / "1ake.cif", chains="A").cartoon()
    scene = ProteinScene(width=64, height=36).add(source)
    scene.play(StructureMorph(source, target), run_time=2)
    trace = np.array([source.topology.residues[i].trace_atom for i in source.topology.chains[0]])
    before = None
    for frame in range(121):
        scene.seek(frame / 60)
        data = state_data(source.topology, source.positions, secondary=secondary_weights(source))
        guides, firm = data[trace, 4:7], data[trace, 7]
        if before is not None:
            # Every width axis turns smoothly, whatever its sign.
            agree = np.sum(guides * before[0], 1)
            assert np.degrees(np.arccos(np.clip(np.abs(agree), 0, 1))).max() < 5
            # A segment reverses its twist (one end's axis flips sign) only while drawn round.
            reversed_ = np.sign(agree[1:]) != np.sign(agree[:-1])
            assert np.all(np.maximum(firm[:-1], before[1][:-1])[reversed_] < 0.1)
        before = guides, firm


def test_rigid_core_anchors_morphs_and_domain_motions(structure_path, tmp_path):
    source = Protein.from_file(structure_path).cartoon()
    target = Protein.from_file(hinged_ubiquitin(structure_path, tmp_path)).cartoon()
    match = match_structures(source, target)
    core = [source.topology.atoms[a].resid for a in match.core]
    # The unmoved part (residues 1-39) superposes; the turned domain is left out.
    assert len(core) >= 35 and np.mean(np.array(core) <= 40) > 0.95
    assert match.report["core_rmsd"] < 0.5  # residues just past the hinge move under 2 Å
    turned = domain_motion(source, target, moving=source.select(residues=(45, 70)), match=match)
    assert turned.angle == pytest.approx(40, abs=0.3)
    # With no align region, the morph holds the core still and swings the rest.
    still = source.select(residues=(5, 30), atoms="CA").atom_indices
    before = source.positions[still].astype(float)
    scene = ProteinScene(width=64, height=36).add(source)
    scene.play(StructureMorph(source, target), run_time=2)
    scene.seek(1)
    np.testing.assert_allclose(source.positions[still], before, atol=0.05)


def test_sequence_alignment_drops_isolated_fragments():
    from proteinmotion.conformations import _align_residues

    head, tail = "MKTAYIAKQRQISFVKSHFSRQ", "LEERLGLIEVQAPILSRVGDGT"
    source = Protein.build(head + "HW" + tail)
    # The target inserts loops on both sides of the two-residue fragment HW.
    target = Protein.build(head + "GGGGGGSG" + "HW" + "PPGSGPPG" + tail)
    pairs = _align_residues(source, target, list(range(46)), list(range(62)))
    names = "".join(source.topology.residues[i].name[0] for i, _ in pairs)
    paired = {i for i, _ in pairs}
    assert 22 not in paired and 23 not in paired  # H and W fade instead of jumping across the insertions
    assert set(range(22)) <= paired and set(range(24, 46)) <= paired
    assert len(names) == 44
