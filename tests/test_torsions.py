"""Torsion measurement and animation, ideal peptides, DSSP, and Ramachandran plots."""

import warnings
from pathlib import Path

import gemmi
import numpy as np
import pytest

from proteinmotion import (
    Protein,
    ProteinScene,
    RamachandranPlot,
    RotateTorsions,
    SecondaryStructure,
    SetTorsions,
    TorsionMarker,
    Write,
)
from proteinmotion.builder import CB_TORSION
from proteinmotion.camera import Camera
from proteinmotion.geometry import segments, studio_cartoon_segments
from proteinmotion.ramachandran import classify
from proteinmotion.secondary import assign
from proteinmotion.torsions import dihedrals, torsion_atoms, wrap

# DSSP of the deposited coordinates, identical to mdtraj.compute_dssp reduced to H/E/C.
UBIQUITIN_DSSP = "CEEEEEECCCCEEEEECCCCCCHHHHHHHHHHHHCCCHHHEEEEECCEECCCCCCCHHHCCCCCCEEEEEECCCCC"


def bond_lengths(protein):
    bonds = protein.topology.bonds
    xyz = protein.positions
    return np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=1)


def test_dssp_reproduces_reference_assignment(structure_path):
    p = Protein.from_file(structure_path, secondary="file")
    assert assign(p.topology, p.positions) == UBIQUITIN_DSSP
    dssp = Protein.from_file(structure_path, secondary="dssp")
    assert dssp.secondary_structure == UBIQUITIN_DSSP
    assert "".join(r.secondary for r in dssp.topology.residues) == UBIQUITIN_DSSP


def test_files_without_records_get_dssp(structure_path, tmp_path):
    st = gemmi.read_structure(str(structure_path))
    st.helices.clear()
    st.sheets.clear()
    st.setup_entities()
    plain = tmp_path / "plain.pdb"
    st.write_pdb(str(plain))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = Protein.from_file(plain)
    assert not [w for w in caught if "helix/sheet" in str(w.message)]
    assert p.secondary_structure == UBIQUITIN_DSSP
    with pytest.warns(UserWarning, match="No helix/sheet"):
        coil = Protein.from_file(plain, secondary="file")
    assert set(coil.secondary_structure) == {"C"}


def test_measured_torsions_match_known_helix(protein):
    angles = protein.torsions(residues=(24, 32))
    assert np.all((-75 < angles.phi) & (angles.phi < -50))
    assert np.all((-55 < angles.psi) & (angles.psi < -30))
    assert angles[28]["phi"] == pytest.approx(-66.1, abs=0.1)
    assert np.isnan(protein.torsions().phi[0]) and np.isnan(protein.torsions().psi[-1])
    assert abs(protein.torsions(residues=8)["A", 8]["omega"]) > 170
    with pytest.raises(KeyError):
        angles[1]


def test_built_peptide_has_exact_torsions_and_l_chirality():
    phi = np.linspace(-150, -50, 12)
    psi = np.linspace(160, -40, 12)
    p = Protein.build("ACDEFGHIKLMN", phi=phi, psi=psi, omega=179.0)
    angles = p.torsions()
    np.testing.assert_allclose(angles.phi[1:], phi[1:], atol=1e-3)
    np.testing.assert_allclose(angles.psi[:-1], psi[:-1], atol=1e-3)
    np.testing.assert_allclose(angles.omega[:-1], 179.0, atol=1e-3)
    names = p.topology.residue_atoms
    quads = [[n["C"], n["N"], n["CA"], n["CB"]] for n in names if "CB" in n]
    np.testing.assert_allclose(dihedrals(p.positions, quads), CB_TORSION, atol=1e-3)
    lengths = bond_lengths(p)
    assert lengths.min() > 1.2 and lengths.max() < 1.85
    # Rings close: phenylalanine and histidine have as many ring bonds as ring atoms.
    for name, ring in (("PHE", 6), ("HIS", 5)):
        ri = next(i for i, r in enumerate(p.topology.residues) if r.name == name)
        atoms = {i for i in names[ri].values()} - {names[ri][k] for k in ("N", "CA", "C", "O", "CB")}
        assert sum(a in atoms and b in atoms for a, b in p.topology.bonds) == ring
    helix = Protein.build("A" * 16, "helix", hydrogens=True)
    assert helix.secondary_structure == "C" + "H" * 14 + "C"
    assert sum(a.name == "H" for a in helix.topology.atoms) == 15
    assert abs(np.ptp(helix.positions[:, 1])) < np.ptp(helix.positions[:, 0])  # laid along x
    proline = Protein.build("APA")
    assert proline.torsions()[2]["phi"] == pytest.approx(-65.0, abs=1e-3)
    with pytest.raises(ValueError, match="Unknown one-letter"):
        Protein.build("AXZ")
    with pytest.raises(ValueError, match="conformation"):
        Protein.build("AAA", "spiral")


def test_set_torsions_is_exact_bond_preserving_and_seekable():
    p = Protein.build("A" * 12)
    lengths = bond_lengths(p)
    start = p.positions.copy()
    scene = ProteinScene(width=64, height=36).add(p)
    scene.play(SetTorsions(p, conformation="helix"), run_time=2)
    scene.play(SetTorsions(p.select(residues=(4, 9)), conformation="strand", stagger=0.4), run_time=2)
    scene.seek(2)
    angles = p.torsions()
    np.testing.assert_allclose(angles.phi[1:], -57, atol=1e-3)
    np.testing.assert_allclose(angles.psi[:-1], -47, atol=1e-3)
    np.testing.assert_allclose(bond_lengths(p), lengths, atol=1e-4)
    assert p.secondary_structure == "C" + "H" * 10 + "C"
    scene.seek(4)
    np.testing.assert_allclose(p.torsions(residues=(4, 9)).psi, 135, atol=1e-3)
    scene.seek(3.1)
    middle = p.positions.copy()
    scene.seek(0)
    np.testing.assert_allclose(p.positions, start, atol=1e-5)
    scene.seek(3.1)
    np.testing.assert_allclose(p.positions, middle, atol=1e-6)


def test_paths_stay_in_allowed_regions_and_shortest_is_available():
    p = Protein.build("A" * 10)
    allowed = SetTorsions(p, conformation="helix")
    allowed.bind()
    shortest = SetTorsions(p.copy(), conformation="helix", path="shortest")
    shortest.bind()
    psi = [np.degrees(d) for d, kind in zip(allowed.plan.deltas, allowed.plan.kinds) if kind == "psi"]
    short_psi = [np.degrees(d) for d, kind in zip(shortest.plan.deltas, shortest.plan.kinds) if kind == "psi"]
    np.testing.assert_allclose(psi, -227, atol=1e-6)  # down through β and the bridge region
    np.testing.assert_allclose(short_psi, 133, atol=1e-6)
    t = np.linspace(0, 1, 30)[:, None]
    chosen, direct = wrap(180 + t * [123, -227]), wrap(180 + t * [123, 133])
    assert np.mean(classify(*chosen.T) != "outlier") > np.mean(classify(*direct.T) != "outlier")


def test_anchors_keep_the_chosen_part_still():
    for anchor, residue in (("n", 1), ("c", 10)):
        p = Protein.build("A" * 10)
        atom = p.topology.residues[residue - 1].ca
        before = p.positions[atom].copy()
        p.set_torsions(conformation="helix", anchor=anchor)
        np.testing.assert_allclose(p.positions[atom], before, atol=1e-4)
    p = Protein.build("A" * 10)
    held = p.select(residues=(7, 10))
    before = held.positions.copy()
    p.set_torsions(phi={3: -60, 4: -70}, anchor=held)
    np.testing.assert_allclose(held.positions, before, atol=1e-4)
    with pytest.raises(ValueError, match="anchor"):
        SetTorsions(p, phi=-60, anchor="middle")


def test_rings_lock_and_relative_turns():
    p = Protein.build("AAPAA")
    with pytest.raises(ValueError, match="rings"):
        p.select(residues=3).set_torsions(phi=-120)
    animation = SetTorsions(p, phi=-120)
    animation.bind()
    assert animation.locked == ((2, "phi"),)
    q = Protein.build("AKA")
    start = q.positions.copy()
    scene = ProteinScene(width=64, height=36).add(q)
    scene.play(RotateTorsions(q.select(residues=2), chi1=360), run_time=1)
    scene.seek(0.5)
    assert np.abs(q.positions - start).max() > 1
    scene.seek(1)
    np.testing.assert_allclose(q.positions, start, atol=1e-4)
    ions = Protein.from_file(Path(__file__).resolve().parents[1] / "examples/data/1cll.cif").select(ions=True)
    with pytest.raises(ValueError, match="no amino acids"):
        SetTorsions(ions, phi=0)


def test_cartoon_blends_secondary_structure_during_torsion_changes():
    p = Protein.build("A" * 12).cartoon()
    coil = segments(p)[:, 4].copy()
    scene = ProteinScene(width=64, height=36).add(p)
    scene.play(SetTorsions(p, conformation="helix"), run_time=2)
    scene.seek(1)
    halfway = p._secondary[5]
    assert 0 < halfway[1] < 1 and halfway.sum() == pytest.approx(1)
    scene.seek(2)
    widths = segments(p)[:, 4]
    assert widths[5] > 4 * coil[5]
    rows = studio_cartoon_segments(p)
    assert np.all((rows[:, 18] >= 0) & (rows[:, 18] <= 1))
    p.set_secondary_structure("C" * 4 + "E" * 4 + "C" * 4)
    rows = studio_cartoon_segments(p)
    assert rows[:, 18].sum() == pytest.approx(1)
    animation = SecondaryStructure(p, "dssp")
    scene.play(animation, run_time=1)
    scene.seek(3)
    assert p.secondary_structure == "C" + "H" * 10 + "C"
    with pytest.raises(ValueError, match="one H/E/C"):
        SecondaryStructure(p, "HHH")


def test_torsion_marker_reports_live_value_and_draws():
    p = Protein.build("A" * 6)
    marker = TorsionMarker(p.select(residues=3), "psi")
    assert marker.value == pytest.approx(180, abs=1e-3) or marker.value == pytest.approx(-180, abs=1e-3)
    scene = ProteinScene(width=640, height=360).add(p, marker)
    scene.camera.frame(p, aspect=640 / 360)
    scene.play(Write(marker), run_time=1)
    scene.play(SetTorsions(p.select(residues=3), psi=-47), run_time=1)
    scene.seek(2)
    assert marker.value == pytest.approx(-47, abs=1e-3)
    layout = marker.layout(scene.camera, 640, 360)
    assert layout.text and len(layout.leaders) >= 3
    assert "−47°" in marker.title.text and marker.title.text.startswith("ψ")
    with pytest.raises(ValueError, match="no phi"):
        TorsionMarker(p.select(residues=1), "phi")
    with pytest.raises(ValueError, match="exactly one"):
        TorsionMarker(p.select(residues=(2, 3)), "phi")


def test_ramachandran_regions_and_live_plot(protein):
    assert list(classify([-63, -120, 63, 60], [-42, 130, 42, -150])) == [
        "favored",
        "favored",
        "favored",
        "outlier",
    ]
    assert classify(80, 10, "glycine") == "favored" and classify(-63, 140, "proline") == "favored"
    plot = RamachandranPlot(protein, highlight=protein.select(residues=(23, 34)))
    assert len(plot.ids) == 74
    favored = np.mean(classify(*plot.angles.T) == "favored")
    assert favored > 0.9
    camera = Camera()
    layout = plot.layout(camera, 1280, 720)
    assert layout.triangles.size and layout.leaders
    peptide = Protein.build("A" * 8)
    live = RamachandranPlot(peptide)
    scene = ProteinScene(width=64, height=36).add(peptide, live)
    scene.play(SetTorsions(peptide, conformation="helix"), run_time=1)
    scene.seek(0)
    still = len(live.layout(camera, 1280, 720).leaders)
    scene.seek(0.5)
    moving = len(live.layout(camera, 1280, 720).leaders)
    assert moving > still  # each moving residue draws its path
    scene.seek(1)
    np.testing.assert_allclose(live.angles, [[-57, -47]] * len(live.ids), atol=1e-3)
    with pytest.raises(ValueError, match="background"):
        RamachandranPlot(protein, background="beta")


def test_torsion_table_rejects_nucleotides_and_handles_chain_breaks(structure_path):
    p = Protein.from_file(structure_path)
    table = torsion_atoms(p.topology)
    assert table.shape == (len(p.topology.residues), 8, 4)
    assert (table[0, 0] == -1).all() and (table[-1, 1] == -1).all()
    with pytest.raises(TypeError):
        SetTorsions("protein", phi=0)


@pytest.mark.gpu
@pytest.mark.parametrize("renderer", ["native", "studio"])
def test_torsion_animation_renders(renderer):
    p = Protein.build("A" * 12).cartoon()
    scene = ProteinScene(width=160, height=90)
    scene.add(p)
    scene.camera.frame(p, aspect=160 / 90)
    scene.play(SetTorsions(p, conformation="helix"), run_time=1)
    first = scene.render_frame(0, renderer=renderer)
    middle = scene.render_frame(0.5, renderer=renderer)
    last = scene.render_frame(1, renderer=renderer)
    assert np.abs(first.astype(int) - last.astype(int)).sum() > 0
    assert np.abs(middle.astype(int) - last.astype(int)).sum() > 0
