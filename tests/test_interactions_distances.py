from dataclasses import replace

import numpy as np
import pytest

from proteinmotion import Distance, Electrostatics, HydrogenBonds, Protein, Region, Scene, Write
from proteinmotion.camera import Camera
from proteinmotion.interactions import COULOMB_KCAL_ANGSTROM, charges_from_pqr
from proteinmotion.structure import Atom, Residue, Topology


def tiny(xyz, names=None, elements=None, bonds=()):
    n = len(xyz)
    names = names or ["CA"] * n
    elements = elements or ["C"] * n
    atoms = tuple(Atom("A", i + 1, "", "ALA", names[i], elements[i], i) for i in range(n))
    residues = tuple(Residue("A", i + 1, "", "ALA", i if names[i] == "CA" else -1, -1) for i in range(n))
    return Protein(
        Topology(atoms, residues, np.asarray(bonds, np.uint32).reshape(-1, 2), ()), xyz
    ).ball_and_stick()


def test_screened_coulomb_energy_potential_and_parameter_cache():
    p = tiny([[0, 0, 0], [5, 0, 0], [0, 5, 0]])
    e = Electrostatics(p, [1, -1, 0], dielectric=80, screening_length=None, min_energy=0)
    expected = -COULOMB_KCAL_ANGSTROM / (80 * 5)
    assert e.pair_energy(0, 1) == pytest.approx(expected)
    assert len(e.pairs) == 1 and e.pairs[0].kind == "attractive"
    np.testing.assert_allclose(e.potential([[2.5, 0, 0]], softening=0), [0], atol=1e-12)
    e.screening_length = 8
    assert e.pairs[0].energy == pytest.approx(expected * np.exp(-5 / 8))
    e.dielectric = 40
    assert e.pairs[0].energy == pytest.approx(expected * 2 * np.exp(-5 / 8))
    p.shift([100, 0, 0]).scale(2)
    assert e.pairs[0].distance == 5  # Physical analysis ignores the display transform.
    p.set_positions([[0, 0, 0], [6, 0, 0], [0, 5, 0]])
    assert e.pairs[0].distance == 6
    with pytest.raises(ValueError, match="singular"):
        e.potential([[0, 0, 0]], softening=0)
    with pytest.raises(ValueError):
        Electrostatics(p, [1, 2])
    with pytest.raises(ValueError):
        Electrostatics(p, [1, 2, 3], dielectric=0)


def test_pqr_identity_charge_import_and_explicit_extra_policy(tmp_path):
    p = tiny([[0, 0, 0], [5, 0, 0]])
    path = tmp_path / "charges.pqr"
    # Deliberately reversed rows: correspondence uses identities, not row order.
    path.write_text("ATOM 2 CA ALA A 2 5.0 0.0 0.0 -0.75 1.7\nATOM 1 CA ALA A 1 0.0 0.0 0.0 0.25 1.7\n")
    np.testing.assert_array_equal(charges_from_pqr(p, path), [0.25, -0.75])
    assert Electrostatics.from_pqr(p, path).charge_source == "PQR: charges.pqr"
    path.write_text(path.read_text() + "ATOM 3 H ALA A 1 0.0 1.0 0.0 0.5 1.1\n")
    with pytest.raises(ValueError, match="extra atoms"):
        charges_from_pqr(p, path)
    np.testing.assert_array_equal(charges_from_pqr(p, path, allow_extra=True), [0.25, -0.75])
    path.write_text("ATOM 1 CA ALA B 1 0 0 0 0.2 1.7\n")
    with pytest.raises(ValueError, match="lacks"):
        charges_from_pqr(p, path)


def test_hbond_explicit_geometry_rejects_bent_and_distance_cutoff():
    p = tiny([[0, 0, 0], [1, 0, 0], [2.8, 0, 0]], names=["N", "H", "O"], elements=["N", "H", "O"])
    # H belongs to the donor residue, unlike the default helper's separate residues.
    atoms = list(p.topology.atoms)
    atoms[1] = replace(atoms[1], residue_index=0, resid=1)
    p.topology = replace(p.topology, atoms=tuple(atoms))
    hb = HydrogenBonds(p, donors=[0], acceptors=[2], hydrogens="explicit")
    assert len(hb.pairs) == 1
    record = hb.pairs[0]
    assert record.angle == pytest.approx(180) and record.distance == pytest.approx(2.8)
    assert record.hydrogen == 1 and not record.inferred_hydrogen
    p.set_positions([[0, 0, 0], [0, 1, 0], [2.8, 0, 0]])
    assert not hb.pairs
    p.set_positions([[0, 0, 0], [1, 0, 0], [4, 0, 0]])
    assert not hb.pairs
    hb.max_distance = 4.2
    assert len(hb.pairs) == 1


def test_real_backbone_hbonds_and_no_silent_sidechain_hydrogens(protein):
    hb = protein.hydrogen_bonds()
    assert len(hb.pairs) > 20
    assert all(r.inferred_hydrogen and protein.topology.atoms[r.a].name == "N" for r in hb.pairs)
    assert not protein.hydrogen_bonds(hydrogens="explicit").pairs
    assert all(r.distance <= 3.5 and r.angle >= 150 for r in hb.pairs)
    formal = protein.electrostatics()
    assert formal.charges.sum() == 0
    assert any(r.kind == "attractive" for r in formal.pairs)
    assert any(r.kind == "repulsive" for r in formal.pairs)
    # Real 1UBQ helix: the 150° cutoff excludes two slightly bent inferred H.
    # Lowering the declared cutoff recovers those pairs without forcing i+4.
    local = HydrogenBonds(
        protein,
        donors=protein.select(residues=(23, 34), atoms="N"),
        acceptors=protein.select(residues=(23, 34), atoms="O"),
    )

    def keys():
        return {(protein.topology.atoms[r.b].resid, protein.topology.atoms[r.a].resid) for r in local.pairs}

    assert keys() == {(23, 27), (25, 29), (26, 30), (27, 31), (28, 32), (30, 34)}
    local.min_angle = 140
    assert keys() == {(i, i + 4) for i in range(23, 31)}


def test_distance_ca_atom_centroid_units_and_display_scale(protein):
    a, b = protein.select(residues=8), protein.select(residues=70)
    d = Distance(a, b, mode="2d", unit="nm", precision=2)
    assert len(d.start.atom_indices) == 1
    expected = float(np.linalg.norm(d.start.positions - d.end.positions))
    assert d.distance == pytest.approx(expected)
    assert d.title.text == f"{expected / 10:.2f} nm"
    protein.scale(2).shift([20, -4, 8])
    assert d.distance == pytest.approx(expected)
    assert Distance(a, b, space="world").distance == pytest.approx(expected * 2)
    atom = protein.select(residues=8, atoms="N")
    assert Distance(atom, b).start.atom_indices[0] == atom.atom_indices[0]
    centroid = Distance(a, b, anchor="centroid")
    assert centroid.distance == pytest.approx(np.linalg.norm(a.positions.mean(0) - b.positions.mean(0)))
    with pytest.raises(ValueError, match="space=world"):
        Distance(a, protein.copy().select(residues=8))


def test_distance_updates_caption_and_tracks_live_coordinates(protein):
    d = protein.select(residues=8).distance_to(protein.select(residues=70), mode="2d")
    camera = Camera().frame(protein)
    first = d.layout(camera, 1920, 1080)
    points = protein.positions.copy()
    points[d.end.atom_indices] += [3, 0, 0]
    protein.set_positions(points)
    second = d.layout(camera, 1920, 1080)
    assert second.text[0].text.text == f"{d.distance:.1f} Å"
    assert not np.array_equal(first.text[0].origin, second.text[0].origin)
    assert all(leader.points.ndim == 3 for leader in second.leaders)


@pytest.mark.gpu
@pytest.mark.parametrize("msaa", [1, 4])
def test_3d_ruler_is_occluded_and_2d_overlay_is_visible(msaa):
    from proteinmotion.renderer import Renderer

    p = tiny([[-4, 0, -2], [4, 0, -2]])
    blocker = tiny([[0, 0, 2]]).ball_and_stick(atom_scale=2.5)
    a, b = Region(p, [0]), Region(p, [1])
    d3 = Distance(a, b, mode="3d", style="solid", show_distance=False, color="#ff0000", radius=0.2)
    d2 = Distance(a, b, mode="2d", style="solid", show_distance=False, color="#ff0000", line_width=10)
    camera = Camera().frame(p, blocker, aspect=1)
    camera.theta = camera.phi = camera.depth_cue = 0
    bg = np.array([0.03, 0.05, 0.08])
    with Renderer(256, 256, msaa=msaa) as r:
        reference = r.render([blocker], camera, bg)
        depth = r.render([blocker, d3], camera, bg)
        overlay = r.render([blocker, d2], camera, bg)
        np.testing.assert_array_equal(depth[122:134, 122:134], reference[122:134, 122:134])
        assert overlay[128, 128, 0] > reference[128, 128, 0] + 20
        assert overlay[128, 128, 1] < reference[128, 128, 1] - 20


@pytest.mark.gpu
def test_dynamic_interaction_labels_keep_gpu_pool_bounded_and_seek(protein):
    from proteinmotion.renderer import Renderer

    scene = Scene(width=480, height=320)
    protein.ball_and_stick()
    scene.add(protein)
    scene.camera.frame(protein, aspect=1.5)
    hb = protein.hydrogen_bonds()
    visual = hb.highlight(mode="3d", show_distances=True, max_pairs=4, font_size=38, label_color="#ffffff")
    scene.play(Write(visual), protein.animate.rotate(0.6), run_time=2)
    with Renderer(480, 320) as r:
        expected = scene.render_frame(1.5, renderer=r)
        count = len(r._molecules)
        scene.render_frame(2, renderer=r)
        scene.render_frame(0, renderer=r)
        np.testing.assert_array_equal(scene.render_frame(1.5, renderer=r), expected)
        assert len(r._molecules) == count == 5
        assert len(r._overlay.text) == 4
        assert visual.total_pairs > len(visual.visible_pairs) == 4
        assert all(np.array_equal(slot.title.color, [1, 1, 1]) for slot in visual._slots)
