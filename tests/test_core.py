from dataclasses import replace

import gemmi
import numpy as np
import pytest

from proteinmotion import (
    Deform,
    FadeIn,
    Morph,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Representation,
    Rotate,
    Trajectory,
    linear,
    there_and_back,
)
from proteinmotion.geometry import state_data
from proteinmotion.math3d import align_coordinates, rotation
from proteinmotion.structure import Topology
from proteinmotion.trajectory import FrameCache


def test_load_annotations_and_chain_breaks(protein):
    p = protein
    assert len(p.topology.atoms) == 602
    assert len(p.topology.residues) == 76
    assert {r.secondary for r in p.topology.residues} == {"H", "E", "C"}
    assert len(p.topology.chains) == 1
    assert len(p.topology.bonds) > 600
    assert not any(a.resname == "HOH" for a in p.topology.atoms)
    from proteinmotion.structure import make_chains

    xyz = p.positions
    split = p.topology.residues[38].ca
    xyz[split:] += 20
    assert len(make_chains(p.topology.residues, xyz)) == 2


def test_multimodel_pdb_and_altlocs(tmp_path, structure_path):
    st = gemmi.read_structure(str(structure_path))
    st.remove_waters()
    second = st[0].clone()
    second.num = 2
    for chain in second:
        for res in chain:
            for atom in res:
                atom.pos.x += 2
    st.add_model(second)
    path = tmp_path / "ensemble.pdb"
    st.write_pdb(str(path))
    p = Protein.from_file(path)
    assert len(p.trajectory) == 2
    np.testing.assert_allclose(
        p.trajectory.frame(1) - p.trajectory.frame(0), np.tile([2, 0, 0], (602, 1)), atol=1e-5
    )
    del st[1][0][0][0]
    st.write_pdb(str(path))
    with pytest.raises(ValueError, match="same selected atom identities"):
        Protein.from_file(path)


def test_highest_occupancy_altloc(tmp_path, structure_path):
    st = gemmi.read_structure(str(structure_path))
    atom = st[0][0][0][1]
    original_x = atom.pos.x
    alternate = atom.clone()
    atom.altloc, atom.occ = "A", 0.3
    alternate.altloc, alternate.occ = "B", 0.7
    alternate.pos.x += 1
    st[0][0][0].add_atom(alternate)
    path = tmp_path / "alternate.cif"
    st.make_mmcif_document().write_file(str(path))
    p = Protein.from_file(path)
    assert len(p.topology.atoms) == 602
    ca = p.topology.residues[0].ca
    assert p.positions[ca, 0] == pytest.approx(original_x + 1, abs=1e-4)


def test_kabsch_removes_rigid_motion_and_preserves_chirality(protein):
    xyz = protein.positions
    rotated = xyz @ rotation(1.3, (1, 2, 3)).T + [7, -10, 3]
    np.testing.assert_allclose(align_coordinates(rotated, xyz), xyz, atol=1e-5)
    mirrored = xyz * np.array([-1, 1, 1])
    assert np.sqrt(np.mean((align_coordinates(mirrored, xyz) - xyz) ** 2)) > 1


def test_morph_identity_matching_and_rejection(protein):
    p = protein
    target = p.copy().set_positions(p.positions + [1, 2, 3])
    order = np.arange(len(p.topology.atoms))[::-1]
    t = target.topology
    target.topology = Topology(tuple(t.atoms[i] for i in order), t.residues, t.bonds, t.chains)
    target.set_positions(target.positions[order])
    scene = ProteinScene()
    scene.add(p)
    scene.play(Morph(p, target, align=False), run_time=2, rate_func=linear)
    scene.seek(1)
    np.testing.assert_allclose(p.positions, target.positions[order] - [0.5, 1, 1.5], atol=1e-5)
    bad = target.copy()
    bad.topology = replace(t, atoms=(replace(t.atoms[0], chain="Z"),) + t.atoms[1:])
    with pytest.raises(ValueError, match="identities"):
        Morph(p, bad).bind()


def test_seek_replay_combined_channels_and_no_drift(protein):
    p = protein
    original = p.positions.copy()
    scene = ProteinScene()
    scene.add(p)
    scene.play(Rotate(p, np.pi), Morph(p, original + [4, 0, 0], align=False), run_time=2, rate_func=linear)
    scene.play(p.animate.shift((0, 3, 0)).scale(1.3), run_time=1)
    scene.seek(2.5)
    expected = (p.positions.copy(), p.model_matrix.copy())
    for time in [0, 1.3, 3, 0.2, 2.5]:
        scene.seek(time)
    np.testing.assert_allclose(p.positions, expected[0])
    np.testing.assert_allclose(p.model_matrix, expected[1])
    scene.seek(0)
    np.testing.assert_allclose(p.positions, original)
    scene.seek(1)
    np.testing.assert_allclose(p.positions, original + [2, 0, 0], atol=1e-5)


def test_easing_end_state_is_used_by_next_clip(protein):
    initial = protein.model_matrix.copy()
    s = ProteinScene()
    s.add(protein)
    s.play(Rotate(protein, 2), rate_func=there_and_back)
    s.play(protein.animate.shift((1, 0, 0)))
    s.seek(2)
    initial[0, 3] += 1
    np.testing.assert_allclose(protein.model_matrix, initial)


def test_conflicting_channels_rejected(protein):
    scene = ProteinScene()
    with pytest.raises(ValueError, match="geometry"):
        scene.play(Morph(protein, protein.positions), PlayTrajectory(protein))
    with pytest.raises(ValueError, match="transform"):
        scene.play(Rotate(protein, 1), protein.animate.shift([1, 0, 0]))


def test_deformation_endpoint_and_representation(protein):
    s = ProteinScene()
    s.add(protein)
    start = protein.positions.copy()
    s.play(
        Deform(protein, lambda xyz: xyz + [0, 4, 0]),
        Representation(protein, "ribbon"),
        run_time=2,
        rate_func=linear,
    )
    s.seek(1)
    np.testing.assert_allclose(protein.positions, start + [0, 2, 0], atol=1e-5)
    np.testing.assert_allclose(protein.representation, [0.5, 0.5, 0])


def test_trajectory_npy_units_reverse_and_cache(protein, tmp_path):
    frames = np.array([protein.positions + [i, 0, 0] for i in range(7)]) / 10
    path = tmp_path / "traj.npy"
    np.save(path, frames)
    t = Trajectory.from_npy(path, units="nm")
    s = ProteinScene()
    s.add(protein)
    s.play(PlayTrajectory(protein, t, start=6, end=0), run_time=6)
    s.seek(2.5)
    np.testing.assert_allclose(protein.positions, t.frame(3) * 0.5 + t.frame(4) * 0.5, atol=1e-5)
    cache = FrameCache(t)
    for i in range(7):
        cache.get(i)
    assert len(cache.cache) == 3
    assert isinstance(t._frames, np.memmap)
    bad = Trajectory(np.zeros((2, 3, 3)))
    with pytest.raises(ValueError, match="atom count"):
        PlayTrajectory(protein, bad)


def test_frames_finite_and_unit_guides(protein):
    data = state_data(protein.topology, protein.positions)
    ca = [r.ca for r in protein.topology.residues if r.ca >= 0]
    assert np.isfinite(data).all()
    np.testing.assert_allclose(np.linalg.norm(data[ca, 4:7], axis=1), 1, atol=1e-5)


def test_fade_auto_add_and_visibility(protein):
    s = ProteinScene()
    s.wait(1)
    s.play(FadeIn(protein))
    assert s.seek(0.5) == []
    assert s.seek(1) == []
    assert s.seek(1.5) == [protein]
    assert protein.opacity == pytest.approx(0.5)


@pytest.mark.parametrize("value", [float("nan"), -1, 0])
def test_bad_timing(protein, value):
    with pytest.raises(ValueError):
        ProteinScene().play(Rotate(protein, 1), run_time=value)


def test_preview_controls():
    from proteinmotion.preview import PreviewControls

    c = PreviewControls(5)
    c.event({"event_type": "pointer_down", "button": 1, "x": 100, "y": 100})
    c.event({"event_type": "pointer_move", "x": 150, "y": 125})
    assert c.theta == pytest.approx(-0.4)
    assert c.phi == pytest.approx(0.2)
    c.event({"event_type": "pointer_up"})
    c.event({"event_type": "wheel", "dy": -100})
    assert c.zoom > 1
    c.event({"event_type": "key_down", "key": "Space"})
    c.advance(1)
    assert c.time == 1
    c.event({"event_type": "key_down", "key": "ArrowLeft"})
    assert c.time == 0.75
    c.event({"event_type": "key_down", "key": "r"})
    assert (c.theta, c.phi, c.zoom) == (0, 0, 1)
