"""Independent sequence-pattern checks for the ideal helix, plus explicit H rendering."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from proteinmotion import HydrogenBonds, Scene, Write

EXAMPLE = runpy.run_path(str(Path(__file__).parents[1] / "examples/alpha_helix_hbonds.py"))


def helix(**kwargs):
    return EXAMPLE["ideal_helix"](**kwargs)


def torsion(a, b, c, d):
    axis = c - b
    axis /= np.linalg.norm(axis)
    v, w = a - b, d - c
    v -= np.dot(v, axis) * axis
    w -= np.dot(w, axis) * axis
    return np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w)))


def test_ideal_alpha_helix_recovers_every_i_plus_four_pair():
    p = helix()
    xyz = p.positions
    ids = {(a.resid, a.name): i for i, a in enumerate(p.topology.atoms)}
    for residue in range(2, 16):
        n, ca, c, following_n = [
            xyz[ids[k]] for k in [(residue, "N"), (residue, "CA"), (residue, "C"), (residue + 1, "N")]
        ]
        previous_c = xyz[ids[residue - 1, "C"]]
        assert torsion(previous_c, n, ca, c) == pytest.approx(-57, abs=1e-4)
        assert torsion(n, ca, c, following_n) == pytest.approx(-47, abs=1e-4)
        assert np.linalg.norm(c - following_n) == pytest.approx(1.329, abs=1e-5)
    hb = HydrogenBonds(p, hydrogens="explicit")
    pairs = {(p.topology.atoms[r.b].resid, p.topology.atoms[r.a].resid) for r in hb.pairs}
    assert pairs == {(i, i + 4) for i in range(1, 13)}
    assert all(r.hydrogen is not None and not r.inferred_hydrogen for r in hb.pairs)
    for r in hb.pairs:
        np.testing.assert_allclose(hb.hydrogen_position(r), xyz[r.hydrogen])
        assert 2.8 < r.distance < 3.2
        assert 1.8 < np.linalg.norm(xyz[r.hydrogen] - xyz[r.b]) < 2.2
        assert 150 < r.angle < 180


def test_bent_amide_h_and_extended_backbone_do_not_get_forced_helix_bonds():
    p = helix()
    hb = HydrogenBonds(p, hydrogens="explicit")
    first = hb.pairs[0]
    xyz = p.positions.copy()
    xyz[first.hydrogen] = 2 * xyz[first.a] - xyz[first.hydrogen]
    p.set_positions(xyz)
    assert len(hb.pairs) == 11
    assert all((r.a, r.b) != (first.a, first.b) for r in hb.pairs)
    assert not HydrogenBonds(helix(phi=-135, psi=135), hydrogens="explicit").pairs


def test_hydrogen_endpoints_match_explicit_and_virtual_coordinates():
    p = helix()
    for mode in ("explicit", "backbone"):
        hb = HydrogenBonds(p, hydrogens=mode)
        visual = hb.highlight(endpoints="hydrogen_acceptor", show_distances=True, precision=2)
        visual._refresh()
        assert len(visual.visible_pairs) == 12
        for pair, slot in zip(visual.visible_pairs, visual._slots):
            h = hb.hydrogen_position(pair)
            np.testing.assert_allclose(slot.start.positions[0], h)
            assert slot.distance == pytest.approx(np.linalg.norm(h - p.positions[pair.b]), abs=1e-6)
        p.rotate(0.4).shift([1, 3, -2])
        np.testing.assert_allclose(
            visual._slots[0].start.world_positions[0],
            p.model_matrix[:3, :3] @ hb.hydrogen_position(hb.pairs[0]) + p.position,
        )
    with pytest.raises(ValueError, match="require HydrogenBonds"):
        p.electrostatics().highlight(endpoints="hydrogen_acceptor")
    with pytest.raises(ValueError, match="endpoints"):
        hb.highlight(endpoints="invalid")


@pytest.mark.gpu
def test_helix_hydrogen_lines_render_and_seek_without_label_gaps():
    from proteinmotion.renderer import Renderer

    p = helix().ball_and_stick(atom_scale=0.25, bond_radius=0.09)
    hb = HydrogenBonds(p, hydrogens="explicit")
    visual = hb.highlight(
        endpoints="hydrogen_acceptor", show_distances=False, color="#f2ba67", radius=0.055, dash_count=5
    )
    scene = Scene(width=512, height=512)
    scene.add(p)
    scene.camera.frame(p, aspect=1)
    scene.play(Write(visual), run_time=1)
    scene.play(p.animate.rotate(0.7), run_time=2)
    with Renderer(512, 512) as renderer:
        plain = scene.render_frame(0, renderer=renderer)
        bonded = scene.render_frame(1, renderer=renderer)
        assert np.abs(bonded.astype(float) - plain.astype(float)).sum() > 10000
        scene.render_frame(3, renderer=renderer)
        np.testing.assert_array_equal(scene.render_frame(1, renderer=renderer), bonded)
        assert len(renderer._molecules) == 13
        assert all(not slot.show_distance for slot in visual._slots)
        for slot, pair in zip(visual._slots, visual.visible_pairs):
            assert slot.start.atom_indices.tolist() == [pair.hydrogen]
            assert slot.end.atom_indices.tolist() == [pair.b]
