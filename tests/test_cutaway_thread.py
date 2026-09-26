"""View-dependent cutaways, depth tunnels, animated fog, and threading wires."""

from pathlib import Path

import numpy as np
import pytest

from proteinmotion import Conceal, Protein, ProteinScene, Reveal, Thread, Unthread
from proteinmotion.camera import Camera
from proteinmotion.thread import _arc_length

DATA = Path(__file__).resolve().parents[1] / "examples/data"


@pytest.fixture
def hemoglobin():
    return Protein.from_file(DATA / "4hhb.cif").cartoon(color="chain").center()


def framed(protein, theta=0.4, phi=0.2):
    scene = ProteinScene(width=384, height=256)
    scene.camera.frame(protein, aspect=1.5)
    scene.camera.theta, scene.camera.phi = theta, phi
    return scene


def test_reveal_opens_closes_and_seeks(protein):
    scene = framed(protein)
    scene.add(protein)
    site = protein.select(residues=(23, 34))
    scene.wait(0.5)
    scene.play(Reveal(scene.camera, site, window=1.5, softness=0.3), run_time=1)
    scene.play(Reveal(scene.camera, site, window=3.0), run_time=1)
    scene.play(Conceal(scene.camera), run_time=1)
    camera = scene.camera
    scene.seek(0.2)
    assert camera._cutaway_target is None and camera.cutaway_geometry() is None
    scene.seek(1.0)
    assert camera._cutaway_target is site and camera.cutaway_opening == pytest.approx(0.5)
    center, keep, window, _, _ = camera.cutaway_geometry()
    points = site.world_positions
    np.testing.assert_allclose(center, (points.min(0) + points.max(0)) / 2)
    assert keep == pytest.approx(np.linalg.norm(points - center, axis=1).max() + camera.cutaway_padding)
    assert window == pytest.approx(keep * 1.5)
    # A second Reveal on the same region reshapes the open window smoothly.
    scene.seek(2.0)
    assert camera.cutaway_window == pytest.approx(2.25) and camera.cutaway_opening == 1
    scene.seek(3.5)
    assert camera.cutaway_opening == 0 and camera.cutaway_geometry() is None
    scene.seek(1.0)
    assert camera.cutaway_opening == pytest.approx(0.5)
    with pytest.raises(ValueError, match="shape"):
        Reveal(scene.camera, site, shape="box")
    with pytest.raises(ValueError, match="softness"):
        Reveal(scene.camera, site, softness=2)
    with pytest.raises(TypeError, match="Camera"):
        Reveal(protein, site)
    with pytest.raises(ValueError, match="No open cutaway"):
        ProteinScene().play(Conceal(Camera()))


def test_tunnel_axis_stays_with_the_molecule(hemoglobin):
    p = hemoglobin
    scene = framed(p)
    scene.add(p)
    heme = p.select(chain="B", resname="HEM")
    scene.play(Reveal(scene.camera, heme, shape="tunnel"), run_time=1)
    scene.play(scene.camera.animate.orbit(0.8, 0.2), run_time=1)
    scene.seek(1.0)
    center, keep, window, outer, axis = scene.camera.cutaway_geometry()
    np.testing.assert_allclose(axis, (scene.camera.eye - center) / np.linalg.norm(scene.camera.eye - center))
    assert keep * scene.camera.cutaway_surface_keep <= outer
    scene.seek(2.0)
    later = scene.camera.cutaway_geometry()
    np.testing.assert_allclose(later[4], axis, atol=1e-9)  # The camera moved; the tunnel did not.
    assert later[3] == pytest.approx(outer)
    # The tunnel ends where the centerline reaches open space: no atoms just beyond it.
    m = p.model_matrix
    xyz = p.positions @ m[:3, :3].T + m[:3, 3] - center
    height = xyz @ axis
    lateral = np.linalg.norm(xyz - height[:, None] * axis, axis=1)
    beyond = (lateral < 6.0) & (height > outer) & (height < outer + 6.0)
    assert not beyond.any()


def test_depth_cue_animation_and_clearest_view(protein):
    scene = framed(protein)
    scene.add(protein)
    scene.camera.depth_cue = 0.2
    scene.play(scene.camera.animate.orbit(0.5).depth_cue(0.8), run_time=2)
    scene.seek(1.0)
    assert scene.camera.depth_cue == pytest.approx(0.5)
    with pytest.raises(ValueError, match="depth_cue"):
        scene.camera.animate.depth_cue(-1)
    site = protein.select(residues=(23, 34))
    theta, phi = scene.camera.clearest_view(site, max_elevation=0.8)
    assert np.isfinite([theta, phi]).all() and abs(phi) <= 0.8


def test_thread_wires_land_on_the_backbone(hemoglobin):
    p = hemoglobin
    scene = framed(p)
    thread = Thread(p, scene.camera, settle=0.2, seed=2)
    scene.play(thread, run_time=4)
    wire = thread.wire
    assert len(wire.counts) == len(p.topology.chains) == 4
    scene.seek(0.0)
    assert p.opacity == 0 and wire.opacity == 0
    scene.seek(4 * 0.8)
    # Threading is complete: each wire runs from the N terminus (head) to the C terminus.
    starts = np.cumsum([0, *wire.counts])
    for k, chain in enumerate(p.topology.chains):
        points = wire.positions[starts[k] : starts[k + 1]]
        first = p.positions[p.topology.residues[chain[0]].trace_atom]
        last = p.positions[p.topology.residues[chain[-1]].trace_atom]
        np.testing.assert_allclose(points[0], first, atol=1e-3)
        np.testing.assert_allclose(points[-1], last, atol=1e-3)
        length = _arc_length(points)[-1]
        assert length == pytest.approx(thread.strands[k].length, rel=0.02)
    assert wire.opacity == 1 and p.opacity == 0
    scene.seek(4.0)
    assert p.opacity == 1 and wire.opacity == 0
    # Seeking is deterministic.
    scene.seek(1.3)
    first = wire.positions.copy()
    scene.seek(3.9)
    scene.seek(1.3)
    np.testing.assert_array_equal(wire.positions, first)


def test_unthread_mirrors_thread_and_stagger_orders_chains(hemoglobin):
    p = hemoglobin
    camera = framed(p).camera
    forward, backward = Thread(p, camera, seed=5), Unthread(p, camera, seed=5)
    for anim in (forward, backward):
        anim.run_time, anim.start_time = 1.0, 0.0
        anim.bind()
    for alpha in (0.1, 0.5, 0.9):
        forward.apply(alpha)
        a = forward.wire.positions.copy()
        backward.apply(1 - alpha)
        np.testing.assert_allclose(backward.wire.positions, a)
    staggered = Thread(p, camera, stagger=0.2, seed=5)
    staggered.run_time, staggered.start_time = 1.0, 0.0
    staggered.bind()
    staggered.apply(0.15)
    strengths = staggered.wire.glow_heads[2]
    assert strengths[0] > 0 and strengths[-1] == 0  # The first chain flies; the last waits.
    rows = staggered.wire._glow()
    assert rows.shape == (4, 8) and rows[0, 7] > 0 and rows[-1, 7] == 0
    with pytest.raises(ValueError, match="stagger"):
        Thread(p, camera, stagger=0.4)
    with pytest.raises(ValueError, match="settle"):
        Thread(p, camera, settle=1.0)
    with pytest.raises(TypeError, match="easing"):
        Thread(p, camera, easing=0.5)
    with pytest.raises(ValueError, match="glow"):
        Thread(p, camera, glow=-1)


def test_thread_wire_colors_follow_residue_tints(hemoglobin):
    p = hemoglobin
    p.select(chain="A").set_color("#ff0000")
    thread = Thread(p, framed(p).camera, settle=0.0, glow_color="#00ff00", glow_brightness=2.0)
    thread.run_time, thread.start_time = 1.0, 0.0
    thread.bind()
    thread.apply(1.0)
    first = thread.wire._metadata_override[: thread.wire.counts[0]]
    red = np.tile([1.0, 0.0, 0.0], (len(first) - 10, 1))
    np.testing.assert_allclose(first[10:, :3], red, atol=1e-6)  # Behind the bright tip.
    heads = thread.wire.glow_heads
    np.testing.assert_allclose(heads[1], np.tile([0, 1, 0], (4, 1)))
    assert np.all(heads[2] == pytest.approx(2.0 * 0.4))


@pytest.mark.gpu
def test_gpu_cutaway_glow_and_tunnel(hemoglobin):
    import wgpu

    from proteinmotion.renderer import Renderer

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter available")
    p = hemoglobin
    heme = p.select(chain="B", resname="HEM")
    scene = framed(p)
    scene.camera.focus(heme, margin=2.5, aspect=1.5, follow=False)
    scene.add(p)
    scene.play(Reveal(scene.camera, heme, window=1.4), run_time=1)
    scene.play(Conceal(scene.camera), run_time=1)
    scene.play(Reveal(scene.camera, heme, window=1.4, shape="tunnel"), run_time=1)
    with Renderer(384, 256, msaa=1) as renderer:
        closed = scene.render_frame(0, renderer=renderer)
        cut = scene.render_frame(1, renderer=renderer)
        assert np.abs(cut.astype(int) - closed.astype(int)).sum() > 20000
        np.testing.assert_array_equal(scene.render_frame(2, renderer=renderer), closed)
        tunnel = scene.render_frame(3, renderer=renderer)
        assert np.abs(tunnel.astype(int) - cut.astype(int)).sum() > 1000
    # The glow pass brightens pixels around the wire heads.
    lit, dark = (ProteinScene(width=384, height=256) for _ in range(2))
    for s, glow in ((lit, 14.0), (dark, 0.0)):
        protein = Protein.from_file(DATA / "4hhb.cif").cartoon().center()
        s.camera.frame(protein, aspect=1.5)
        s.play(Thread(protein, s.camera, glow=glow, glow_brightness=2.0, seed=1), run_time=2)
    with Renderer(384, 256, msaa=1) as renderer:
        a = lit.render_frame(1.4, renderer=renderer).astype(int)
        b = dark.render_frame(1.4, renderer=renderer).astype(int)
    assert (a[:, :, :3] - b[:, :, :3]).min() >= 0
    assert (a[:, :, :3] - b[:, :, :3]).sum() > 5000
