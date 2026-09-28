"""Authoring workflows: defaults, composition, and seek-stable public APIs."""

from dataclasses import replace

import numpy as np
import pytest

from proteinmotion import (
    Colorize,
    Distance,
    FadeIn,
    FadeOut,
    PlayTrajectory,
    ProteinScene,
    Representation,
    SetOpacity,
    StudioLook,
    Thread,
    Trajectory,
    smooth,
)
from proteinmotion.styling import current_details, current_opacities, current_tints


def test_fade_roundtrip_restores_intended_opacity(protein):
    protein.set_opacity(0.35)
    scene = ProteinScene().add(protein)
    scene.play(FadeOut(protein))
    scene.play(FadeIn(protein))
    for time, opacity in [(2, 0.35), (0, 0.35), (1, 0), (1.5, 0.175), (2, 0.35)]:
        scene.seek(time)
        assert protein.opacity == pytest.approx(opacity)


def test_setters_are_timeline_events_and_seeking_is_not_an_edit(protein):
    scene = ProteinScene().add(protein)
    protein.set_opacity(0.25)
    scene.wait(1)
    protein.set_opacity(0.75)
    scene.wait(1)
    for time, expected in [(1.5, 0.75), (0.5, 0.25), (2, 0.75)]:
        scene.seek(time)
        assert protein.opacity == expected
    scene.seek(0.5)
    scene.play(protein.animate.shift((1, 0, 0)))
    scene.seek(3)
    assert protein.opacity == 0.75


def test_opacity_scope_and_fluent_mixed_operations(protein):
    scene = ProteinScene().add(protein.set_opacity(0.2))
    region = protein.select(residue_range=(23, 34))
    scene.play(SetOpacity(protein, 1), region.animate.set_color("#ff0000").set_opacity(0.4).show_atoms())
    scene.seek(1)
    assert protein.opacity == 1
    np.testing.assert_allclose(current_opacities(protein)[region.atom_indices], 0.4)
    np.testing.assert_allclose(current_details(protein)[region.atom_indices], 1)
    scene.play(protein.animate.shift((3, 0, 0)).set_color("#00ff00").show_atoms())
    scene.seek(2)
    np.testing.assert_allclose(current_tints(protein)[:, :3], np.tile([0, 1, 0], (len(protein.positions), 1)))
    scene.play(SetOpacity(protein, 1, scope="residues"))
    scene.seek(3)
    np.testing.assert_allclose(current_opacities(protein), 1)


def test_appearance_easing_and_stagger_fit(protein):
    region = protein.select(residue_range=(23, 34))
    scene = ProteinScene().add(protein)
    scene.play(Colorize(region, "#ff0000", easing=smooth, stagger=0.6), run_time=0.1)
    scene.play(Colorize(region, "#00ff00", easing=lambda t: t * t), run_time=1)
    scene.seek(0.6)
    np.testing.assert_allclose(
        current_tints(protein)[region.atom_indices, :3], np.tile([0.75, 0.25, 0], (len(region), 1)), atol=1e-6
    )
    scene.seek(1.1)
    np.testing.assert_allclose(current_tints(protein)[region.atom_indices, 1], 1)


def test_timed_actions_bind_from_correct_state(protein):
    from proteinmotion.timeline import AnimationGroup

    scene = ProteinScene().add(protein)
    initial = protein.position.copy()
    scene.play(
        AnimationGroup(
            protein.animate.shift((2, 0, 0)),
            protein.animate.shift((0, 4, 0)),
            durations=[1, 2],
            offsets=[0, 1],
        )
    )
    assert scene.duration == 3
    for time, offset in [(3, [2, 4, 0]), (0.5, [1, 0, 0]), (2, [2, 2, 0]), (0, [0, 0, 0])]:
        scene.seek(time)
        np.testing.assert_allclose(protein.position, initial + offset)


def test_timed_concurrent_actions_and_conflicts(protein):
    scene = ProteinScene().add(protein)
    scene.play(protein.animate.shift((2, 0, 0)).during(2), Colorize(protein, "#ff0000").during(0.5, delay=1))
    assert scene.duration == 2
    scene.seek(0.5)
    np.testing.assert_allclose(current_tints(protein)[:, 3], 0)
    scene.seek(2)
    np.testing.assert_allclose(current_tints(protein)[:, 3], 1)
    with pytest.raises(ValueError, match="Concurrent"):
        ProteinScene().play(
            protein.animate.shift((1, 0, 0)).during(2), protein.animate.rotate(1).during(1, delay=0.5)
        )


def test_representation_options_and_style(protein):
    from proteinmotion.styles import MolecularStyle

    scene = ProteinScene().add(protein)
    scene.play(Representation(protein, "surface", kind="vdw", grid_spacing=0.9))
    scene.play(Representation(protein, MolecularStyle("ball_and_stick", {"atom_scale": 0.45})))
    scene.seek(1)
    assert protein._surface_options.kind == "vdw"
    scene.seek(2)
    assert protein.atom_scale == 0.45
    scene.seek(0)
    assert protein.surface_opacity == 0


def test_preset_replacement_and_hex_colors():
    original = StudioLook(material="medium", effects="film", key=2.3)
    changed = replace(original, material="rough", effects="clean")
    assert changed.roughness == 0.85 and changed.grain == 0 and changed.key == 2.3
    assert original.material == "medium" and original.grain > 0
    original.material = "rough"
    assert original.roughness == 0.85
    original.effects = "clean"
    assert original.grain == 0
    named = StudioLook(grain="fine").with_presets(effects="clean")
    assert named.grain > 0
    color = StudioLook(rim_color="#ff8800").rim_color
    np.testing.assert_allclose(color, [1, 136 / 255, 0])


def test_selection_algebra_filters_optional_and_live(protein):
    helix = protein.select(residue_range=(23, 34))
    carbons = protein.select(element="C")
    assert set((helix & carbons).atom_indices) == set(helix.atom_indices) & set(carbons.atom_indices)
    assert len(helix | ~helix) == len(protein.positions)
    assert not (helix - helix)
    assert protein.select(ions=True, required=False) is None
    assert len(helix.filter(element="C")) == len(helix & carbons)
    assert protein.summary()["chains"] == ["A"]
    ligand = protein.select(residues=23)
    live = protein.select(within=4, of=ligand, updating=True)
    frozen = live.freeze()
    xyz = protein.positions.copy()
    xyz[ligand.atom_indices] += 1000
    protein.set_positions(xyz)
    assert len(frozen) > 0 and len(live) == 0


def test_camera_inherits_aspect_and_composes(protein):
    scene = ProteinScene(width=600, height=900).add(protein)
    scene.camera.frame(protein, screen_position=(0.7, 0.4))
    center = (protein.positions.min(0) + protein.positions.max(0)) / 2
    center = protein.model_matrix @ np.r_[center, 1]
    clip = scene.camera.matrix(600 / 900) @ center
    xy = clip[:2] / clip[3]
    np.testing.assert_allclose([(xy[0] + 1) / 2, (1 - xy[1]) / 2], [0.7, 0.4], atol=1e-6)
    scene.focus(protein, screen_position=(0.3, 0.6))
    scene.seek(scene.duration)
    clip = scene.camera.matrix(600 / 900) @ center
    xy = clip[:2] / clip[3]
    np.testing.assert_allclose([(xy[0] + 1) / 2, (1 - xy[1]) / 2], [0.3, 0.6], atol=1e-6)
    thread = Thread(protein)
    ProteinScene(width=500, height=800).play(thread)
    assert thread.camera is not None and thread.aspect == 500 / 800


def test_distance_and_plot_share_endpoints_and_explicit_trajectory(protein):
    a, b = protein.select(residues=23), protein.select(residues=34)
    xyz = protein.positions.copy()
    later = xyz.copy()
    later[b.atom_indices] += [2, 0, 0]
    trajectory = Trajectory([xyz, later], topology=protein.topology, times=[0, 25], time_unit="ns")
    ruler = Distance(a, b)
    plot = ruler.plot(trajectory=trajectory)
    assert plot.current_value == pytest.approx(ruler.distance)
    assert plot.xlabel == "Time (ns)"
    scene = ProteinScene().add(protein, plot)
    scene.play(PlayTrajectory(protein, trajectory), run_time=2)
    scene.seek(1)
    assert plot.cursor == 12.5 and plot.current_value == pytest.approx(ruler.distance)
    aligned = trajectory.aligned(selection=protein.select(atoms="CA"))
    np.testing.assert_array_equal(aligned.times, trajectory.times)
    assert aligned.time_unit == "ns"


def test_scene_render_settings_and_cli_defaults(tmp_path):
    from proteinmotion.cli import load_scene

    path = tmp_path / "movie.py"
    path.write_text(
        "from proteinmotion import ProteinScene, StudioLook\n"
        "class Movie(ProteinScene):\n"
        '    renderer = "studio"\n'
        '    look = StudioLook(effects="film")\n'
        "    def __init__(self, **kwargs):\n"
        '        kwargs.setdefault("fps", 60)\n'
        "        super().__init__(**kwargs)\n"
    )
    scene = load_scene(path, fps=None)
    assert scene.fps == 60
    backend, look, _ = scene._render_settings(None, None, None)
    assert backend == "studio" and look.grain > 0
    assert scene._render_settings("native", None, None)[1] is None


def test_text_styles_wrapping_groups_and_timeline_replacement():
    from proteinmotion import Text, TextGroup, TextStyle, Write
    from proteinmotion.camera import Camera

    style = TextStyle(font_size=28, color="#ff8800")
    title = Text("A longer heading with words that wrap", style=style, max_width=180)
    assert title.geometry.width <= 180
    group = TextGroup(title, Text("Ligand contacts", style=style), gap=12)
    layout = group.layout(Camera(), 960, 540)
    assert layout.text[1].origin[1] > layout.text[0].origin[1]
    scene = ProteinScene().add(group)
    scene.play(Write(group, delay_seconds=0.005, easing="smooth"))
    scene.play(FadeOut(group))
    scene.play(FadeIn(group))
    scene.seek(3)
    assert group.opacity == 1
    caption = Text("Before")
    scene.add(caption)
    scene.play(caption.animate.set_text("After").move_to((0.2, 0.3)))
    for time, text in [(4, "After"), (3, "Before"), (4, "After")]:
        scene.seek(time)
        assert caption.text == text


def test_ligand_labels_and_hydrogen_diagnostics():
    from pathlib import Path

    from proteinmotion import Protein, ResidueLabels

    protein = Protein.from_file(Path(__file__).resolve().parents[1] / "examples/data/2dri.cif", chains="A")
    ligand = protein.select(ligands=True)
    labels = ResidueLabels(ligand)
    assert len(labels.labels) == len(ligand.residue_indices)
    report = protein.hydrogen_bonds().diagnostics()
    assert report["missing_hydrogen_donors"]
    assert report["virtual_backbone_donors"]
    assert len(report["explicit_donors"]) == 0
    assert (
        sum(
            len(report[key])
            for key in ("missing_hydrogen_donors", "virtual_backbone_donors", "explicit_donors")
        )
        == report["donors"]
    )


def test_interaction_between_filter_and_truncation(protein):
    analysis = protein.hydrogen_bonds()
    pairs = analysis.pairs
    assert len(pairs) > 1
    with pytest.warns(UserWarning, match="Showing 1 of"):
        capped = analysis.highlight(max_pairs=1)
        capped._refresh()
    assert capped.total_pairs > len(capped.visible_pairs)
    a = protein.select(residues=protein.topology.atoms[pairs[0].a].resid)
    b = protein.select(residues=protein.topology.atoms[pairs[0].b].resid)
    between = analysis.highlight(between=(a, b))
    between._refresh()
    assert between.visible_pairs
    left, right = set(a.atom_indices), set(b.atom_indices)
    assert all(
        (p.a in left and p.b in right) or (p.b in left and p.a in right) for p in between.visible_pairs
    )


def test_density_attachment_and_fluent_contour(protein):
    from proteinmotion import DensityMap

    xyz = protein.positions
    origin = xyz.min(0) - 5
    shape = np.ceil(xyz.max(0) - origin + 5).astype(int)
    grid = np.indices(shape).transpose(1, 2, 3, 0) + origin
    values = np.exp(-np.square(grid - xyz.mean(0)).sum(-1) / 50)
    density = DensityMap(values, origin=origin)
    cropped = density.crop(protein.select(residue_range=(23, 34)))
    surface = cropped.isosurface(0.1, units="absolute")
    assert surface.follow is protein
    plane = cropped.slice(samples=24)
    assert plane.follow is protein and plane.resolution == 24
    scene = ProteinScene().add(surface)
    scene.play(surface.animate.set_level(0.2).set_opacity(0.7))
    scene.seek(1)
    assert surface.level == 0.2 and surface.opacity == 0.7


def test_explicit_plot_region_names_and_common_scene_easing(protein):
    from proteinmotion import ContactMap, SequenceTrack

    region = protein.select(residue_range=(23, 34))
    sequence = SequenceTrack(protein, display_region=region, highlight_region=region)
    contact = ContactMap(protein, display_region=region, highlight_region=region)
    assert len(sequence.ids) == len(contact.ids) == 12
    scene = ProteinScene().add(protein)
    scene.play(Colorize(region, "#ff0000", stagger=0.5), protein.animate.shift((2, 0, 0)), easing="linear")
    scene.seek(0.25)
    np.testing.assert_allclose(current_tints(protein)[region.atom_indices[0], 3], 0.5)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["native", "studio"])
def test_new_authoring_workflow_renders_and_seeks(protein, backend):
    from proteinmotion import MolecularStyle, Text, TextGroup

    scene = ProteinScene(
        width=320,
        height=240,
        renderer=backend,
        look=StudioLook(effects="clean") if backend == "studio" else None,
    )
    scene.add(protein)
    protein.set_opacity(0.8)
    scene.camera.frame(protein, screen_position=(0.58, 0.55))
    scene.add(TextGroup(Text("ProteinMotion", font_size=34), Text("Usability test", font_size=24)))
    region = protein.select(residue_range=(23, 34))
    scene.play(region.animate.set_color("#f08040").show_atoms(), run_time=0.2)
    scene.play(
        Representation(protein, MolecularStyle("surface", {"kind": "vdw", "grid_spacing": 1.0})), run_time=0.2
    )
    first = scene.render_frame(0.1)
    last = scene.render_frame(0.4)
    repeated = scene.render_frame(0.1)
    np.testing.assert_array_equal(first, repeated)
    assert np.abs(first.astype(float) - last).sum() > 1000


def test_delayed_entrances_are_hidden_before_start():
    from proteinmotion import Text, Write

    fade = Text("Fading")
    write = Text("Writing")
    scene = ProteinScene()
    scene.play(FadeIn(fade).during(1, delay=1), Write(write).during(1, delay=1))
    scene.seek(0.5)
    assert fade.opacity == 0 and write.text_progress == 0
    scene.seek(2)
    assert fade.opacity == 1 and write.text_progress == 1
    scene.seek(0)
    assert fade.opacity == 0 and write.text_progress == 0


def test_framing_and_orbit_compose_in_one_chain(protein):
    scene = ProteinScene().add(protein)
    theta = scene.camera.theta
    scene.play(scene.camera.animate.frame(protein).orbit(0.5).zoom(1.2))
    expected = scene.camera.snapshot()
    assert scene.camera.theta == theta + 0.5
    scene.seek(0.2)
    scene.seek(1)
    assert scene.camera.theta == theta + 0.5
    assert scene.camera.distance == pytest.approx(expected["distance"])


def test_during_shorthand_allows_sequential_writes(protein):
    scene = ProteinScene().add(protein)
    initial = protein.position.copy()
    scene.play(
        protein.animate.shift((2, 0, 0)).during(1), protein.animate.shift((0, 4, 0)).during(1, delay=1)
    )
    scene.seek(0.5)
    np.testing.assert_allclose(protein.position, initial + [1, 0, 0])
    scene.seek(1.5)
    np.testing.assert_allclose(protein.position, initial + [2, 2, 0])
    scene.seek(2)
    np.testing.assert_allclose(protein.position, initial + [2, 4, 0])


def test_md_timestamps_are_lazy_strided_and_preserved(tmp_path, structure_path):
    import gemmi

    mda = pytest.importorskip("MDAnalysis")
    structure = gemmi.read_structure(str(structure_path))
    structure.remove_waters()
    topology = tmp_path / "topology.pdb"
    structure.write_pdb(str(topology))
    universe = mda.Universe(str(topology))
    path = tmp_path / "timed.xtc"
    with mda.Writer(str(path), n_atoms=len(universe.atoms)) as writer:
        for index in range(5):
            universe.trajectory.ts.time = index * 2.5
            writer.write(universe.atoms)
    trajectory = Trajectory.from_mdanalysis(topology, path, stride=2)
    assert trajectory._time_reader is not None
    before = trajectory.frame(1).copy()
    np.testing.assert_allclose(trajectory.times, [0, 5, 10])
    assert trajectory._time_reader is None
    np.testing.assert_array_equal(trajectory.frame(1), before)
    np.testing.assert_array_equal(trajectory.aligned().times, [0, 5, 10])


def test_grouped_focus_tracks_delayed_molecular_motion(protein):
    from proteinmotion import Focus

    scene = ProteinScene().add(protein)
    scene.camera.frame(protein)
    initial = scene.camera.target.copy()
    scene.play(
        Focus(scene.camera, protein, easing="linear").during(2),
        protein.animate.shift((10, 0, 0)).during(1, delay=0.5),
    )
    scene.seek(1)
    positions = protein.positions @ protein.model_matrix[:3, :3].T + protein.position
    center = (positions.min(0) + positions.max(0)) / 2
    np.testing.assert_allclose(scene.camera.target, (initial + center) / 2)
