"""Vector contour writing, amino-acid identity, moving anchors and native GPU overlays."""

from dataclasses import replace

import numpy as np
import pytest

from proteinmotion import (
    Callout,
    FadeIn,
    FadeOut,
    PlayTrajectory,
    Protein,
    Scene,
    Text,
    Trajectory,
    Unwrite,
    Write,
    linear,
)
from proteinmotion.annotations import project_region
from proteinmotion.camera import Camera
from proteinmotion.text_geometry import _triangulate, layout_text


def test_shaping_kerning_ligatures_unicode_and_cache():
    assert layout_text("office").glyph_count < layout_text("office", ligatures=False).glyph_count
    assert layout_text("AV").width < layout_text("A").width + layout_text("V").width
    geometry = layout_text("α helix · β sheet\nGly 76")
    assert geometry is layout_text("α helix · β sheet\nGly 76")
    assert geometry.height > 36
    assert not geometry.fill.flags.writeable and not geometry.strokes.flags.writeable
    for index in range(geometry.glyph_count):
        segments = geometry.strokes[geometry.strokes[:, 6] == index]
        assert segments[0, 4] == 0 and segments[-1, 5] == 1
        np.testing.assert_allclose(segments[1:, 4], segments[:-1, 5])
    for text in ("", "  ", "\n"):
        assert layout_text(text).glyph_count == 0
    with pytest.raises(ValueError, match="glyph"):
        Text("\U0010ffff")


def test_triangulation_keeps_holes_islands_and_disconnected_marks():
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    hole = np.array([[2, 2], [8, 2], [8, 8], [2, 8]])
    island = np.array([[4, 4], [6, 4], [6, 6], [4, 6]])
    accent = np.array([[12, 0], [14, 0], [14, 2], [12, 2]])
    vertices = _triangulate([hole[::-1], accent, outer, island[::-1]]).reshape(-1, 3, 2)
    a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2
    assert area == pytest.approx(100 - 36 + 4 + 4)


def test_write_unwrite_timeline_auto_add_and_conflicts():
    text = Text("Backbone")
    scene = Scene()
    scene.wait(0.5)
    scene.play(Write(text, lag_ratio=0.15, stroke_width=2), run_time=2)
    scene.play(Unwrite(text), run_time=1)
    assert text not in scene.seek(0.4)
    for time, expected in [(0.5, 0), (1.5, 0.5), (2.5, 1), (3, 0.5), (3.5, 0), (1.5, 0.5)]:
        scene.seek(time)
        assert text._write == expected
    with pytest.raises(ValueError, match="write"):
        Scene().play(Write(text), Unwrite(text))
    Scene().play(Write(text), FadeIn(text))
    Scene().play(Write(Text("")))
    with pytest.raises(ValueError, match="lag_ratio"):
        Write(text, lag_ratio=-1)
    with pytest.raises(ValueError, match="stroke_width"):
        Write(text, stroke_width=0)


def test_screen_position_and_label_offset_animation(protein):
    text = Text("A", position=(0.1, 0.2))
    label = protein.select(chain="A", residues=8).label(offset=(20, -30))
    scene = Scene()
    scene.play(
        text.animate.move_to((0.3, 0.4)).shift((0.1, 0)),
        label.animate.set_offset((40, -10)),
        run_time=2,
        rate_func=linear,
    )
    scene.seek(1)
    np.testing.assert_allclose(text.position, (0.25, 0.3))
    np.testing.assert_allclose(label.offset, (30, -20))
    with pytest.raises(ValueError, match="follow atoms"):
        label.move_to((0.2, 0.3))
    with pytest.raises(ValueError, match="follow atoms"):
        label.animate.move_to((0.2, 0.3))


def test_residue_names_author_numbering_insertion_codes_and_ca(protein):
    region = protein.select(chain="A", residues=8)
    label = region.label()
    assert label.title.text == "A · Leu 8"
    assert protein.topology.atoms[label.region.atom_indices[0]].name == "CA"
    assert region.label(format="one_letter", include_chain=False).title.text == "L 8"
    index = region.residue_indices[0]
    residues = list(protein.topology.residues)
    residues[index] = replace(residues[index], resid=108, icode="B")
    p = Protein(replace(protein.topology, residues=tuple(residues)), protein.positions)
    from proteinmotion import Region

    assert Region(p, region.atom_indices).label().title.text == "A · Leu 108B"
    with pytest.raises(ValueError, match="exactly one residue"):
        protein.select(residues=(8, 10)).label()


def test_callout_anchor_follows_state_transform_camera_and_parent_visibility(protein):
    region = protein.select(chain="A", residues=(23, 34), atoms="CA")
    callout = region.callout("α helix", subtitle="Residues 23–34", position=(0.75, 0.3), tip="arrow")
    scene = Scene()
    scene.add(protein)
    scene.camera.frame(protein)
    frames = np.array([protein.positions, protein.positions * [1.3, 0.8, 1.1]])
    scene.play(
        Write(callout),
        PlayTrajectory(protein, Trajectory(frames)),
        protein.animate.rotate(0.5).shift([3, -2, 1]),
        scene.camera.animate.orbit(0.4),
        run_time=2,
    )
    scene.seek(1.6)
    layout = callout.layout(scene.camera, 1920, 1080)
    anchor = layout.anchor.copy()
    # The world centroid projects to the same point as a tiny selection placed there.
    clip = scene.camera.matrix(16 / 9) @ np.r_[region.world_positions.mean(0), 1]
    np.testing.assert_allclose(anchor, [960 * (1 + clip[0] / clip[3]), 540 * (1 - clip[1] / clip[3])])
    np.testing.assert_allclose(layout.text[0].origin, [1440, 324])
    scene.seek(0)
    assert np.linalg.norm(callout.layout(scene.camera, 1920, 1080).anchor - anchor) > 2
    scene.seek(2)
    scene.seek(1.6)
    np.testing.assert_array_equal(callout.layout(scene.camera, 1920, 1080).anchor, anchor)
    protein.set_opacity(0.3)
    assert callout.layout(scene.camera, 1920, 1080).text[0].opacity == pytest.approx(0.3)
    protein.shift([10000, 0, 0])
    assert not callout.layout(scene.camera, 1920, 1080).text


def test_label_group_offsets_overlap_and_scaling(protein):
    labels = protein.label_residues(chain="A", residues=[8, 44, 70], offsets={44: (120, -40)})
    assert [label.title.text for label in labels.labels] == ["A · Leu 8", "A · Ile 44", "A · Val 70"]
    assert labels.fixed == (False, True, False)
    camera = Camera().frame(protein)
    full = labels.layout(camera, 1920, 1080)
    half = labels.layout(camera, 960, 540)
    for a, b in zip(full.text, half.text):
        np.testing.assert_allclose(a.origin / 2, b.origin)
    assert [p.glyph_offset for p in full.text] == [
        0,
        labels.labels[0].glyph_count,
        sum(label.glyph_count for label in labels.labels[:2]),
    ]
    np.testing.assert_allclose(
        full.text[1].origin, project_region(labels.labels[1].region, camera, 1920, 1080) + [120, -40]
    )
    # For overlapping anchors, automatic placement changes the later label's offset.
    points = protein.positions.copy()
    for label in labels.labels:
        points[label.region.atom_indices] = points[labels.labels[0].region.atom_indices]
    protein.set_positions(points)
    loose = protein.label_residues(chain="A", residues=[8, 44, 70], avoid_overlap=False)
    auto = protein.label_residues(chain="A", residues=[8, 44, 70])
    assert not np.array_equal(
        auto.layout(camera, 1920, 1080).text[1].origin, loose.layout(camera, 1920, 1080).text[1].origin
    )


@pytest.mark.gpu
@pytest.mark.parametrize("msaa", [1, 4])
def test_gpu_contours_then_fill_glyph_lag_reverse_holes_and_fade(msaa):
    from proteinmotion.renderer import Renderer

    text = Text("OO", font_size=400, position=(0.1, 0.2), color="#ffffff")
    write = Write(text, lag_ratio=1, stroke_width=3)
    camera = Camera()
    background = np.zeros(3)
    with Renderer(640, 360, msaa=msaa) as renderer:

        def frame(progress):
            write.apply(progress)
            return renderer.render([text], camera, background)[:, :, 0]

        empty, outline, first, second_outline, filled = [frame(t) for t in (0, 0.25, 0.5, 0.75, 1)]
        assert not empty.any()
        assert 100 < outline.sum() < first.sum()
        assert first.sum() < second_outline.sum() < filled.sum()
        # Interior holes remain empty, including after the glyph fills.
        strokes = text.geometry.strokes
        for i in (0, 1):
            points = strokes[strokes[:, 6] == i, :2]
            center = (points.min(0) + points.max(0)) / 2 / 3 + [64, 72]
            x, y = np.round(center).astype(int)
            assert not filled[y - 4 : y + 5, x - 4 : x + 5].any()
        Write(text, lag_ratio=1, reverse=True).apply(0.5)
        reverse = renderer.render([text], camera, background)[:, :, 0]
        np.testing.assert_array_equal(np.maximum(first, reverse), filled)
        write.apply(1)
        text.set_opacity(0.5)
        faded = renderer.render([text], camera, background)[:, :, 0]
        np.testing.assert_allclose(faded, filled * 0.5, atol=2)
        assert len(renderer._overlay.text) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("msaa", [1, 4])
def test_gpu_moving_annotations_transparency_deterministic_seek_and_cached_buffers(protein, msaa):
    from proteinmotion.renderer import Renderer

    scene = Scene(width=640, height=360, msaa=msaa)
    protein.set_opacity(0.65)
    scene.add(protein)
    scene.camera.frame(protein, aspect=16 / 9)
    callout = Callout(protein.select(residues=(23, 34)), "α helix", position=(0.72, 0.3), font_size=48)
    labels = protein.label_residues(residues=[8, 44, 70], font_size=40)
    scene.play(Write(callout), Write(labels), protein.animate.rotate(0.6), run_time=2)
    scene.play(FadeOut(labels), run_time=1)
    with Renderer(640, 360, msaa=msaa) as renderer:
        frames = {t: scene.render_frame(t, renderer=renderer) for t in (0, 0.5, 1.1, 2, 2.5, 3)}
        buffers = [(g.fill, g.strokes) for g in renderer._overlay.text.values()]
        for t in (2.5, 0.5, 3, 0, 1.1, 2):
            np.testing.assert_array_equal(scene.render_frame(t, renderer=renderer), frames[t])
        assert buffers == [(g.fill, g.strokes) for g in renderer._overlay.text.values()]
        assert len(renderer._overlay.text) == 4
        assert renderer.transparency_pipelines is not None
