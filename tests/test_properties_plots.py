"""Numerical meaning, timeline synchronization, and property-renderer integration."""

from dataclasses import replace

import numpy as np
import pytest

from proteinmotion import (
    ColorByProperty,
    Colorize,
    ColorLegend,
    ColorScale,
    ContactMap,
    PlayTrajectory,
    ProteinScene,
    ResidueValues,
    SequenceTrack,
    TimeSeriesPlot,
    Trajectory,
)
from proteinmotion._eevee_geometry import MeshExporter
from proteinmotion.camera import Camera
from proteinmotion.geometry import segments
from proteinmotion.math3d import rotation
from proteinmotion.styling import current_tints


def test_b_factors_and_identity_mapping(protein):
    values = ResidueValues.b_factors(protein)
    assert values.values[0] == pytest.approx(10.38, abs=0.01)  # 1UBQ A:1 CA, file field
    imported = ResidueValues.from_mapping(protein, {("A", 1): 91, ("A", 4, ""): 72}, name="pLDDT")
    assert imported.values[0] == 91 and imported.values[3] == 72
    assert np.isnan(imported.values[1])
    with pytest.raises(ValueError, match="matches 0"):
        ResidueValues.from_mapping(protein, {("X", 1): 10})
    with pytest.raises(ValueError, match="per topology"):
        ResidueValues(protein, [1])
    other = protein.copy()
    other.topology = replace(protein.topology, residues=tuple(reversed(protein.topology.residues)))
    with pytest.raises(ValueError, match="identity/order"):
        other.color_by(values)


def test_scale_clipping_missing_and_fixed_limits():
    scale = ColorScale(0, 10, colors=("#000000", "#ffffff"), missing="#ff0000")
    np.testing.assert_allclose(
        scale.map([-1, 5, 20, np.nan]), [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [1, 0, 0]]
    )
    assert ColorScale.from_values([3, 3]).vmax == 3.5
    for args in ((0, 0), (2, 1), (0, np.nan)):
        with pytest.raises(ValueError):
            ColorScale(*args)
    with pytest.raises(ValueError, match="finite value"):
        ColorScale.from_values([np.nan])


def test_rmsf_alignment_mean_and_stride(protein):
    xyz = protein.positions
    frames = [xyz, xyz @ rotation(0.6, (0, 0, 1)).T + [5, 3, 2], xyz + [2, -4, 1]]
    trajectory = Trajectory(frames, topology=protein.topology)
    values = ResidueValues.rmsf(protein, trajectory)
    assert np.nanmax(values.values) < 1e-5
    translated = Trajectory([xyz - [2, 0, 0], xyz + [2, 0, 0]], topology=protein.topology)
    np.testing.assert_allclose(ResidueValues.rmsf(protein, translated, align=False).values[:76], 2, atol=1e-5)
    with pytest.raises(ValueError, match="stride"):
        ResidueValues.rmsf(protein, stride=0)


def test_property_animation_seek_stagger_and_conflicts(protein):
    values = np.linspace(0, 1, len(protein.topology.residues))
    scale = ColorScale(0, 1, colors=("#0000ff", "#ff0000"))
    original = segments(protein)
    scene = ProteinScene().add(protein)
    scene.wait(0.5)
    scene.play(
        ColorByProperty(protein, values, scale=scale, thickness=(0.5, 2), residue_delay=0.01), run_time=3
    )
    scene.seek(3.5)
    atom_ids = np.array([a.residue_index for a in protein.topology.atoms])
    np.testing.assert_allclose(current_tints(protein)[:, :3], scale.map(values)[atom_ids], atol=1e-6)
    np.testing.assert_allclose(protein._cartoon_scale, 0.5 + 1.5 * values)
    assert not np.array_equal(segments(protein)[:, 4:8], original[:, 4:8])
    scene.seek(0.6)
    assert protein._cartoon_scale[-1] == 1  # last residue has not begun
    early = current_tints(protein).copy()
    scene.seek(3.5)
    scene.seek(0.6)
    np.testing.assert_array_equal(current_tints(protein), early)
    with pytest.raises(ValueError, match="color"):
        ProteinScene().play(ColorByProperty(protein, values), Colorize(protein, "#ffffff"))


@pytest.mark.parametrize("representation", ["cartoon", "ribbon", "ball_and_stick", "surface"])
def test_property_color_in_every_export_representation(protein, representation):
    getattr(protein, representation)()
    protein.color_by(
        np.zeros(len(protein.topology.residues)), scale=ColorScale(0, 1, colors=("#ff0000", "#0000ff"))
    )
    meshes = MeshExporter().meshes(protein)
    for mesh in meshes:
        np.testing.assert_allclose(mesh["colors"], np.tile([1, 0, 0], (len(mesh["colors"]), 1)))


def test_plot_cursor_reverse_easing_seek_and_exact_distance(protein):
    xyz = protein.positions.copy()
    end = xyz.copy()
    selected = protein.select(residues=50)
    end[selected.atom_indices] += [10, 8, 0]
    protein.trajectory = Trajectory([xyz, end], topology=protein.topology)
    a, b = protein.select(residues=5, atoms="CA"), protein.select(residues=50, atoms="CA")
    plot = TimeSeriesPlot.distance(a, b, times=[10, 20])
    clock = TimeSeriesPlot([0, 4], [0, 1])
    scene = ProteinScene().add(protein, plot, clock)
    scene.wait(1)
    scene.play(PlayTrajectory(protein, start=1, end=0, state_easing=lambda t: t * t), run_time=2)
    for time, cursor in ((0, 10), (1, 20), (2, 12.5), (3, 10), (2, 12.5)):
        scene.seek(time)
        assert plot.cursor == cursor
        assert clock.cursor == time
        assert plot.current_value == pytest.approx(np.linalg.norm(a.positions.mean(0) - b.positions.mean(0)))
    # Numerical distance of interpolated coordinates differs from interpolating measurements.
    assert abs(plot.current_value - np.interp(plot.cursor, plot.times, plot.values)) > 0.01


def test_panels_contacts_selection_missing_and_gaps(protein):
    region = protein.select(residues=(10, 20))
    contacts = ContactMap(protein, selection=region)
    matrix = contacts.matrix
    assert np.array_equal(matrix, matrix.T) and not matrix.diagonal().any()
    ca = protein.positions[contacts.cas]
    i, j = np.argwhere(matrix)[0]
    assert np.linalg.norm(ca[i] - ca[j]) <= contacts.cutoff
    assert abs(int(contacts.ids[i]) - int(contacts.ids[j])) > contacts.min_separation
    scale = ColorScale(0, 1)
    panels = [
        contacts,
        SequenceTrack(protein, selection=region),
        ColorLegend(scale, title="Confidence"),
        TimeSeriesPlot([0, 1, 2, 3], [0, np.nan, 0.5, 1]),
    ]
    for panel in panels:
        layout = panel.layout(Camera(), 960, 540)
        assert np.isfinite(layout.triangles).all()
        assert layout.triangles.shape[1] == 6
        for line in layout.leaders:
            assert np.isfinite(line.points).all()
    with pytest.raises(ValueError, match="max_residues"):
        ContactMap(protein, max_residues=5)


@pytest.mark.gpu
def test_native_properties_and_panels_seek_pixels(protein):
    from proteinmotion.renderer import Renderer

    scene = ProteinScene(width=480, height=270).add(protein)
    scene.camera.frame(protein)
    values = ResidueValues.b_factors(protein)
    scene.add(SequenceTrack(protein), ColorLegend(ColorScale(0, 50)), TimeSeriesPlot([0, 2], [0, 1]))
    scene.play(ColorByProperty(protein, values, thickness=(0.5, 2)), run_time=2)
    with Renderer(480, 270) as renderer:
        before = scene.render_frame(0, renderer=renderer)
        after = scene.render_frame(2, renderer=renderer)
        again = scene.render_frame(0, renderer=renderer)
    np.testing.assert_array_equal(before, again)
    assert np.abs(after.astype(float) - before).sum() > 10000


def test_plot_rejects_different_played_trajectory(protein):
    protein.trajectory = Trajectory([protein.positions, protein.positions + 1], topology=protein.topology)
    plot = TimeSeriesPlot([0, 1], [1, 2], protein=protein)
    other = Trajectory([protein.positions, protein.positions + 2], topology=protein.topology)
    scene = ProteinScene().add(protein, plot)
    scene.play(PlayTrajectory(protein, other))
    with pytest.raises(ValueError, match="different trajectory"):
        scene.seek(0.5)


def test_sequence_track_element_palette(protein):
    protein.surface(color="element")
    track = SequenceTrack(protein)
    assert np.isfinite(track.layout(Camera(), 640, 360).triangles).all()


def test_long_trace_preserves_nan_gaps():
    values = np.sin(np.arange(10000) / 50)
    values[405:408] = np.nan
    values[412:415] = np.nan
    plot = TimeSeriesPlot(np.arange(len(values)), values)
    layout = plot.layout(Camera(), 1920, 1080)
    trace = next(line for line in layout.leaders if line.points.ndim == 3)
    # The trace's pixel x coordinate is an affine image of sample index.
    x = trace.points[:, :, 0]
    # Both endpoints are finite and retained, independent of the plot's margins.
    left, right = x.min(), x.max()
    index = (x - left) / (right - left) * 9999
    for start, end in index:
        assert not (start < 405 and end > 408)
        assert not (start < 412 and end > 415)
