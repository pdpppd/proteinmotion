"""Heatmaps, AlphaFold confidence, and hydropathy and conservation color presets."""

import json

import numpy as np
import pytest

from proteinmotion import (
    ColorByProperty,
    ColorLegend,
    ColorScale,
    Heatmap,
    Protein,
    ProteinScene,
    ResidueValues,
    SetTorsions,
)
from proteinmotion.camera import Camera
from proteinmotion.conservation import column_conservation, read_alignment
from proteinmotion.geometry import residue_colors
from proteinmotion.styling import current_tints


def test_heatmap_bins_large_matrices_and_validates():
    matrix = np.arange(400.0).reshape(20, 20)
    plot = Heatmap(matrix, max_cells=8, title="Values")
    binned = plot._binned()
    assert binned.shape == (8, 8)
    assert binned[0, 0] == pytest.approx(matrix[:3, :3].mean())
    blocks = np.array_split(np.arange(20), 8)
    assert binned[5, 2] == pytest.approx(matrix[np.ix_(blocks[5], blocks[2])].mean())
    layout = plot.layout(Camera(), 1280, 720)
    assert layout.triangles.size and layout.text
    with_nan = matrix.copy()
    with_nan[0, 0] = np.nan
    assert np.isfinite(Heatmap(with_nan, max_cells=8)._binned()[0, 0])
    for bad in (np.zeros(5), np.full((3, 3), np.nan), np.full((3, 3), np.inf)):
        with pytest.raises(ValueError):
            Heatmap(bad)
    with pytest.raises(ValueError, match="protein="):
        Heatmap(matrix, rows=[0, 1])


def test_pae_formats_and_residue_alignment(protein, tmp_path):
    n = len(protein.topology.residues)
    matrix = np.add.outer(np.arange(n), np.arange(n)) % 30 + 0.5
    afdb = tmp_path / "afdb.json"
    afdb.write_text(
        json.dumps([{"predicted_aligned_error": matrix.tolist(), "max_predicted_aligned_error": 31.75}])
    )
    colabfold = tmp_path / "scores.json"
    colabfold.write_text(json.dumps({"pae": matrix.tolist(), "max_pae": 30.0, "plddt": list(range(n))}))
    rows, cols = np.indices(matrix.shape)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            [
                {
                    "residue1": (rows.ravel() + 1).tolist(),
                    "residue2": (cols.ravel() + 1).tolist(),
                    "distance": matrix.ravel().tolist(),
                }
            ]
        )
    )
    np.save(tmp_path / "pae.npy", matrix)
    for source in (afdb, colabfold, legacy, tmp_path / "pae.npy", matrix):
        plot = Heatmap.pae(protein, source, highlight=protein.select(residues=(10, 20)))
        np.testing.assert_allclose(plot.matrix, matrix)
        assert plot.layout(Camera(), 1280, 720).leaders
    assert Heatmap.pae(protein, afdb).color_scale.vmax == 31.75
    # AlphaFold 3: tokens for one residue are averaged.
    tokens = [("A", r.resid) for r in protein.topology.residues] + [("A", 76)]
    big = np.ones((n + 1, n + 1))
    big[-1, :] = big[:, -1] = 3
    af3 = {
        "pae": big.tolist(),
        "token_chain_ids": [c for c, _ in tokens],
        "token_res_ids": [r for _, r in tokens],
    }
    plot = Heatmap.pae(protein, af3)
    assert plot.matrix.shape == (n, n) and plot.matrix[-1, -1] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="polymer residues"):
        Heatmap.pae(protein, np.ones((5, 5)))
    with pytest.raises(ValueError, match="square"):
        Heatmap.pae(protein, np.ones((n, n - 1)))


def test_live_distances_follow_coordinates_and_reference():
    p = Protein.build("A" * 10)
    plot = Heatmap.distances(p)
    extended = plot.matrix.copy()
    assert extended[0, -1] > 30
    change = Heatmap.distances(p, reference=p.copy())
    np.testing.assert_allclose(change.matrix, 0, atol=1e-4)
    assert change.color_scale.vmin == -change.color_scale.vmax
    scene = ProteinScene(width=64, height=36).add(p, plot, change)
    scene.play(SetTorsions(p, conformation="helix"), run_time=1)
    scene.seek(1)
    assert plot.matrix[0, -1] < 16
    assert change.matrix[0, -1] < -15
    centroid = Heatmap.distances(p, anchor="centroid", region=p.select(residues=(2, 6)))
    assert centroid.matrix.shape == (5, 5)
    with pytest.raises(ValueError, match="anchor"):
        Heatmap.distances(p, anchor="cb")


def test_hydropathy_values_scheme_and_shortcuts(structure_path):
    p = Protein.from_file(structure_path, include_water=True)
    values = ResidueValues.hydropathy(p)
    names = [r.name for r in p.topology.residues]
    assert values.values[names.index("ILE")] == 4.5 and values.values[names.index("ARG")] == -4.5
    assert np.isnan(values.values[names.index("HOH")])
    assert values.scale.vmin == -4.5 and values.scale.vmax == 4.5
    eisenberg = ResidueValues.hydropathy(p, "eisenberg")
    assert eisenberg.values[names.index("ILE")] == pytest.approx(1.38)
    np.testing.assert_allclose(
        residue_colors(p.cartoon(color="hydropathy")), values.scale.map(values.values), atol=1e-6
    )
    p.color_by("hydropathy")
    ile = next(i for i, a in enumerate(p.topology.atoms) if a.resname == "ILE")
    np.testing.assert_allclose(current_tints(p)[ile, :3], values.scale.map([4.5])[0], atol=1e-6)
    ColorByProperty(p, "hydropathy")
    legend = ColorLegend(values)
    assert legend.title.startswith("Hydropathy") and legend.layout(Camera(), 1280, 720).text
    with pytest.raises(ValueError, match="Unknown"):
        ResidueValues.hydropathy(p, "octanol")
    with pytest.raises(ValueError, match="Unknown residue property"):
        p.color_by("charge")


def test_conservation_scores_parsers_and_mapping(protein, tmp_path):
    scores = column_conservation(["AAC-", "AAD-", "AEFW", "AKGW"])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0] and scores[3] < scores[0]
    assert scores[2] < scores[1]
    # A3M: lowercase insertions are removed; FASTA keeps its columns.
    a3m = read_alignment(">q\nMKV\n>a\nMgKV\n>b\nM-V")
    assert [s for _, s in a3m] == ["MKV", "MKV", "M-V"]
    stockholm = read_alignment("# STOCKHOLM 1.0\nq MK.V\nb MKaV\nq EE\nb DE\n//\n")
    assert [s for _, s in stockholm] == ["MK-VEE", "MKAVDE"]
    clustal = read_alignment("CLUSTAL W\n\nq MKV\nb MRV\n   * *\n")
    assert [s for _, s in clustal] == ["MKV", "MRV"]
    # Ubiquitin's sequence with substitutions: invariant positions score 1.
    sequence = "".join(
        __import__("gemmi").find_tabulated_residue(r.name).one_letter_code.upper()
        for r in protein.topology.residues
    )
    variant = sequence[:10] + ("A" if sequence[10] != "A" else "G") + sequence[11:]
    path = tmp_path / "msa.fasta"
    path.write_text(f">query\n{sequence[5:]}\n>variant\n{variant[5:]}\n")
    values = ResidueValues.conservation(protein, path)
    assert np.isnan(values.values[:5]).all()  # residues 1–5 are not in the alignment
    assert values.values[5] == pytest.approx(1.0)
    assert values.values[10] < 1
    assert 0 <= values.scale.vmin < values.scale.vmax <= 1
    with pytest.raises(ValueError, match="does not match"):
        ResidueValues.conservation(protein, [("x", "WWWWWWWW"), ("y", "WWWWWWWW")])


def test_plddt_bands_and_confidence_files(protein, tmp_path):
    scale = ColorScale.plddt()
    colors = scale.map([10, 55, 75, 95, np.nan])
    np.testing.assert_allclose(colors[:4], scale.colors, atol=1e-6)
    np.testing.assert_allclose(colors[4], scale.missing)
    with pytest.raises(ValueError, match="one fewer"):
        ColorScale(0, 1, colors=("#000000", "#ffffff"), boundaries=(0.2, 0.5))
    n = len(protein.topology.residues)
    afdb = tmp_path / "confidence.json"
    afdb.write_text(json.dumps({"residueNumber": list(range(1, n + 1)), "confidenceScore": [80.0] * n}))
    values = ResidueValues.plddt(protein, afdb)
    assert np.all(values.values == 80) and values.name == "pLDDT"
    colab = tmp_path / "scores.json"
    colab.write_text(json.dumps({"plddt": list(np.linspace(30, 95, n))}))
    assert ResidueValues.plddt(protein, colab).values[-1] == pytest.approx(95)
    from_bfactors = ResidueValues.plddt(protein)
    assert from_bfactors.values[0] == pytest.approx(10.38, abs=0.01)
    legend = ColorLegend(values)
    assert legend.layout(Camera(), 1280, 720).text
