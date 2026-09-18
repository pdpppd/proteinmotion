"""DNA/RNA topology, shared appearance, animated geometry and native rendering."""

import warnings
from dataclasses import replace
from pathlib import Path

import gemmi
import numpy as np
import pytest

from proteinmotion import (
    BaseStyle,
    Colorize,
    ContactMap,
    NucleicAcid,
    PlayTrajectory,
    Protein,
    Representation,
    ResidueValues,
    Scene,
    SequenceTrack,
    SetOpacity,
    Trajectory,
)
from proteinmotion._eevee_geometry import MeshExporter
from proteinmotion.camera import Camera
from proteinmotion.geometry import BASE_COLORS, atom_metadata, residue_colors, segments, state_data
from proteinmotion.math3d import color, rotation
from proteinmotion.nucleic import BASE_STYLES, BaseGeometry
from proteinmotion.structure import make_chains

DATA = Path(__file__).resolve().parents[1] / "examples/data"


@pytest.fixture
def dna():
    return NucleicAcid.from_file(DATA / "1bna.cif").center()


def test_loader_anchors_modifications_and_no_protein_warning(dna):
    assert [len(c) for c in dna.topology.chains] == [12, 12]
    assert all(r.ca == -1 and r.kind == "dna" for r in dna.topology.residues)
    assert all(dna.topology.atoms[r.trace_atom].name == "C4'" for r in dna.topology.residues)
    with warnings.catch_warnings(record=True) as caught:
        rna = NucleicAcid.from_file(DATA / "1ehz.cif")
    assert not caught
    assert [len(c) for c in rna.topology.chains] == [76]
    residues = {r.resid: r for r in rna.topology.residues if r.is_nucleic}
    assert residues[40].name == "5MC" and residues[40].base == "C"
    assert residues[37].name == "YYG" and residues[37].base == "G"
    assert residues[39].name == "PSU" and residues[39].base == "U"
    assert all(rna.topology.residues[i].is_nucleic for i in rna.topology.chains[0])
    assert len(segments(rna)) == 75
    assert np.isfinite(state_data(rna.topology, rna.positions)).all()
    with pytest.raises(ValueError, match="DNA/RNA"):
        NucleicAcid.from_file(DATA / "1ubq.cif")


def test_chain_breaks_and_legacy_primes(dna, tmp_path):
    st = gemmi.read_structure(str(DATA / "1bna.cif"))
    for chain in st[0]:
        for residue in chain:
            for atom in residue:
                atom.name = atom.name.replace("'", "*")
    path = tmp_path / "legacy.pdb"
    st.write_pdb(str(path))
    legacy = NucleicAcid.from_file(path)
    assert [len(c) for c in legacy.topology.chains] == [12, 12]
    assert len(legacy.select(atoms="C4'").atom_indices) == 24
    assert len(legacy.select(atoms="C4*").atom_indices) == 24
    assert all(legacy.topology.atoms[r.morph_atom].name == "C1*" for r in legacy.topology.residues)
    assert len(legacy.select(atoms="C1'").atom_indices) == 24
    residues = list(dna.topology.residues)
    residues[5] = replace(residues[5], backbone=-1)
    assert [len(c) for c in make_chains(residues, dna.positions, dna.topology.atoms)] == [5, 6, 12]
    xyz = dna.positions.copy()
    p = next(i for i, a in enumerate(dna.topology.atoms) if a.residue_index == 5 and a.name == "P")
    xyz[p] += 10
    assert [len(c) for c in make_chains(dna.topology.residues, xyz, dna.topology.atoms)] == [5, 7, 12]


def test_mixed_complex_and_multiple_models(tmp_path):
    st = gemmi.read_structure(str(DATA / "1bna.cif"))
    chain = gemmi.read_structure(str(DATA / "1ubq.cif"))[0][0].clone()
    chain.name = "P"
    st[0].add_chain(chain)
    second = st[0].clone()
    second.num = 2
    for c in second:
        for r in c:
            for a in r:
                a.pos.x += 2
    st.add_model(second)
    path = tmp_path / "mixed.cif"
    st.make_mmcif_document().write_file(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = Protein.from_file(path)
    assert len(p.trajectory) == 2
    assert sum(r.is_nucleic for r in p.topology.residues) == 24
    assert sum(r.ca >= 0 for r in p.topology.residues) == 76
    assert [len(c) for c in p.topology.chains] == [12, 12, 76]
    np.testing.assert_allclose(
        p.trajectory.frame(1) - p.positions, np.tile([2, 0, 0], (len(p.positions), 1)), atol=1e-5
    )
    assert len(segments(p)) == 97


@pytest.mark.parametrize("style", BASE_STYLES)
def test_base_mesh_closed_outward_and_follows_rigid_motion(dna, style):
    dna.set_bases(style)
    g = BaseGeometry(dna)
    part = BASE_STYLES.index(style)
    vertices, faces, colors, alpha, normals = g.evaluate(dna, part)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nonzero = np.linalg.norm(cross, axis=1) > 1e-5
    assert np.all((cross * normals[faces].mean(1)).sum(1)[nonzero] > 0)
    assert np.isfinite(vertices).all() and alpha.min() == 1
    transform = rotation(0.7, (0.3, 1, 0.6))
    dna.set_positions(dna.positions @ transform.T + [4, -2, 7])
    moved = g.evaluate(dna, part)
    # Cylinder meshes may rotate around their axes, but the shape stays in place.
    mode = g.parts[part][0][:, 13].copy().view(np.uint32)
    if np.any(mode == 0):
        np.testing.assert_allclose(
            moved[0][mode == 0], (vertices @ transform.T + [4, -2, 7])[mode == 0], atol=1e-5
        )
    assert g.matches(dna)  # No rebuilding during coordinate animation.
    np.testing.assert_array_equal(moved[2], colors)
    assert np.isfinite(moved[4]).all()


def test_incomplete_and_coarse_models(dna, tmp_path):
    st = gemmi.read_structure(str(DATA / "1bna.cif"))
    del st[0][0][0][next(i for i, a in enumerate(st[0][0][0]) if a.name == "C2")]
    path = tmp_path / "incomplete.cif"
    st.make_mmcif_document().write_file(str(path))
    p = NucleicAcid.from_file(path)
    with pytest.warns(UserWarning, match="Incomplete base rings use sticks"):
        g = BaseGeometry(p)
    assert np.isfinite(g.evaluate(p, 0)[0]).all()
    # A P-only model has a trace and no invented base coordinates.
    for chain in st[0]:
        for residue in chain:
            for i in range(len(residue) - 1, -1, -1):
                if residue[i].name != "P":
                    del residue[i]
    st.make_mmcif_document().write_file(str(path))
    p = NucleicAcid.from_file(path)
    assert len(segments(p)) == 20
    assert all(not len(f) for _, f, _ in BaseGeometry(p).parts)


def test_labels_properties_plots_and_palette(dna):
    labels = dna.label_residues(chain="A", residues=(1, 4), format="one_letter")
    assert len(labels.labels) == 4
    assert labels.labels[0].title.text == "A · C 1"
    assert len(SequenceTrack(dna).ids) == 24
    assert ContactMap(dna).matrix.shape == (24, 24)
    assert np.isfinite(ResidueValues.b_factors(dna).values).all()
    np.testing.assert_allclose(ResidueValues.rmsf(dna).values, 0, atol=1e-5)
    for i, r in enumerate(dna.topology.residues):
        np.testing.assert_allclose(residue_colors(dna)[i], color(BASE_COLORS[r.base]), atol=1e-7)
    dna.ball_and_stick(color="base")
    for i, a in enumerate(dna.topology.atoms):
        np.testing.assert_allclose(atom_metadata(dna)[i, :3], residue_colors(dna)[a.residue_index])
    assert isinstance(dna.copy(), NucleicAcid)
    ruler = dna.select(chain="A", residues=1).distance_to(dna.select(chain="A", residues=4))
    anchors = [dna.topology.residues[i].trace_atom for i in (0, 3)]
    assert ruler.distance == pytest.approx(
        np.linalg.norm(dna.positions[anchors[0]] - dna.positions[anchors[1]])
    )
    dna.select(chain="A", residues=2).set_color("#ff0000")
    camera = Camera().set_focus(dna, chain="A", residues=(2, 4))
    before = camera.focus_point
    dna.shift([3, 0, 0])
    np.testing.assert_allclose(camera.focus_point, before + [3, 0, 0])


def test_nucleotide_hbond_sites_need_explicit_hydrogens(dna):
    hb = dna.hydrogen_bonds()
    assert len(hb.donors) == 32
    assert all(dna.topology.atoms[i].element == "N" for i in hb.donors)
    assert all(dna.topology.atoms[i].name != "N9" for i in hb.acceptors)
    assert hb.pairs == ()  # 1BNA has no deposited H; do not invent base-pair bonds.


def test_nucleic_md_roundtrip_default_selection(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    path = tmp_path / "dna.pdb"
    gemmi.read_structure(str(DATA / "1bna.cif")).write_pdb(str(path))
    universe = mda.Universe(str(path))
    movie = tmp_path / "dna.xtc"
    with mda.Writer(str(movie), n_atoms=len(universe.atoms)) as writer:
        for shift in (0, 2):
            universe.atoms.positions += [shift, 0, 0]
            writer.write(universe.atoms)
    t = Trajectory.from_mdanalysis(path, movie)
    p = NucleicAcid.from_trajectory(t)
    assert len(t) == 2 and len(p.topology.atoms) == 486
    assert [len(c) for c in p.topology.chains] == [12, 12]
    np.testing.assert_allclose(t.frame(1) - t.frame(0), np.tile([2, 0, 0], (486, 1)), atol=0.012)


def test_style_timeline_stagger_opacity_and_random_seek(dna):
    scene = Scene()
    scene.add(dna)
    chain = dna.select(chain="A")
    scene.play(
        BaseStyle(dna, "rings"),
        Colorize(chain, "#ff0000", residue_delay=0.03),
        SetOpacity(dna.select(chain="B"), 0.2),
        run_time=2,
    )
    scene.play(Representation(dna, "surface"), run_time=1)
    scene.play(Representation(dna, "cartoon", bases="ladder"), run_time=1)
    scene.seek(1)
    first = dna.snapshot()
    assert dna.base_style[0] == dna.base_style[1] == 0.5
    assert dna.atom_opacities[dna.select(chain="B").atom_indices[0]] == pytest.approx(0.6)
    scene.seek(4)
    np.testing.assert_array_equal(dna.base_style, [0, 0, 0, 1])
    scene.seek(1)
    np.testing.assert_array_equal(dna.base_style, first["base_style"])
    scene.seek(0)
    np.testing.assert_array_equal(dna.base_style, [1, 0, 0, 0])
    for args in ({"base_thickness": float("nan")}, {"backbone_radius": 0}, {"base_radius": -1}):
        with pytest.raises(ValueError):
            dna.set_bases("rings", **args)
    with pytest.raises(ValueError, match="Base style"):
        BaseStyle(dna, "bad")
    with pytest.raises(ValueError, match="base_style"):
        Scene().play(BaseStyle(dna, "rings"), Representation(dna, "cartoon", bases="ladder"))


@pytest.mark.parametrize("representation", [*BASE_STYLES, "ball_and_stick", "surface"])
def test_eevee_export_color_opacity_and_surface(dna, representation):
    if representation in BASE_STYLES:
        dna.cartoon(bases=representation)
    else:
        getattr(dna, representation)()
    dna.set_color("#ff0000").set_residue_opacity(0.35, chain="B")
    pieces = MeshExporter().meshes(dna)
    assert pieces and all(len(p["faces"]) for p in pieces)
    for p in pieces:
        assert np.isfinite(p["vertices"]).all()
        np.testing.assert_allclose(p["colors"], np.tile([1, 0, 0], (len(p["colors"]), 1)))
        assert set(np.round(p["opacity"], 2)) <= {np.float32(0.35), np.float32(1)}


@pytest.mark.gpu
def test_native_styles_surface_fades_and_trajectory_seek(dna):
    from proteinmotion.renderer import Renderer

    camera = Camera().frame(dna, aspect=1.5)
    with Renderer(360, 240) as renderer:
        images = []
        for style in (*BASE_STYLES, "none"):
            dna.cartoon(bases=style)
            images.append(renderer.render([dna], camera, [0.02, 0.03, 0.05]))
        for image in images:
            assert image[:, :, :3].max() > 100
            assert (image[:, :, 3] == 255).all()
        for a, b in zip(images, images[1:]):
            assert np.abs(a.astype(int) - b).sum() > 5000
        dna.cartoon().set_color("#ff0000")
        dna.set_residue_opacity(0.15, chain="B")
        scene = Scene(width=360, height=240)
        scene.background = np.zeros(3)
        scene.camera = camera
        scene.add(dna)
        xyz = dna.positions.copy()
        frames = [xyz, xyz @ rotation(0.2, (0, 1, 0)).T + [1, 0, 0]]
        scene.play(
            PlayTrajectory(dna, Trajectory(frames, topology=dna.topology)),
            BaseStyle(dna, "rings"),
            run_time=2,
        )
        first = scene.render_frame(0.7, renderer=renderer)
        buffers = renderer._molecules[dna].bases.buffers.copy()
        scene.render_frame(1.6, renderer=renderer)
        np.testing.assert_array_equal(scene.render_frame(0.7, renderer=renderer), first)
        assert renderer._molecules[dna].bases.buffers == buffers
        assert first[:, :, 0].sum() > first[:, :, 1].sum()
        dna.surface(resolution=1.0)
        surface = renderer.render([dna], camera, [0.02, 0.03, 0.05])
        assert surface[:, :, 0].sum() > first[:, :, 0].sum()
