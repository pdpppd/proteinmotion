"""Ligands, ions and side chains drawn as ball-and-stick detail over cartoons."""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from proteinmotion import (
    Colorize,
    HideAtoms,
    HideSideChains,
    Protein,
    ProteinScene,
    Representation,
    ShowAtoms,
    ShowSideChains,
)
from proteinmotion._eevee_geometry import MeshExporter, _sphere
from proteinmotion.geometry import (
    ELEMENT_COLORS,
    ION_SCALE,
    LIGAND_CARBON,
    atom_colors,
    atom_metadata,
    detail_colors,
    packed_metadata,
    residue_colors,
)
from proteinmotion.math3d import color
from proteinmotion.styling import atom_weights, current_details

DATA = Path(__file__).resolve().parents[1] / "examples/data"


@pytest.fixture
def calmodulin():
    return Protein.from_file(DATA / "1cll.cif").center()


def names(protein, region):
    atoms = protein.topology.atoms
    return {(atoms[i].resname, atoms[i].resid, atoms[i].name) for i in region.atom_indices}


def test_residue_categories_and_default_detail(calmodulin, protein):
    categories = calmodulin.topology.residue_categories
    assert categories.count("ion") == 4 and categories.count("ligand") == 1
    assert categories.count("polymer") == 144
    detail = current_details(calmodulin)
    untraced = calmodulin.topology.untraced_atoms
    np.testing.assert_array_equal(detail, untraced.astype(np.float32))
    assert {calmodulin.topology.atoms[i].resname for i in np.flatnonzero(untraced)} == {"CA", "EOH"}
    # A structure without hetero groups gains no detail atoms.
    assert not current_details(protein).any()
    assert set(protein.topology.residue_categories) == {"polymer"}
    # Only a cartoon or ribbon hides untraced atoms; ball-and-stick shows every atom.
    np.testing.assert_array_equal(atom_weights(calmodulin), detail)
    calmodulin.ball_and_stick()
    assert (atom_weights(calmodulin) == 1).all()


def test_water_is_categorized_when_loaded():
    wet = Protein.from_file(DATA / "1cll.cif", include_water=True)
    categories = wet.topology.residue_categories
    assert categories.count("water") > 0
    waters = wet.select(water=True)
    assert {wet.topology.atoms[i].resname for i in waters.atom_indices} == {"HOH"}


def test_ion_radii_ligand_colors_and_detail_palette(calmodulin):
    meta = atom_metadata(calmodulin)
    ions = calmodulin.select(ions=True).atom_indices
    np.testing.assert_allclose(meta[ions, 3], 2.31 * ION_SCALE, rtol=1e-3)
    np.testing.assert_allclose(meta[ions, :3], np.tile(color(ELEMENT_COLORS["Ca"]), (4, 1)), atol=1e-6)
    ligand = calmodulin.select(ligands=True).atom_indices
    np.testing.assert_allclose(meta[ligand, :3], np.tile(color(LIGAND_CARBON), (len(ligand), 1)), atol=1e-6)
    # Polymer atoms keep the structure palette in the ball-and-stick representation.
    polymer = np.flatnonzero(~calmodulin.topology.untraced_atoms)
    palette = residue_colors(calmodulin)[[calmodulin.topology.atoms[i].residue_index for i in polymer]]
    np.testing.assert_allclose(meta[polymer, :3], palette)
    # Over a cartoon, heteroatoms use element colors and carbons the structure palette.
    detail = detail_colors(calmodulin)
    for i, atom in enumerate(calmodulin.topology.atoms[:200]):
        expected = meta[i, :3] if atom.element == "C" else color(ELEMENT_COLORS[atom.element])
        np.testing.assert_allclose(detail[i], expected, atol=1e-6)
    np.testing.assert_allclose(atom_colors(calmodulin), detail)
    calmodulin.representation = np.array([0.5, 0, 0.5])
    np.testing.assert_allclose(atom_colors(calmodulin), (detail + meta[:, :3]) / 2, atol=1e-6)
    calmodulin.ball_and_stick()
    np.testing.assert_allclose(detail_colors(calmodulin), atom_metadata(calmodulin)[:, :3])


def test_packed_metadata_preserves_colors_and_radii(calmodulin):
    packed = packed_metadata(calmodulin)
    for column, colors in ((0, atom_metadata(calmodulin)[:, :3]), (1, detail_colors(calmodulin))):
        bits = packed[:, column].view(np.uint32)
        unpacked = np.stack([(bits >> shift) & 255 for shift in (0, 8, 16)], 1) / 255
        np.testing.assert_allclose(unpacked, colors, atol=0.5 / 255 + 1e-6)
        assert ((bits >> 24) == 255).all()
    np.testing.assert_array_equal(packed[:, 3], atom_metadata(calmodulin)[:, 3])


def test_selectors_combine_categories_names_and_distance(calmodulin):
    p = calmodulin
    assert len(p.select(ions=True).atom_indices) == 4
    assert names(p, p.select(ligands=True)) == {("EOH", 153, "C1"), ("EOH", 153, "C2")}
    both = p.select(ligands=True, ions=True)
    np.testing.assert_array_equal(
        both.atom_indices, np.union1d(p.select(ligands=True).atom_indices, p.select(ions=True).atom_indices)
    )
    np.testing.assert_array_equal(
        p.select(resname="asp").atom_indices, p.select(resname=["ASP"]).atom_indices
    )
    assert {a for a, _, _ in names(p, p.select(resname=("ASP", "GLU")))} == {"ASP", "GLU"}
    assert len(p.select(ions=True, residues=(149, 150)).atom_indices) == 2
    ion = p.select(ions=True, residues=149)
    site = p.select(within=3.0, of=ion)
    # Whole residues with any atom within 3 Å of the ion, excluding the ion itself.
    distances = cdist(p.positions, ion.positions).min(1)
    owners = np.array([a.residue_index for a in p.topology.atoms])
    expected = set(owners[distances <= 3.0]) - set(ion.residue_indices)
    assert set(site.residue_indices) == expected
    assert {p.topology.residues[i].resid for i in site.residue_indices} == {20, 22, 24, 26, 31}
    assert all(owners[i] in expected for i in site.atom_indices)
    assert len(site.atom_indices) == sum(np.isin(owners, list(expected)))
    # Distance is measured in world space, so a separate copy can be the reference.
    other = p.copy()
    assert set(p.select(within=3.0, of=other.select(ions=True, residues=149)).residue_indices) == (
        expected | set(ion.residue_indices)
    )
    other.shift([100, 0, 0])
    with pytest.raises(ValueError, match="no atoms"):
        p.select(within=3.0, of=other.select(ions=True, residues=149))
    with pytest.raises(ValueError, match="together"):
        p.select(within=3.0)
    with pytest.raises(ValueError, match="positive"):
        p.select(within=0, of=ion)
    with pytest.raises(TypeError, match="Region or Protein"):
        p.select(within=3.0, of=[0, 0, 0])
    with pytest.raises(ValueError, match="no atoms"):
        p.select(water=True)


def test_side_chains_join_the_cartoon_at_ca(calmodulin, protein):
    p = calmodulin
    chains = p.select(residues=(19, 24)).side_chains()
    atoms = names(p, chains)
    assert ("ASP", 20, "CA") in atoms and ("ASP", 20, "OD1") in atoms
    assert not any(name in ("N", "C", "O") for _, _, name in atoms)
    assert {resid for _, resid, _ in atoms} == {19, 20, 21, 22, 24}  # Gly23 has no side chain.
    proline = protein.select(residues=19).side_chains()  # Ubiquitin Pro19 keeps N for its ring.
    assert {protein.topology.atoms[i].name for i in proline.atom_indices} == {"N", "CA", "CB", "CG", "CD"}
    # Ions, ligands and whole proteins are accepted; only amino acid side chains are returned.
    everything = ShowSideChains(p)
    assert all(
        p.topology.residue_categories[p.topology.atoms[i].residue_index] == "polymer" for i in everything.ids
    )
    with pytest.raises(ValueError, match="side chains"):
        p.select(ions=True).side_chains()


def test_show_side_chains_stagger_seek_and_channels(calmodulin):
    p = calmodulin
    site = p.select(within=3.0, of=p.select(ions=True, residues=149))
    chains = site.side_chains()
    scene = ProteinScene()
    scene.add(p)
    scene.wait(1)
    scene.play(
        ShowSideChains(site, residue_delay=0.25, easing="linear"), Colorize(chains, "#ff0000"), run_time=2
    )
    scene.play(HideSideChains(site, reverse=True), run_time=1)
    first = [i for i in chains.atom_indices if p.topology.atoms[i].resid == 20]
    last = [i for i in chains.atom_indices if p.topology.atoms[i].resid == 31]
    scene.seek(0.5)
    np.testing.assert_array_equal(current_details(p), p.topology.untraced_atoms)
    scene.seek(1.5)
    # Five residues share 2 s: each takes 1 s, the last starting 1 s after the first.
    np.testing.assert_allclose(current_details(p)[first], 0.5)
    np.testing.assert_allclose(current_details(p)[last], 0)
    scene.seek(3)
    np.testing.assert_allclose(current_details(p)[chains.atom_indices], 1)
    assert (current_details(p)[p.select(ions=True).atom_indices] == 1).all()
    backbone = p.select(residues=20, atoms=("N", "C", "O")).atom_indices
    assert not current_details(p)[backbone].any()
    expected = current_details(p).copy()
    scene.seek(4)
    np.testing.assert_allclose(current_details(p)[chains.atom_indices], 0)
    scene.seek(1.2)
    scene.seek(3)
    np.testing.assert_array_equal(current_details(p), expected)
    with pytest.raises(ValueError, match="detail"):
        ProteinScene().play(ShowAtoms(chains), HideSideChains(site))


def test_static_detail_snapshot_and_representation_blend(calmodulin):
    p = calmodulin
    p.hide_atoms()
    assert not current_details(p).any()
    p.select(residues=(20, 24)).show_atoms()
    shown = p.select(residues=(20, 24)).atom_indices
    assert (current_details(p)[shown] == 1).all() and current_details(p).sum() == len(shown)
    copy = p.copy()
    np.testing.assert_array_equal(current_details(copy), current_details(p))
    p.select(ions=True).show_atoms()
    scene = ProteinScene()
    scene.add(p)
    scene.play(Representation(p, "ball_and_stick"), run_time=1)
    scene.seek(0.5)
    weights = atom_weights(p)
    ball = p.representation[2]
    assert 0 < ball < 1
    np.testing.assert_allclose(weights[shown], 1)
    hidden = np.setdiff1d(np.arange(len(weights)), np.r_[shown, p.select(ions=True).atom_indices])
    np.testing.assert_allclose(weights[hidden], ball)
    scene.play(HideAtoms(p), run_time=1)
    scene.seek(2)
    assert not current_details(p).any()
    np.testing.assert_allclose(atom_weights(p), 1)
    p.show_atoms()
    assert (current_details(p) == 1).all()


def test_eevee_exports_only_visible_detail_atoms(calmodulin):
    p = calmodulin
    verts, faces = _sphere()
    exporter = MeshExporter()
    cartoon, spheres, sticks = exporter.meshes(p)
    # Four ions and two ethanol carbons; only the C1–C2 bond has both atoms shown.
    assert len(spheres["vertices"]) == 6 * len(verts)
    assert len(spheres["faces"]) == len(spheres["opacity"]) == 6 * len(faces)
    assert np.all(spheres["opacity"] == 1)
    assert len(sticks["vertices"]) == 32
    chains = p.select(residues=20).side_chains()
    chains.show_atoms()
    p.select(residues=20).set_opacity(0.4)
    _, spheres, sticks = exporter.meshes(p)
    count = 6 + len(chains.atom_indices)
    assert len(spheres["vertices"]) == count * len(verts)
    np.testing.assert_allclose(np.unique(spheres["opacity"]), [0.4, 1], rtol=1e-6)
    # Asp20 carboxylate oxygens keep the element color over the cartoon.
    red = np.isclose(spheres["colors"], color(ELEMENT_COLORS["O"]), atol=1e-6).all(1)
    assert red.sum() == 2 * len(verts)
    p.hide_atoms()
    assert len(exporter.meshes(p)) == 1


@pytest.mark.gpu
def test_gpu_detail_atoms_render_and_match_static_state(calmodulin):
    import wgpu

    from proteinmotion.renderer import Renderer

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter available")
    p = calmodulin
    site = p.select(within=3.0, of=p.select(ions=True, residues=149))
    scene = ProteinScene(width=384, height=256)
    scene.add(p)
    scene.camera.frame(site | p.select(ions=True, residues=149), aspect=1.5)
    scene.play(ShowSideChains(site, residue_delay=0.1), run_time=1)
    with Renderer(384, 256, msaa=1) as renderer:
        start, middle, end = (scene.render_frame(t, renderer=renderer) for t in (0, 0.5, 1))
        assert np.abs(end.astype(int) - start.astype(int)).sum() > 20000
        assert np.abs(middle.astype(int) - start.astype(int)).sum() > 1000
        np.testing.assert_array_equal(scene.render_frame(0.5, renderer=renderer), middle)
        # The finished animation and the equivalent static detail draw identically.
        still = Protein.from_file(DATA / "1cll.cif").center()
        still.select(within=3.0, of=still.select(ions=True, residues=149)).side_chains().show_atoms()
        np.testing.assert_array_equal(renderer.render([still], scene.camera, scene.background), end)
        # Hiding the default ions and ligand changes the plain cartoon.
        still = Protein.from_file(DATA / "1cll.cif").center()
        with_ions = renderer.render([still], scene.camera, scene.background)
        np.testing.assert_array_equal(with_ions, start)
        still.hide_atoms()
        without = renderer.render([still], scene.camera, scene.background)
        assert np.abs(with_ions.astype(int) - without.astype(int)).sum() > 2000

        # The ion is drawn in its element color: the frame gains green-dominant pixels.
        def green(image):
            rgb = image[:, :, :3].astype(int)
            return ((rgb[:, :, 1] > rgb[:, :, 0] + 25) & (rgb[:, :, 1] > rgb[:, :, 2] + 10)).sum()

        assert green(with_ions) > green(without) + 50


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("PROTEINMOTION_TEST_EEVEE"), reason="Set PROTEINMOTION_TEST_EEVEE=1")
def test_eevee_renders_detail_atoms(calmodulin):
    from proteinmotion.camera import Camera
    from proteinmotion.eevee import EEVEE, EEVEEOptions

    p = calmodulin
    site = p.select(within=3.0, of=p.select(ions=True, residues=149))
    camera = Camera().frame(site, aspect=16 / 9)
    with EEVEE(160, 90, options=EEVEEOptions(samples=16, supersampling=1)) as renderer:
        p.hide_atoms()
        plain = renderer.render([p], camera, [0.03, 0.04, 0.07])
        p.select(ions=True).show_atoms()
        site.side_chains().show_atoms()
        detailed = renderer.render([p], camera, [0.03, 0.04, 0.07])
        assert np.abs(detailed.astype(int) - plain.astype(int)).sum() > 5000
