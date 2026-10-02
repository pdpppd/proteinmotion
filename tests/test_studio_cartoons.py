"""Classic Studio cartoons: silhouettes, chain boundaries and animated GPU state."""

import numpy as np
import pytest
from scipy.ndimage import label

from proteinmotion import (
    Colorize,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Representation,
    SetOpacity,
    StudioLook,
    Trajectory,
)
from proteinmotion.camera import Camera
from proteinmotion.geometry import segments, studio_cartoon_segments
from proteinmotion.structure import Atom, Residue, Topology
from proteinmotion.studio import StudioRenderer


def strand(secondary="CCEEEECC"):
    atoms, residues, xyz = [], [], []
    for i, ss in enumerate(secondary):
        ca = len(atoms)
        for name, element, y in [("CA", "C", 0), ("O", "O", 1)]:
            atoms.append(Atom("A", i + 1, "", "ALA", name, element, i))
            xyz.append([3.8 * i, y, 0])
        residues.append(Residue("A", i + 1, "", "ALA", ca, ca + 1, ss))
    return Protein(
        Topology(tuple(atoms), tuple(residues), np.empty((0, 2), np.uint32), (np.arange(len(residues)),)),
        xyz,
    ).cartoon(color="#ffffff")


@pytest.mark.parametrize("secondary", ["CCEEECC", "CCEEE", "EEEEE", "CCEECCEECC"])
def test_arrows_end_at_each_strand_including_chain_termini(secondary):
    p = strand(secondary)
    original = segments(p)
    rows = studio_cartoon_segments(p)
    runs = sum(ss == "E" and (i == 0 or secondary[i - 1] != "E") for i, ss in enumerate(secondary))
    assert rows[:, 18].sum() == runs
    # Geometry customization must preserve atom indices, colors, caps and source metadata.
    np.testing.assert_array_equal(rows[:, :4], original[:, :4])
    np.testing.assert_array_equal(rows[:, 8:11], original[:, 8:11])
    np.testing.assert_array_equal(rows[:, 12:15], original[:, 12:15])
    np.testing.assert_array_equal(rows[:, 16:18], original[:, 16:18])
    np.testing.assert_array_equal(segments(p), original)


@pytest.mark.gpu
def test_flat_sheet_arrow_silhouette_and_round_loop_connection():
    p = strand()
    camera = Camera().frame(p, aspect=2)
    camera.theta = camera.phi = camera.depth_cue = 0
    look = StudioLook(effects="clean", ambient_occlusion=0)
    with StudioRenderer(768, 384, look=look) as renderer:
        image = renderer.render([p], camera, (0, 0, 0))
    mask = np.any(image[:, :, :3] > 20, axis=2)
    assert label(mask)[1] == 1  # Tube, arrow shoulder and tip join without cracks.
    heights = mask.sum(0)

    def x_at(residue):
        point = np.array([3.8 * residue, 0, 0, 1]) @ camera.matrix(2).T
        return round((point[0] / point[3] + 1) * 384)

    loop = heights[x_at(0.8)]
    body = heights[x_at(3.0)]
    head = heights[x_at(5.2)]
    tip = heights[x_at(5.95)]
    assert loop > 0 and body > 4 * loop
    assert head > 1.4 * body and tip < 0.4 * body


@pytest.mark.gpu
@pytest.mark.parametrize("msaa", [1, 4])
def test_classic_cartoon_animation_deformation_and_seek(protein, msaa):
    scene = ProteinScene(width=256, height=256, msaa=msaa)
    scene.add(protein)
    scene.camera.frame(protein, aspect=1)
    xyz = protein.positions.copy()
    frames = np.array([xyz, xyz * [1.1, 0.9, 1]])
    scene.play(
        PlayTrajectory(protein, Trajectory(frames)),
        Colorize(protein.select(residues=(1, 76)), "#eeaa66"),
        run_time=1,
    )
    scene.play(SetOpacity(protein.select(residues=(1, 76)), 0.65), run_time=1)
    scene.play(Representation(protein, "ribbon"), run_time=1)
    with StudioRenderer(256, 256, msaa=msaa, look=StudioLook(effects="clean")) as renderer:
        times = (0, 0.5, 1, 2, 2.5, 3)
        images = [scene.render_frame(t, renderer=renderer) for t in times]
        assert renderer._molecules[protein].surface is None
        for a, b in zip(images, images[1:]):
            assert np.abs(a.astype(int) - b).sum() > 1000
        for t, expected in reversed(list(zip(times, images))):
            np.testing.assert_array_equal(scene.render_frame(t, renderer=renderer), expected)
    # Smoothing affects only the display spline, never molecular/analysis coordinates.
    scene.seek(0)
    np.testing.assert_array_equal(protein.positions, xyz)


@pytest.mark.gpu
def test_nucleotide_geometry_and_uniform_ribbon_are_preserved(structure_path):
    from proteinmotion import NucleicAcid

    dna = NucleicAcid.from_file(structure_path.parent / "1bna.cif").center()
    a, b = segments(dna), studio_cartoon_segments(dna)
    np.testing.assert_array_equal(a[:, 4:8], b[:, 4:8])
    assert (b[:, 11] == -1).all() and not b[:, 18].any()
    p = strand().ribbon(color="#ffffff")
    for molecule in (dna, p):
        camera = Camera().frame(molecule, aspect=1)
        images = []
        for style in ("classic", "legacy"):
            with StudioRenderer(192, 192, look=StudioLook(cartoon_style=style, effects="clean")) as r:
                images.append(r.render([molecule], camera, (0, 0, 0)))
        # The profile is unchanged; denser Studio tessellation slightly changes edge pixels.
        assert np.mean(np.abs(images[0].astype(int) - images[1])) < 2


@pytest.mark.gpu
def test_per_atom_fades_converge_to_the_solid_image():
    from proteinmotion import SetOpacity

    p = strand("CCEEEECCEEEECC")
    images = {}
    for opacity in (1.0, 0.999, 0.9):
        q = p.copy()
        scene = ProteinScene(width=256, height=128, renderer="studio", look=StudioLook(effects="clean"))
        scene.add(q)
        scene.camera.frame(q, aspect=2)
        scene.play(SetOpacity(q.select(residues=(1, 7)), opacity), run_time=1)
        images[opacity] = scene.render_frame(1).astype(int)
    # Nearly opaque atoms look solid, so a fade ends without a jump.
    assert np.abs(images[0.999] - images[1.0]).mean() < 0.05
    assert np.abs(images[0.9] - images[1.0]).mean() > np.abs(images[0.999] - images[1.0]).mean()
