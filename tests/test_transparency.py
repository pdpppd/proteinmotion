"""Numerical fade checks catch stipple patterns, depth leaks and ordering artifacts."""

import numpy as np
import pytest

from proteinmotion import Protein
from proteinmotion.camera import Camera
from proteinmotion.renderer import Renderer
from proteinmotion.structure import Atom, Residue, Topology


def atom(element="C", position=(0, 0, 0)):
    topology = Topology(
        (Atom("A", 1, "", "ALA", "CA", element, 0),),
        (Residue("A", 1, "", "ALA", 0, -1),),
        np.empty((0, 2), np.uint32),
        (),
    )
    return Protein(topology, [position]).ball_and_stick(atom_scale=0.8)


@pytest.fixture(params=[1, 4])
def renderer(request):
    import wgpu

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter")
    with Renderer(192, 192, msaa=request.param) as renderer:
        yield renderer


@pytest.mark.gpu
def test_fade_is_continuous_alpha_without_stipple(renderer):
    p = atom()
    camera = Camera().frame(p, aspect=1)
    camera.theta = camera.phi = camera.depth_cue = 0
    background = np.array([0.03, 0.05, 0.08])
    empty = renderer.render([], camera, background).astype(float)
    solid = renderer.render([p], camera, background).astype(float)
    assert renderer.transparency_pipelines is None  # Opaque frames keep the one-pass path.
    core = np.max(solid[:, :, :3] - empty[:, :, :3], axis=2) > 70
    assert core.sum() > 500
    for opacity in [0.03, 0.25, 0.5, 0.75, 0.97]:
        p.set_opacity(opacity)
        actual = renderer.render([p], camera, background)
        expected = opacity * solid + (1 - opacity) * empty
        # A screen-door fade would differ by tens of levels at alternating core pixels.
        np.testing.assert_allclose(actual[:, :, :3][core], expected[:, :, :3][core], atol=2)
        assert (actual[:, :, 3] == 255).all()
    p.set_opacity(1)
    np.testing.assert_array_equal(renderer.render([p], camera, background), solid)


@pytest.mark.gpu
def test_transparent_atom_behind_opaque_atom_is_fully_occluded(renderer):
    front, back = atom(position=(0, 0, 2)), atom("O", (0, 0, -2)).set_opacity(0.6)
    camera = Camera().frame(front, back, aspect=1)
    camera.theta = camera.phi = camera.depth_cue = 0
    background = np.array([0.03, 0.05, 0.08])
    reference = renderer.render([front], camera, background)
    actual = renderer.render([back, front], camera, background)
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.gpu
def test_transparent_overlap_does_not_depend_on_draw_order(renderer):
    a = atom("N", (-0.2, 0, 0.6)).set_opacity(0.35)
    b = atom("O", (0.2, 0, -0.6)).set_opacity(0.6)
    camera = Camera().frame(a, b, aspect=1)
    background = np.array([0.03, 0.05, 0.08])
    forward = renderer.render([a, b], camera, background)
    backward = renderer.render([b, a], camera, background)
    np.testing.assert_allclose(forward, backward, atol=1)


@pytest.mark.gpu
def test_bond_fades_once_at_its_exterior_surface(renderer):
    atoms = (Atom("A", 1, "", "ALA", "CA", "C", 0), Atom("A", 1, "", "ALA", "C", "C", 0))
    topology = Topology(atoms, (Residue("A", 1, "", "ALA", 0, -1),), np.array([[0, 1]], np.uint32), ())
    p = Protein(topology, [[-4, 0, 0], [4, 0, 0]]).ball_and_stick(bond_radius=0.5)
    camera = Camera().frame(p, aspect=1)
    camera.theta = camera.phi = camera.depth_cue = 0
    background = np.array([0.03, 0.05, 0.08])
    empty = renderer.render([], camera, background).astype(float)
    solid = renderer.render([p], camera, background).astype(float)
    p.set_opacity(0.5)
    faded = renderer.render([p], camera, background).astype(float)
    # Restrict to the bond's central region, away from the atom/bond overlaps and edges.
    region = np.s_[93:99, 86:106, :3]
    assert np.min(solid[region] - empty[region]) > 20
    np.testing.assert_allclose(faded[region], 0.5 * (solid[region] + empty[region]), atol=2)
