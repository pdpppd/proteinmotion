"""Map coordinate conventions and deterministic native/EEVEE density geometry."""

import gzip
import itertools
import os
import struct

import numpy as np
import pytest

from proteinmotion import DensityMap, EEVEEOptions, ProteinScene
from proteinmotion._eevee_geometry import MeshExporter


def write_map(path, *, axes=(0, 1, 2), endian="<", header_origin=(0, 0, 0), starts=(2, -3, 4), spacegroup=0):
    values = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(np.float32)
    stored = values.transpose(axes)
    header = bytearray(1024)

    def ints(word, *values):
        struct.pack_into(endian + "i" * len(values), header, (word - 1) * 4, *values)

    def floats(word, *values):
        struct.pack_into(endian + "f" * len(values), header, (word - 1) * 4, *values)

    ints(1, *stored.shape, 2)
    ints(5, *[starts[a] for a in axes])
    ints(8, 4, 5, 6)
    floats(11, 8, 15, 24, 90, 90, 90)
    ints(17, *[a + 1 for a in axes])
    floats(20, float(values.min()), float(values.max()), float(values.mean()))
    ints(23, spacegroup, 0)
    floats(50, *header_origin)
    header[208:212] = b"MAP "
    header[212:216] = b"DD\x00\x00" if endian == "<" else b"\x11\x11\x00\x00"
    payload = header + stored.astype(endian + "f4").tobytes(order="F")
    path.write_bytes(gzip.compress(payload) if path.suffix == ".gz" else payload)
    return values


@pytest.mark.parametrize("axes", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("endian", ["<", ">"])
def test_mrc_axes_origins_and_endianness(tmp_path, axes, endian):
    path = tmp_path / "test.mrc"
    expected = write_map(path, axes=axes, endian=endian)
    density = DensityMap.from_file(path)
    np.testing.assert_array_equal(density.values, expected)
    np.testing.assert_allclose(density.origin, [4, -9, 16])
    np.testing.assert_allclose(density.basis, np.diag([2, 3, 4]))
    assert density.sample([6, -6, 20]) == pytest.approx(expected[1, 1, 1])
    assert np.isnan(density.sample([1000, 0, 0]))


def test_mrc_header_origin_gzip_periodic_and_limits(tmp_path):
    path = tmp_path / "test.ccp4.gz"
    values = write_map(path, header_origin=(10, 20, 30), spacegroup=1)
    density = DensityMap.from_file(path)
    np.testing.assert_array_equal(density.origin, [10, 20, 30])
    assert density.periodic
    assert density.sample([18, 20, 30]) == values[0, 0, 0]  # one cell along x
    np.testing.assert_array_equal(DensityMap.from_file(path, origin="start").origin, [4, -9, 16])
    with pytest.raises(ValueError, match="max_voxels"):
        DensityMap.from_file(path, max_voxels=2)
    path.write_bytes(b"bad")
    with pytest.raises((ValueError, gzip.BadGzipFile)):
        DensityMap.from_file(path)


def gaussian():
    xyz = np.indices((30, 28, 26)).transpose(1, 2, 3, 0) - [14, 13, 12]
    return DensityMap(np.exp(-np.sum(xyz * xyz, axis=-1) / 32), spacing=(1, 2, 3), origin=(-14, -26, -36))


def test_surface_physical_coordinates_normals_and_cache():
    density = gaussian()
    surface = density.isosurface(0.5, units="absolute")
    mesh = surface.mesh_data()
    radius = np.linalg.norm(mesh["vertices"] / [1, 2, 3], axis=1)
    np.testing.assert_allclose(radius, np.sqrt(32 * np.log(2)), atol=0.08)
    assert np.all(np.einsum("ij,ij->i", mesh["vertices"], mesh["normals"]) > 0)
    assert surface.mesh_data() is mesh
    surface.set_level(2)
    assert not len(surface.mesh_data()["faces"])


def test_slice_trilinear_colors_skew_basis_and_crop(protein):
    indices = np.indices((8, 9, 10))
    data = indices[0] + 2 * indices[1] + 3 * indices[2]
    basis = np.array([[2, 0.4, 0], [0, 3, 0.5], [0, 0, 1.0]])
    density = DensityMap(data, basis=basis, origin=(10, 20, 30))
    point = density.origin + basis @ [1.5, 2.5, 3.5]
    assert density.sample(point) == pytest.approx(17)
    slice_ = density.slice("z", 0.5, resolution=8)
    mesh = slice_.mesh_data()
    grid = (mesh["vertices"] - density.origin) @ np.linalg.inv(basis).T
    np.testing.assert_allclose(grid[:, 2], 4.5, atol=1e-5)
    np.testing.assert_allclose(
        mesh["colors"], slice_.color_scale.map(density.sample(mesh["vertices"])), atol=1e-5
    )
    # A periodic crop can cross a unit-cell edge while retaining contour statistics.
    d = DensityMap(np.arange(8 * 9 * 10).reshape(8, 9, 10), periodic=True)
    region = protein.select(residues=1, atoms="CA")
    protein.set_positions(protein.positions - region.positions.mean(0))
    local = d.crop(region, padding=2)
    assert local.mean == d.mean and local.std == d.std and not local.periodic
    assert local.sample([0, 0, 0]) == d.sample([0, 0, 0])
    assert local.sample([-1, 0, 0]) == d.sample([-1, 0, 0])


def test_density_animation_and_follow_seek(protein):
    density = gaussian()
    surface = density.isosurface(0.5, units="absolute", follow=protein)
    slice_ = density.slice("x", 0, follow=protein, resolution=12)
    scene = ProteinScene().add(protein, surface, slice_)
    scene.play(
        surface.animate.set_level(0.8),
        slice_.animate.set_slice(1),
        protein.animate.shift([4, 0, 0]),
        run_time=2,
    )
    scene.seek(1)
    assert surface.level == pytest.approx(0.65) and slice_.coordinate == 0.5
    first = MeshExporter().meshes(surface)[0]["vertices"].copy()
    scene.seek(2)
    scene.seek(1)
    np.testing.assert_array_equal(MeshExporter().meshes(surface)[0]["vertices"], first)
    scene.camera.frame(surface)
    expected = surface.positions @ surface.model_matrix[:3, :3].T + surface.model_matrix[:3, 3]
    np.testing.assert_allclose(scene.camera.target, (expected.max(0) + expected.min(0)) / 2)


@pytest.mark.gpu
def test_native_density_surface_slice_transparency_and_seek():
    from proteinmotion.renderer import Renderer

    d = gaussian()
    surface = d.isosurface(0.4, units="absolute", opacity=0.35)
    slice_ = d.slice("z", 0.2, resolution=24)
    scene = ProteinScene(width=320, height=180).add(surface, slice_)
    scene.camera.frame(surface)
    scene.play(slice_.animate.set_slice(0.8), run_time=1)
    with Renderer(320, 180) as r:
        a = scene.render_frame(0, renderer=r)
        b = scene.render_frame(1, renderer=r)
        again = scene.render_frame(0, renderer=r)
    np.testing.assert_array_equal(a, again)
    assert np.abs(a.astype(float) - b).sum() > 1000


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("PROTEINMOTION_TEST_EEVEE") != "1", reason="opt-in Blender render")
def test_eevee_density_and_plot_render():
    from proteinmotion import ColorLegend, ColorScale

    d = gaussian()
    surface = d.isosurface(0.4, units="absolute")
    scene = ProteinScene(width=320, height=180).add(
        surface, d.slice(resolution=24), ColorLegend(ColorScale(0, 1))
    )
    scene.camera.frame(surface)
    scene.wait(1)
    frame = scene.render_frame(0, renderer="eevee", eevee=EEVEEOptions(samples=8, supersampling=1))
    assert np.std(frame[:, :, :3]) > 10


def test_coarse_surface_without_crossings_is_empty():
    data = np.zeros((16, 16, 16))
    data[5, 5, 5] = 1
    shell = DensityMap(data).isosurface(0.5, units="absolute", step_size=8)
    assert not shell.mesh_data()["faces"].size
