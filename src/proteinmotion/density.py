"""MRC/CCP4 density maps in physical coordinates, with surfaces and sampled slices."""

from itertools import product

import gemmi
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

from .animation import Animate, Animation
from .math3d import color as parse_color
from .math3d import normalize
from .mesh import MeshObject
from .properties import ColorScale


class DensityMap:
    """Immutable scalar grid. Arrays use (x, y, z) order; coordinates are in Å.

    ``basis`` is an optional 3×3 matrix whose columns are voxel step vectors.
    It supports non-orthogonal crystallographic cells. ``spacing`` is the simpler
    orthogonal equivalent. Values are kept in their original units.
    """

    def __init__(
        self, values, *, spacing=1.0, origin=(0, 0, 0), basis=None, periodic=False, max_voxels=32_000_000
    ):
        data = np.asarray(values)
        if data.ndim != 3 or min(data.shape) < 2 or data.size > max_voxels:
            raise ValueError("Density needs a 3D grid with at least 2 samples per axis, within max_voxels")
        self.periodic = bool(periodic)
        self.values = np.array(data, dtype=np.float32, copy=True, order="C")
        if not np.isfinite(self.values).all():
            raise ValueError("Density values must be finite")
        self.origin = np.asarray(origin, dtype=float).copy()
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("origin must be a finite 3-vector")
        if basis is None:
            spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
            if not np.isfinite(spacing).all() or np.any(spacing <= 0):
                raise ValueError("spacing must be positive and finite")
            basis = np.diag(spacing)
        self.basis = np.array(basis, dtype=float, copy=True)
        if (
            self.basis.shape != (3, 3)
            or not np.isfinite(self.basis).all()
            or abs(np.linalg.det(self.basis)) < 1e-12
        ):
            raise ValueError("basis must be a finite invertible 3×3 matrix")
        self.values.flags.writeable = self.origin.flags.writeable = self.basis.flags.writeable = False
        self.mean, self.std = float(self.values.mean(dtype=float)), float(self.values.std(dtype=float))
        self.minimum, self.maximum = float(self.values.min()), float(self.values.max())

    @classmethod
    def from_file(cls, path, *, origin="auto", max_voxels=32_000_000):
        """Read MRC/CCP4 using Gemmi, preserving cropped maps and axis permutations.

        ``origin='auto'`` uses a nonzero ORIGIN field, otherwise NX/NY/NZSTART.
        Choose 'header' or 'start' explicitly for files using another convention.
        No symmetry expansion, resampling, or density normalization is performed.
        """
        from pathlib import Path

        if origin not in ("auto", "header", "start"):
            raise ValueError("origin must be auto, header, or start")
        # Check allocation before asking Gemmi to load the array.
        import gzip

        open_map = gzip.open if str(path).endswith(".gz") else open
        with open_map(Path(path), "rb") as source:
            header = source.read(1024)
        if len(header) < 1024:
            raise ValueError("Truncated MRC/CCP4 header")
        dimensions = [np.frombuffer(header[:12], dtype=d) for d in ("<i4", ">i4")]
        if not any(np.all(d >= 2) and np.prod(d.astype(object)) <= max_voxels for d in dimensions):
            raise ValueError("Invalid map dimensions or map exceeds max_voxels; crop or downsample the input")
        m = gemmi.read_ccp4_map(str(path), setup=False)
        axes = np.array([m.header_i32(i) - 1 for i in (17, 18, 19)])
        if sorted(axes) != [0, 1, 2]:
            raise ValueError("MRC MAPC/MAPR/MAPS must be a permutation of 1, 2, 3")
        order = np.argsort(axes)
        values = np.asarray(m.grid).transpose(order)
        sampling = np.array([m.header_i32(i) for i in (8, 9, 10)])
        if np.any(sampling <= 0):
            raise ValueError("MRC sampling intervals must be positive")
        cell = m.grid.unit_cell
        if not cell.is_crystal():
            raise ValueError("MRC/CCP4 map needs valid cell dimensions and angles")
        basis = np.asarray(cell.orth.mat) / sampling[None, :]
        start = np.array([m.header_i32(i) for i in (5, 6, 7)])[order]
        header_origin = np.array([m.header_float(i) for i in (50, 51, 52)])
        offset = (
            header_origin
            if origin == "header" or (origin == "auto" and np.any(header_origin != 0))
            else basis @ start
        )
        periodic = 1 <= m.header_i32(23) <= 230 and tuple(values.shape) == tuple(sampling)
        return cls(values, basis=basis, origin=offset, periodic=periodic, max_voxels=max_voxels)

    @property
    def bounds(self):
        """Eight corners in map coordinates, suitable for framing."""
        corners = np.array(list(product(*[(0, n - 1) for n in self.values.shape])))
        return corners @ self.basis.T + self.origin

    def sample(self, points):
        """Trilinear sampling in Å. Periodic grids wrap; other outside points return NaN."""
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Sample points must be finite and shaped (..., 3)")
        ijk = (points - self.origin) @ np.linalg.inv(self.basis).T
        if not self.periodic:
            # Recover boundary samples after a float32 mesh round-trip through Å.
            edge = np.asarray(self.values.shape) - 1
            near = (ijk >= -1e-5) & (ijk <= edge + 1e-5)
            ijk = np.where(near, np.clip(ijk, 0, edge), ijk)
        return map_coordinates(
            self.values,
            ijk.reshape(-1, 3).T,
            order=1,
            mode="grid-wrap" if self.periodic else "constant",
            cval=np.nan,
        ).reshape(points.shape[:-1])

    def crop(self, region, *, padding=4.0):
        """Crop around a Region in its protein's local coordinates, before transforms."""
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("padding must be finite and nonnegative")
        region._validate()
        points = region.positions
        corners = np.array(list(product(*zip(points.min(0) - padding, points.max(0) + padding))))
        ijk = (corners - self.origin) @ np.linalg.inv(self.basis).T
        lo = np.floor(ijk.min(0)).astype(int)
        hi = np.ceil(ijk.max(0)).astype(int) + 1
        if not self.periodic:
            lo, hi = np.maximum(lo, 0), np.minimum(hi, self.values.shape)
        if np.prod((hi - lo).astype(object)) > 32_000_000:
            raise ValueError("Crop exceeds 32 million voxels; reduce padding or region size")
        if np.any(hi - lo < 2):
            raise ValueError("Selected region does not overlap a usable part of the map")
        values = (
            self.values[np.ix_(*[np.arange(a, b) % n for a, b, n in zip(lo, hi, self.values.shape)])]
            if self.periodic
            else self.values[tuple(slice(a, b) for a, b in zip(lo, hi))]
        )
        cropped = DensityMap(
            values,
            basis=self.basis,
            origin=self.origin + self.basis @ lo,
            max_voxels=max(self.values.size, values.size),
        )
        cropped.mean, cropped.std = self.mean, self.std
        return cropped

    def isosurface(self, level=1.0, *, units="sigma", color="#75d5cb", opacity=0.3, step_size=1, follow=None):
        """Create a shaded surface at an absolute or mean-plus-sigma contour level."""
        return DensitySurface(
            self, level, units=units, color=color, opacity=opacity, step_size=step_size, follow=follow
        )

    def slice(self, axis="z", position=0.5, *, scale=None, opacity=1.0, resolution=128, follow=None):
        """Create a colored plane at a fractional grid position in [0, 1]."""
        return DensitySlice(
            self, axis, position, scale=scale, opacity=opacity, resolution=resolution, follow=follow
        )


def _mesh(vertices, faces, colors, normals):
    return dict(
        vertices=np.asarray(vertices, np.float32).reshape(-1, 3),
        faces=np.asarray(faces, np.int32).reshape(-1, 3),
        colors=np.asarray(colors, np.float32).reshape(-1, 3),
        normals=np.asarray(normals, np.float32).reshape(-1, 3),
    )


class DensitySurface(MeshObject):
    """Cached marching-cubes surface. Changing the contour rebuilds its mesh on the CPU."""

    def __init__(
        self, density, level=1, *, units="sigma", color="#75d5cb", opacity=0.3, step_size=1, follow=None
    ):
        super().__init__(opacity=opacity, follow=follow)
        if not isinstance(density, DensityMap):
            raise TypeError("density must be a DensityMap")
        if units not in ("sigma", "absolute"):
            raise ValueError("units must be sigma or absolute")
        if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size < 1:
            raise ValueError("step_size must be a positive integer")
        if units == "sigma" and density.std == 0:
            raise ValueError("A constant map has no sigma contour; use absolute units")
        self.density, self.units, self.step_size = density, units, step_size
        self.color = parse_color(color)
        self.set_level(level)

    @property
    def positions(self):
        return self.density.bounds

    def set_level(self, level):
        """Set the contour in the units chosen at construction."""
        if not np.isfinite(level):
            raise ValueError("Contour level must be finite")
        self.level = float(level)
        return self

    def mesh_data(self):
        key = (self.level, tuple(self.color), self.step_size)
        if key != self._mesh_key:
            d = self.density
            level = d.mean + self.level * d.std if self.units == "sigma" else self.level
            if not d.minimum < level < d.maximum:
                self._mesh = _mesh([], [], [], [])
            else:
                try:
                    vertices, faces, normals, _ = marching_cubes(
                        d.values.copy(), level, step_size=self.step_size, allow_degenerate=False
                    )
                except RuntimeError as exc:
                    if "No surface found" not in str(exc):
                        raise
                    # A coarse grid can skip every crossing of a valid contour.
                    self._mesh, self._mesh_key = _mesh([], [], [], []), key
                    return self._mesh
                vertices = vertices @ d.basis.T + d.origin
                normals = normalize(normals @ np.linalg.inv(d.basis))
                # Orient faces consistently with the outward marching-cubes normals.
                if len(faces):
                    tri = vertices[faces]
                    orient = np.einsum(
                        "ij,ij->i",
                        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                        normals[faces].mean(1),
                    )
                    faces = faces.copy()
                    faces[orient < 0] = faces[orient < 0][:, ::-1]
                self._mesh = _mesh(vertices, faces, np.tile(self.color, (len(vertices), 1)), normals)
            self._mesh_key = key
        return self._mesh


class DensitySlice(MeshObject):
    """Trilinearly sampled grid plane. Colors use a fixed scale throughout motion."""

    def __init__(
        self, density, axis="z", position=0.5, *, scale=None, opacity=1.0, resolution=128, follow=None
    ):
        super().__init__(opacity=opacity, follow=follow)
        if not isinstance(density, DensityMap):
            raise TypeError("density must be a DensityMap")
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be x, y, or z (map grid axes)")
        if isinstance(resolution, bool) or not isinstance(resolution, int) or not 2 <= resolution <= 512:
            raise ValueError("resolution must be an integer from 2 to 512")
        self.density, self.axis, self.resolution = density, "xyz".index(axis), resolution
        self.color_scale = ColorScale.from_values(density.values) if scale is None else scale
        if not isinstance(self.color_scale, ColorScale):
            raise TypeError("scale must be a ColorScale")
        self.unlit = True
        self.set_slice(position)

    @property
    def positions(self):
        return self.density.bounds

    def set_slice(self, position):
        """Set the fractional grid position from 0 to 1 along this slice's axis."""
        if not np.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("Slice position must be finite and in [0, 1]")
        self.coordinate = float(position)
        return self

    def mesh_data(self):
        key = (self.coordinate, self.resolution, id(self.color_scale))
        if key != self._mesh_key:
            d, n = self.density, self.resolution
            axes = [i for i in range(3) if i != self.axis]
            grid = np.zeros((n, n, 3))
            grid[..., axes[0]], grid[..., axes[1]] = np.meshgrid(
                *[np.linspace(0, d.values.shape[i] - 1, n) for i in axes], indexing="ij"
            )
            grid[..., self.axis] = self.coordinate * (d.values.shape[self.axis] - 1)
            values = map_coordinates(d.values, grid.reshape(-1, 3).T, order=1, mode="nearest")
            vertices = grid.reshape(-1, 3) @ d.basis.T + d.origin
            a = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None]).ravel()
            faces = np.concatenate((np.stack((a, a + n, a + 1), -1), np.stack((a + 1, a + n, a + n + 1), -1)))
            normal = normalize(np.cross(d.basis[:, axes[0]], d.basis[:, axes[1]]))
            self._mesh = _mesh(
                vertices, faces, self.color_scale.map(values), np.tile(normal, (len(vertices), 1))
            )
            self._mesh_key = key
        return self._mesh


class DensityAnimate(Animate):
    """Fluent contour or slice animation, plus the usual mesh transforms and fades."""

    def set_level(self, level):
        if self.operations:
            raise ValueError("Play contour and transform animations separately")
        return _DensityParameter(self.target, "level", level)

    def set_slice(self, position):
        if self.operations:
            raise ValueError("Play slice and transform animations separately")
        return _DensityParameter(self.target, "coordinate", position)


class _DensityParameter(Animation):
    def __init__(self, target, field, value):
        super().__init__(target)
        if not hasattr(target, field) or not np.isfinite(value):
            raise ValueError("Choose a finite parameter supported by this density object")
        if field == "coordinate" and not 0 <= value <= 1:
            raise ValueError("Slice position must be in [0, 1]")
        self.field, self.end = field, float(value)
        self.channels = frozenset({field})

    def bind(self):
        super().bind()
        self.start = getattr(self.target, self.field)

    def apply(self, alpha):
        setattr(self.target, self.field, (1 - alpha) * self.start + alpha * self.end)
