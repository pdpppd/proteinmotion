"""Residue measurements, numerical color scales, and animated property styling."""

import numpy as np

from .math3d import color
from .styling import _AppearanceAnimation, progress


class ColorScale:
    """A fixed linear scale shared by structures, plots, and legends.

    Values outside the limits use the endpoint colors. NaN uses ``missing``.
    Intermediate colors are linearly interpolated between equally spaced stops.
    """

    def __init__(self, vmin, vmax, *, colors=("#355f9e", "#56c5bd", "#f5d477"), missing="#87909c"):
        if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
            raise ValueError("ColorScale needs finite limits with vmax > vmin")
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.colors = np.array([color(c) for c in colors])
        if len(self.colors) < 2:
            raise ValueError("Supply at least two color stops")
        self.colors.flags.writeable = False
        self.missing = color(missing)

    @classmethod
    def from_values(cls, values, **kwargs):
        """Choose limits from finite values; expand a constant range by ±0.5."""
        a = np.asarray(getattr(values, "values", values), dtype=float)
        a = a[np.isfinite(a)]
        if not len(a):
            raise ValueError("A color scale needs at least one finite value")
        lo, hi = float(a.min()), float(a.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return cls(lo, hi, **kwargs)

    def normalize(self, values):
        """Return clipped values in [0, 1], preserving NaN for missing data."""
        return np.clip((np.asarray(values, dtype=float) - self.vmin) / (self.vmax - self.vmin), 0, 1)

    def map(self, values):
        """Return RGB colors with shape (*values.shape, 3)."""
        t = self.normalize(values)
        safe = np.nan_to_num(t, nan=0).ravel()
        stops = np.linspace(0, 1, len(self.colors))
        result = np.stack([np.interp(safe, stops, self.colors[:, i]) for i in range(3)], -1)
        result[~np.isfinite(t).ravel()] = self.missing
        return result.reshape(t.shape + (3,)).astype(np.float32)


class ResidueValues:
    """One value per topology residue, in topology order. NaN denotes missing data.

    ``name`` and ``unit`` describe the measurement; they do not change its values.
    Residue identity is checked before applying values to another protein.
    """

    def __init__(self, protein, values, *, name="Value", unit=""):
        self.keys = tuple((r.chain, r.resid, r.icode, r.name) for r in protein.topology.residues)
        self.values = np.array(values, dtype=float, copy=True)
        if self.values.shape != (len(self.keys),) or np.isinf(self.values).any():
            raise ValueError("Supply one finite value or NaN per topology residue")
        self.values.flags.writeable = False
        self.name, self.unit = str(name), str(unit)

    def _validate(self, protein):
        keys = tuple((r.chain, r.resid, r.icode, r.name) for r in protein.topology.residues)
        if keys != self.keys:
            raise ValueError("Residue values belong to a different residue identity/order")

    @classmethod
    def from_mapping(cls, protein, values, *, name="Value", unit=""):
        """Read {(chain, residue_number[, insertion_code]): value}; omissions become NaN."""
        keys = [(r.chain, r.resid, r.icode) for r in protein.topology.residues]
        output = np.full(len(keys), np.nan)
        for key, value in values.items():
            if not isinstance(key, tuple) or len(key) not in (2, 3):
                raise ValueError("Keys must be (chain, residue_number[, insertion_code])")
            ids = [i for i, k in enumerate(keys) if k[: len(key)] == key]
            if len(ids) != 1:
                raise ValueError(f"Residue key {key!r} matches {len(ids)} residues; include insertion codes")
            output[ids[0]] = value
        return cls(protein, output, name=name, unit=unit)

    @classmethod
    def b_factors(cls, protein, *, atoms="CA", name="B factor", unit="Å²"):
        """Read first-model B factors. Use atoms=None for the mean of each residue's atoms.

        Files containing confidence in the B-factor field can use name='pLDDT', unit=''.
        The meaning of this field comes from the source file.
        """
        values = np.full(len(protein.topology.residues), np.nan)
        groups = [[] for _ in values]
        for atom in protein.topology.atoms:
            if (atoms is None or atom.name == atoms) and np.isfinite(atom.bfactor):
                groups[atom.residue_index].append(atom.bfactor)
        for i, group in enumerate(groups):
            if group:
                values[i] = np.mean(group)
        return cls(protein, values, name=name, unit=unit)

    @classmethod
    def rmsf(cls, protein, trajectory=None, *, align=True, alignment=None, atoms="CA", stride=1):
        """Root-mean-square fluctuation about the mean position, in Å.

        Align frames to the first sampled frame using Cα atoms or an explicit Region.
        Average atomic mean-square fluctuations within each residue before taking
        the square root. Frames are accumulated one at a time. Unwrap periodic MD
        coordinates before supplying them here.
        """
        from .math3d import align_coordinates

        trajectory = protein.trajectory if trajectory is None else trajectory
        if trajectory.n_atoms != len(protein.topology.atoms):
            raise ValueError("Trajectory atom count does not match protein")
        if trajectory.topology is not None and trajectory.topology.keys != protein.topology.keys:
            raise ValueError("Trajectory atom identities/order do not match protein")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        ids = [r.ca for r in protein.topology.residues if r.ca >= 0]
        if alignment is not None:
            alignment._validate()
            if alignment.protein is not protein:
                raise ValueError("Alignment region must belong to this protein")
            ids = alignment.atom_indices
        reference = trajectory.frame(0)
        mean = np.zeros_like(reference, dtype=float)
        m2 = np.zeros_like(mean)
        for count, frame in enumerate(range(0, len(trajectory), stride), 1):
            xyz = trajectory.frame(frame)
            if align:
                xyz = align_coordinates(xyz, reference, ids)
            delta = xyz - mean
            mean += delta / count
            m2 += delta * (xyz - mean)
        msf = m2.sum(1) / count
        groups = [[] for _ in protein.topology.residues]
        for i, atom in enumerate(protein.topology.atoms):
            if atoms is None or atom.name == atoms:
                groups[atom.residue_index].append(msf[i])
        return cls(protein, [np.sqrt(np.mean(g)) if g else np.nan for g in groups], name="RMSF", unit="Å")


def _style(protein, values, scale, thickness):
    values = values if isinstance(values, ResidueValues) else ResidueValues(protein, values)
    values._validate(protein)
    scale = ColorScale.from_values(values) if scale is None else scale
    if not isinstance(scale, ColorScale):
        raise TypeError("scale must be a ColorScale")
    atom_residues = np.array([a.residue_index for a in protein.topology.atoms])
    tint = np.column_stack((scale.map(values.values)[atom_residues], np.ones(len(atom_residues))))
    widths = None
    if thickness is not None:
        limits = np.asarray(thickness, dtype=float)
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or np.any(limits <= 0)
            or limits[1] < limits[0]
        ):
            raise ValueError("thickness needs positive (minimum, maximum) cartoon scale factors")
        t = scale.normalize(values.values)
        widths = np.where(np.isnan(t), 1, limits[0] + np.nan_to_num(t) * (limits[1] - limits[0]))
        widths = widths.astype(np.float32)
        widths.flags.writeable = False
    return tint, widths


def color_by(protein, values, *, scale=None, thickness=None):
    """Apply property colors and optional cartoon thickness before adding the protein."""
    tint, widths = _style(protein, values, scale, thickness)
    data = protein._appearance.copy()
    data[:, :4] = data[:, 4:8] = tint
    data[:, 8:11] = [0, 1, 0]
    data.flags.writeable = False
    protein._appearance, protein._color_mix = data, 1.0
    if widths is not None:
        protein._cartoon_scale = widths
    return protein


class ColorByProperty(_AppearanceAnimation):
    """Animate property colors and optional thickness, with residue delays in seconds."""

    kind = "color"

    def __init__(
        self, protein, values, *, scale=None, thickness=None, residue_delay=0, reverse=False, easing="smooth"
    ):
        tint, self.widths = _style(protein, values, scale, thickness)
        super().__init__(protein, tint, residue_delay=residue_delay, reverse=reverse, easing=easing)
        if self.widths is not None:
            self.channels |= {"cartoon_thickness"}

    def bind(self):
        super().bind()
        self.start_widths = self.protein._cartoon_scale

    def apply(self, alpha):
        super().apply(alpha)
        if self.widths is not None:
            ranks = np.arange(len(self.widths))
            if self.reverse:
                ranks = ranks[::-1]
            t = progress(
                alpha * self.duration, ranks * self.residue_delay, self.span, self.easing == "smooth"
            )
            widths = ((1 - t) * self.start_widths + t * self.widths).astype(np.float32)
            widths.flags.writeable = False
            self.protein._cartoon_scale = widths
