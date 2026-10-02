"""Residue measurements, numerical color scales, and animated property styling."""

import numpy as np

from .math3d import color
from .styling import _AppearanceAnimation

# Residue hydropathy (positive = hydrophobic). Kyte & Doolittle, J. Mol. Biol. 157,
# 105–132 (1982); Eisenberg normalized consensus, J. Mol. Biol. 179, 125–142 (1984).
HYDROPATHY = {
    "kyte-doolittle": dict(
        ALA=1.8, ARG=-4.5, ASN=-3.5, ASP=-3.5, CYS=2.5, GLN=-3.5, GLU=-3.5, GLY=-0.4, HIS=-3.2, ILE=4.5,
        LEU=3.8, LYS=-3.9, MET=1.9, PHE=2.8, PRO=-1.6, SER=-0.8, THR=-0.7, TRP=-0.9, TYR=-1.3, VAL=4.2,
    ),
    "eisenberg": dict(
        ALA=0.62, ARG=-2.53, ASN=-0.78, ASP=-0.90, CYS=0.29, GLN=-0.85, GLU=-0.74, GLY=0.48, HIS=-0.40,
        ILE=1.38, LEU=1.06, LYS=-1.50, MET=0.64, PHE=1.19, PRO=0.12, SER=-0.18, THR=-0.05, TRP=0.81,
        TYR=0.26, VAL=1.08,
    ),
}  # fmt: skip
# ConSurf's nine conservation grades, from variable (turquoise) to conserved (maroon).
CONSURF = ("#10c8d1", "#8cffff", "#d7ffff", "#eaffff", "#ffffff", "#fcedf4", "#fac9de", "#f07dab", "#a02560")
# AlphaFold DB pLDDT bands: very low < 50 ≤ low < 70 ≤ confident < 90 ≤ very high.
PLDDT = ("#ff7d45", "#ffdb13", "#65cbf3", "#0053d6")


class ColorScale:
    """A fixed linear scale shared by structures, plots, and legends.

    Values outside the limits use the endpoint colors. NaN uses ``missing``.
    Intermediate colors are linearly interpolated between equally spaced stops.
    With ``boundaries`` the scale is banded instead: one color per interval, so
    ``len(colors) == len(boundaries) + 1``.
    Presets: hydropathy(), conservation(), plddt(), pae(), distance(), difference().
    """

    def __init__(
        self, vmin, vmax, *, colors=("#355f9e", "#56c5bd", "#f5d477"), missing="#87909c", boundaries=None
    ):
        if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
            raise ValueError("ColorScale needs finite limits with vmax > vmin")
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.colors = np.array([color(c) for c in colors])
        if len(self.colors) < 2:
            raise ValueError("Supply at least two color stops")
        self.colors.flags.writeable = False
        self.missing = color(missing)
        self.boundaries = None
        if boundaries is not None:
            bounds = np.asarray(boundaries, dtype=float)
            if bounds.ndim != 1 or len(bounds) != len(self.colors) - 1 or not np.isfinite(bounds).all():
                raise ValueError("Banded scales need one fewer boundary than colors")
            if np.any(np.diff(bounds) <= 0):
                raise ValueError("Boundaries must increase")
            bounds.flags.writeable = False
            self.boundaries = bounds

    @classmethod
    def hydropathy(cls, scale="kyte-doolittle"):
        """Hydrophilic blue, neutral near-white, hydrophobic orange; symmetric about 0."""
        if scale not in HYDROPATHY:
            raise ValueError(f"Unknown hydropathy scale {scale!r}; choose from {', '.join(HYDROPATHY)}")
        limit = max(abs(v) for v in HYDROPATHY[scale].values())
        return cls(-limit, limit, colors=("#3f7fd8", "#e8edf2", "#e8913a"))

    @classmethod
    def conservation(cls, vmin=0.0, vmax=1.0):
        """ConSurf's palette from variable (turquoise) through white to conserved (maroon)."""
        return cls(vmin, vmax, colors=CONSURF)

    @classmethod
    def plddt(cls):
        """AlphaFold's four confidence bands, split at pLDDT 50, 70 and 90."""
        return cls(0, 100, colors=PLDDT, boundaries=(50, 70, 90))

    @classmethod
    def pae(cls, max_error=31.75):
        """Predicted aligned error in Å: confident dark green to uncertain near-white."""
        return cls(0, max_error, colors=("#0e5a2e", "#4f9d63", "#b6dcbd", "#f2f6f2"))

    @classmethod
    def distance(cls, max_distance=40.0):
        """Short distances bright, long distances dark."""
        return cls(0, max_distance, colors=("#f5d477", "#e07b4f", "#7b3f8c", "#1f2a4f"))

    @classmethod
    def difference(cls, limit=10.0):
        """A signed change in Å: closer in blue, unchanged near-white, farther in red."""
        return cls(-limit, limit, colors=("#3f7fd8", "#e8edf2", "#e0604a"))

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
        if self.boundaries is not None:
            values = np.asarray(values, dtype=float)
            bands = np.searchsorted(
                self.boundaries, np.nan_to_num(values, nan=self.vmin).ravel(), side="right"
            )
            result = self.colors[bands].copy()
            result[~np.isfinite(values).ravel()] = self.missing
            return result.reshape(values.shape + (3,)).astype(np.float32)
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

    def __init__(self, protein, values, *, name="Value", unit="", scale=None):
        self.keys = tuple((r.chain, r.resid, r.icode, r.name) for r in protein.topology.residues)
        self.values = np.array(values, dtype=float, copy=True)
        if self.values.shape != (len(self.keys),) or np.isinf(self.values).any():
            raise ValueError("Supply one finite value or NaN per topology residue")
        self.values.flags.writeable = False
        self.name, self.unit = str(name), str(unit)
        if scale is not None and not isinstance(scale, ColorScale):
            raise TypeError("scale must be a ColorScale")
        self._scale = scale

    @property
    def scale(self):
        """The preset ColorScale for these values, or one fitted to their range."""
        return self._scale if self._scale is not None else ColorScale.from_values(self.values)

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
    def hydropathy(cls, protein, scale="kyte-doolittle"):
        """Residue hydropathy (positive = hydrophobic); NaN for non-amino acids.

        Scales: "kyte-doolittle" (default) or "eisenberg". Modified residues use
        their parent amino acid where the structure records one.
        """
        import gemmi

        if scale not in HYDROPATHY:
            raise ValueError(f"Unknown hydropathy scale {scale!r}; choose from {', '.join(HYDROPATHY)}")
        table, values = HYDROPATHY[scale], []
        for r in protein.topology.residues:
            name = r.name.upper()
            if name not in table and not r.is_nucleic:
                code = gemmi.find_tabulated_residue(name)
                parent = code.one_letter_code.upper() if code is not None else ""
                name = next(
                    (k for k in table if gemmi.find_tabulated_residue(k).one_letter_code == parent), name
                )
            values.append(table.get(name, np.nan) if not r.is_nucleic else np.nan)
        label = {"kyte-doolittle": "Hydropathy (Kyte–Doolittle)", "eisenberg": "Hydrophobicity (Eisenberg)"}[
            scale
        ]
        return cls(protein, values, name=label, scale=ColorScale.hydropathy(scale))

    @classmethod
    def conservation(cls, protein, alignment, *, chain=None, query=None, weighting=True):
        """Sequence conservation from 0 (variable) to 1 (invariant) for each residue.

        ``alignment`` is a FASTA, A3M, Stockholm or Clustal file, or a list of
        aligned sequences. The query (``query`` by name or index, by default the
        sequence most similar to the structure) is aligned to every protein chain
        it matches (``chain`` restricts this). The score is 1 − H/ln 20, where H is
        the Shannon entropy of the column's amino acids with Henikoff sequence
        weights, scaled by the fraction of sequences without a gap there. The
        preset color scale spans the 5th–95th percentile of this protein's scores.
        """
        from .conservation import residue_conservation

        values = residue_conservation(protein, alignment, chain=chain, query=query, weighting=weighting)
        # Like ConSurf grades, the preset scale spans this protein's own range of scores.
        lo, hi = np.round(np.nanpercentile(values, [5, 95]), 2)
        scale = ColorScale.conservation(lo, hi) if hi > lo else ColorScale.conservation()
        return cls(protein, values, name="Conservation", scale=scale)

    @classmethod
    def plddt(cls, protein, source=None):
        """AlphaFold pLDDT per residue, read from the B-factor field or a confidence JSON file.

        AlphaFold models store pLDDT in the B-factor column; ``source`` may instead
        be an AlphaFold DB ``confidence_v*.json`` file or a ColabFold scores file.
        """
        if source is None:
            values = cls.b_factors(protein).values
        else:
            from .confidence import read_plddt

            values = read_plddt(protein, source)
        return cls(protein, values, name="pLDDT", unit="", scale=ColorScale.plddt())

    @classmethod
    def b_factors(cls, protein, *, atoms="backbone", name="B factor", unit="Å²"):
        """Read first-model B factors at Cα/C4′/P anchors, or named atoms.

        Use atoms=None for the mean of each residue's atoms.

        Files containing confidence in the B-factor field can use name='pLDDT', unit=''.
        The meaning of this field comes from the source file.
        """
        values = np.full(len(protein.topology.residues), np.nan)
        groups = [[] for _ in values]
        anchors = {r.trace_atom for r in protein.topology.residues}
        for i, atom in enumerate(protein.topology.atoms):
            if (
                atoms is None
                or (
                    i in anchors
                    if atoms == "backbone"
                    else atom.name.replace("*", "'") == atoms.replace("*", "'")
                )
            ) and np.isfinite(atom.bfactor):
                groups[atom.residue_index].append(atom.bfactor)
        for i, group in enumerate(groups):
            if group:
                values[i] = np.mean(group)
        return cls(protein, values, name=name, unit=unit)

    @classmethod
    def rmsf(cls, protein, trajectory=None, *, align=True, alignment=None, atoms="backbone", stride=1):
        """Root-mean-square fluctuation about the mean position, in Å.

        Align frames to the first sampled frame using Cα/C4′/P anchors or an explicit Region.
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
        ids = [r.trace_atom for r in protein.topology.residues if r.trace_atom >= 0]
        anchors = set(ids)
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
                xyz = align_coordinates(xyz, reference, ids if len(ids) >= 3 else None)
            delta = xyz - mean
            mean += delta / count
            m2 += delta * (xyz - mean)
        msf = m2.sum(1) / count
        groups = [[] for _ in protein.topology.residues]
        for i, atom in enumerate(protein.topology.atoms):
            if atoms is None or (
                i in anchors
                if atoms == "backbone"
                else atom.name.replace("*", "'") == atoms.replace("*", "'")
            ):
                groups[atom.residue_index].append(msf[i])
        return cls(protein, [np.sqrt(np.mean(g)) if g else np.nan for g in groups], name="RMSF", unit="Å")


def resolve_values(protein, values):
    """ResidueValues from values, an array, or a preset name: hydropathy or plddt."""
    if isinstance(values, str):
        presets = {"hydropathy": ResidueValues.hydropathy, "plddt": ResidueValues.plddt}
        if values.lower() not in presets:
            raise ValueError(f"Unknown residue property {values!r}; use hydropathy, plddt or ResidueValues")
        return presets[values.lower()](protein)
    return values if isinstance(values, ResidueValues) else ResidueValues(protein, values)


def _style(protein, values, scale, thickness):
    values = resolve_values(protein, values)
    values._validate(protein)
    scale = values.scale if scale is None else scale
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
        self,
        protein,
        values,
        *,
        scale=None,
        thickness=None,
        residue_delay=0,
        delay_seconds=None,
        stagger=None,
        reverse=False,
        easing="smooth",
    ):
        tint, self.widths = _style(protein, values, scale, thickness)
        super().__init__(
            protein,
            tint,
            residue_delay=residue_delay,
            delay_seconds=delay_seconds,
            stagger=stagger,
            reverse=reverse,
            easing=easing,
        )
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
            from .rates import evaluate

            delay = (
                self.duration * self.stagger / max(1, len(ranks) - 1)
                if self.stagger is not None
                else self.residue_delay
            )
            local = np.clip((alpha * self.duration - ranks * delay) / self.span, 0, 1)
            t = np.array([evaluate(self.easing, value) for value in local])
            widths = ((1 - t) * self.start_widths + t * self.widths).astype(np.float32)
            widths.flags.writeable = False
            self.protein._cartoon_scale = widths
