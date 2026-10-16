"""Independent atom/residue appearance tracks, evaluated identically on CPU and GPU.

Every Protein holds ``_appearance``, a read-only float32 array with one 24-column row per
atom (layout at initial_appearance()). It carries three tracks: a color tint, an opacity
multiplier, and a ball-and-stick detail weight that draws an atom as sticks over a cartoon,
ribbon or surface. Each track stores a value before and after a transition, the
transition's start and span in scene seconds, and an easing flag. The protein's clocks
``_color_mix``, ``_opacity_mix`` and ``_detail_mix`` say where each track is in time.

The native and Studio renderers upload the array and evaluate the tracks per atom in the
shaders; CPU code (EEVEE, Studio surfaces, plots, nucleotide bases) calls current_tints()
and the other current_*() functions here. set_color(), set_opacity() and set_detail() change a track at
once; Colorize, SetOpacity, ShowAtoms and their relatives animate it residue by residue.
"""

import numpy as np

from .animation import Animation
from .math3d import color as parse_color
from .rates import evaluate, linear, resolve, smooth

# Columns of an appearance row. The shaders read a row as six vec4s, in the order of
# struct Appearance in molecule.wgsl (before, after, timing, alpha, detail, detail_timing).
#   0:4     tint before: RGB already scaled by the strength, then the strength a;
#           atoms draw as base color × (1 − a) + RGB
#   4:8     tint after
#   8, 9    tint start and span, in scene seconds
#   10      tint easing flag (1 smooth, 0 linear)
#   11      opacity easing flag
#   12, 13  opacity before and after
#   14, 15  opacity start and span
#   16, 17  detail before and after: ball-and-stick weight in [0, 1]
#   18, 19  detail start and span
#   20      detail easing flag; 21:24 pad the row to six vec4s


def initial_appearance(count, detail=None):
    """Appearance rows for ``count`` atoms with no tint, full opacity and no transition.

    ``detail`` sets each atom's ball-and-stick weight; None leaves it at 0.
    """
    data = np.zeros((count, 24), np.float32)
    # Every track spans 1 s from time 0 with equal before and after values, so it reads
    # the same at any clock. Opacity is 1 before and after.
    data[:, 9] = data[:, 15] = data[:, 19] = 1
    data[:, 12] = data[:, 13] = 1
    if detail is not None:
        data[:, 16] = data[:, 17] = detail
    data.flags.writeable = False
    return data


def progress(clock, start, span, smooth):
    """Per-atom transition progress in [0, 1] at scene time ``clock``.

    ``smooth`` is a boolean mask that picks smootherstep over linear progress for each atom.
    The shaders' ease() uses the same polynomial, mirrored about 0.5 like rates.smooth().
    """
    t = np.clip((clock - start) / np.maximum(span, 1e-8), 0, 1)
    return np.where(smooth, t * t * t * (10 + t * (-15 + 6 * t)), t)


def current_tints(protein):
    """Current tint per atom, (atoms, 4); draw base color × (1 − a) + RGB."""
    s = protein._appearance
    t = progress(protein._color_mix, s[:, 8], s[:, 9], s[:, 10] > 0.5)[:, None]
    return (1 - t) * s[:, :4] + t * s[:, 4:8]


def current_opacities(protein):
    """Current per-atom opacity multiplier, (atoms,), not yet multiplied by the object's opacity."""
    s = protein._appearance
    t = progress(protein._opacity_mix, s[:, 14], s[:, 15], s[:, 11] > 0.5)
    return (1 - t) * s[:, 12] + t * s[:, 13]


def current_details(protein):
    """Current ball-and-stick detail weight per atom, (atoms,), in [0, 1]."""
    s = protein._appearance
    t = progress(protein._detail_mix, s[:, 18], s[:, 19], s[:, 20] > 0.5)
    return (1 - t) * s[:, 16] + t * s[:, 17]


def atom_weights(protein):
    """Per-atom ball-and-stick visibility: the representation blend, or per-atom detail."""
    # representation[2] is the ball-and-stick share of the whole-protein representation.
    ball = float(protein.representation[2])
    return ball + (1 - ball) * current_details(protein)


def selection(target):
    """Return the protein and atom indices a styling call acts on: a Region's atoms, or all atoms."""
    from .protein import Protein
    from .regions import Region

    if isinstance(target, Region):
        target._validate()
        return target.protein, target.atom_indices
    if isinstance(target, Protein):
        return target, np.arange(len(target.topology.atoms))
    raise TypeError("Styling requires a Protein or Region")


# The three setters below change a track at once. Each freezes every atom at its current
# value, overwrites the selected atoms, and marks the track finished (start 0, span 1 s,
# linear, clock 1). Arrays are replaced rather than edited, because snapshots share them.


def set_color(target, value):
    """Tint the target's atoms with ``value`` at once; None removes the tint."""
    protein, ids = selection(target)
    tint = np.r_[parse_color(value), 1] if value is not None else np.zeros(4)
    data = protein._appearance.copy()
    current = current_tints(protein)
    current[ids] = tint
    data[:, :4] = data[:, 4:8] = current
    data[:, 8:11] = [0, 1, 0]
    data.flags.writeable = False
    protein._appearance, protein._color_mix = data, 1.0
    return target


def set_opacity(target, value):
    """Set the target's per-atom opacity to ``value`` in [0, 1] at once."""
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Opacity must be finite and in [0, 1]")
    protein, ids = selection(target)
    data = protein._appearance.copy()
    current = current_opacities(protein)
    current[ids] = value
    data[:, 12] = data[:, 13] = current
    data[:, 14:16] = [0, 1]
    data[:, 11] = 0
    data.flags.writeable = False
    protein._appearance, protein._opacity_mix = data, 1.0
    return target


def set_detail(target, value):
    """Set the target's ball-and-stick detail weight to ``value`` (0 hidden, 1 shown) at once."""
    protein, ids = selection(target)
    data = protein._appearance.copy()
    current = current_details(protein)
    current[ids] = value
    data[:, 16] = data[:, 17] = current
    data[:, 18:21] = [0, 1, 0]
    data.flags.writeable = False
    protein._appearance, protein._detail_mix = data, 1.0
    return target


# Per track: (from, to, start, span, easing) columns, the column range an animation
# owns, the current-value evaluator, and the protein clock attribute. The opacity easing
# flag sits in column 11, before its values, so the opacity range starts there.
_TRACKS = {
    "color": ((slice(0, 4), slice(4, 8), 8, 9, 10), (0, 11), current_tints, "_color_mix"),
    "opacity": ((12, 13, 14, 15, 11), (11, 16), current_opacities, "_opacity_mix"),
    "detail": ((16, 17, 18, 19, 20), (16, 21), current_details, "_detail_mix"),
}


class _AppearanceAnimation(Animation):
    """Animate one appearance track (``kind``) of the selected atoms to ``value``.

    Residues can start one after another: ``residue_delay`` (or ``delay_seconds``) gives
    the gap in seconds, or ``stagger`` spreads the starts over that fraction of the clip.
    ``easing`` shapes each residue's own transition.
    """

    # Residue start times are in seconds, so play() must not warp the clip clock.
    requires_linear_timeline = True

    def _set_easing(self, easing):
        self.easing = resolve(easing)

    def __init__(
        self,
        target,
        value,
        *,
        residue_delay=0,
        delay_seconds=None,
        stagger=None,
        reverse=False,
        easing="smooth",
    ):
        if delay_seconds is not None:
            if residue_delay:
                raise ValueError("Use delay_seconds or its legacy alias residue_delay, not both")
            residue_delay = delay_seconds
        self.protein, self.ids = selection(target)
        # The clip clock stays linear; ``easing`` is applied per residue instead.
        super().__init__(self.protein, rate_func=linear)
        # One channel per atom and track: animations on different atoms can share a clip,
        # and the scene rejects two that write the same track of the same atom at once.
        self.channels = frozenset((self.kind, int(i)) for i in self.ids)
        if not np.isfinite(residue_delay) or residue_delay < 0:
            raise ValueError("residue_delay must be finite and nonnegative")
        if stagger is not None and (not np.isfinite(stagger) or not 0 <= stagger < 1):
            raise ValueError("stagger must be the fraction of the clip reserved for staggering, in [0, 1)")
        if stagger is not None and residue_delay:
            raise ValueError("Choose stagger (fraction) or residue_delay (seconds), not both")
        self.stagger = stagger
        easing = resolve(easing)
        self.value, self.residue_delay, self.reverse, self.easing = (
            value,
            float(residue_delay),
            reverse,
            easing,
        )

    def bind(self):
        super().bind()
        protein = self.protein
        # Rank the selected atoms by residue, N to C, or C to N with reverse.
        residues = np.array([protein.topology.atoms[i].residue_index for i in self.ids])
        order = np.unique(residues)
        ranks = np.searchsorted(order, residues)
        if self.reverse:
            ranks = len(order) - 1 - ranks
        # Seconds between successive residue starts. Each residue's transition lasts what
        # is left of the clip after the last residue starts.
        duration = getattr(self, "run_time", 1.0)
        delay = (
            duration * self.stagger / max(1, len(order) - 1)
            if self.stagger is not None
            else self.residue_delay
        )
        span = duration - (len(order) - 1) * delay
        if span <= 0:
            raise ValueError(
                f"run_time must exceed total residue delay: {len(order)} residues need more "
                f"than {(len(order) - 1) * delay:g}s; use stagger=0.5 to fit the clip"
            )
        self.start_time = getattr(self, "start_time", 0.0)
        self.duration = duration
        self.begin, self.span = self.start_time + ranks * delay, span
        # This animation's copy of the appearance array: each selected atom runs from its
        # current value to the new one, starting at its residue's time in scene seconds.
        # The GPU knows only smooth and linear progress; apply() evaluates other easings.
        self.track = protein._appearance.copy()
        (before, after, begin, width, eased), written, current, _ = _TRACKS[self.kind]
        self.track[self.ids, before] = current(protein)[self.ids]
        self.track[self.ids, after] = self.value
        self.track[self.ids, begin] = self.begin
        self.track[self.ids, width] = self.span
        self.track[self.ids, eased] = self.easing is smooth
        self.track.flags.writeable = False
        # The cells this animation owns (selected rows × this track's columns), and a mask
        # of every other cell, for merging in apply().
        self._write_indices = np.ix_(self.ids, np.arange(*written))
        self._other_mask = np.ones(self.track.shape, bool)
        self._other_mask[self._write_indices] = False
        self._combined = None

    def apply(self, alpha):
        # Separate tracks can run together without replacing one another's fields.
        protein = self.protein
        other = protein._appearance
        track = self.track
        if self.easing not in (smooth, linear):
            # Evaluate a custom easing per atom on the CPU and write the result as both the
            # before and the after value, so the shader's own progress makes no difference.
            track = self.track.copy()
            (before, after, begin, width, eased), _, _, _ = _TRACKS[self.kind]
            local = np.clip((self.start_time + float(alpha) * self.duration - self.begin) / self.span, 0, 1)
            eased_progress = np.array([evaluate(self.easing, t) for t in local])
            if isinstance(before, slice):
                eased_progress = eased_progress[:, None]
            values = (1 - eased_progress) * track[self.ids, before] + eased_progress * track[self.ids, after]
            track[self.ids, before] = track[self.ids, after] = values
            self._combined = None
        # Merge this animation's cells into the protein's current array. Reuse the merged
        # array while the other cells are unchanged: the renderer uploads _appearance again
        # only when it becomes a different object.
        if self._combined is None or not np.array_equal(
            self._combined[self._other_mask], other[self._other_mask]
        ):
            data = other.copy()
            data[self._write_indices] = track[self._write_indices]
            data.flags.writeable = False
            self._combined = data
        protein._appearance = self._combined
        # The track's clock is the scene time; begin and span are in the same seconds.
        setattr(protein, _TRACKS[self.kind][3], self.start_time + float(alpha) * self.duration)


class Colorize(_AppearanceAnimation):
    """Fade to a color across every representation; None restores the base palette."""

    kind = "color"
    channels = frozenset({"color"})

    def __init__(self, target, color, **kwargs):
        tint = np.r_[parse_color(color), 1] if color is not None else np.zeros(4)
        super().__init__(target, tint, **kwargs)


class SetOpacity(_AppearanceAnimation):
    """Set object opacity for a Protein, or atom opacity for a Region.

    Use scope='residues' to change all of a protein's per-residue opacities.
    """

    kind = "opacity"
    channels = frozenset({"residue_opacity"})

    def _set_easing(self, easing):
        if self._global:
            self.rate_func = resolve(easing)
        else:
            super()._set_easing(easing)

    def __init__(self, target, opacity, *, scope=None, **kwargs):
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("Opacity must be finite and in [0, 1]")
        from .protein import Protein

        if scope not in (None, "object", "residues"):
            raise ValueError("scope must be 'object' or 'residues'")
        # A whole Protein animates its object opacity, the value FadeIn, FadeOut and
        # Protein.set_opacity() use. A Region, or scope='residues', animates the per-atom track.
        self._global = isinstance(target, Protein) and scope != "residues"
        if scope == "object" and not isinstance(target, Protein):
            raise ValueError("Object opacity requires a Protein; regions have residue opacity")
        if self._global:
            if set(kwargs) - {"easing"}:
                raise ValueError("Use scope='residues' for staggered protein opacity")
            # A plain Animation: one value with no residue delays, so the clip clock may be eased.
            Animation.__init__(self, target, easing=kwargs.get("easing", "smooth"))
            self.value, self.channels, self.requires_linear_timeline = (
                float(opacity),
                frozenset({"opacity"}),
                False,
            )
        else:
            super().__init__(target, float(opacity), **kwargs)

    def bind(self):
        if self._global:
            Animation.bind(self)
            self.start = self.target.opacity
        else:
            super().bind()

    def apply(self, alpha):
        if self._global:
            self.target.opacity = (1 - alpha) * self.start + alpha * self.value
        else:
            super().apply(alpha)


class ShowAtoms(_AppearanceAnimation):
    """Grow atoms in as ball-and-stick over a cartoon, ribbon or surface.

    With residue_delay the atoms appear residue by residue, N to C (reverse=True for C to N).
    Bonds appear once both of their atoms are visible.
    """

    kind = "detail"
    channels = frozenset({"atom_detail"})
    visible = 1.0

    def __init__(self, target, **kwargs):
        super().__init__(self._atoms(target), self.visible, **kwargs)

    @staticmethod
    def _atoms(target):
        # Which atoms to animate; ShowSideChains narrows the target to side chains.
        return target


class HideAtoms(ShowAtoms):
    """Fade ball-and-stick detail atoms back into the cartoon, optionally residue by residue."""

    visible = 0.0


class ShowSideChains(ShowAtoms):
    """Show amino acid side chains as ball-and-stick, attached at Cα to the cartoon."""

    @staticmethod
    def _atoms(target):
        return side_chains(target)


class HideSideChains(ShowSideChains):
    """Fade amino acid side chains back into the cartoon."""

    visible = 0.0


# Amino acid backbone atoms, including terminal and backbone hydrogens.
BACKBONE = frozenset({"N", "C", "O", "OXT", "HXT", "H", "H1", "H2", "H3", "HN", "HA", "HA2", "HA3"})


def side_chains(target):
    """Region of side-chain atoms plus Cα (and proline N) for the target's amino acids.

    Cα joins the sticks to the cartoon, which passes through it. Glycine is skipped.
    """
    from .regions import Region

    protein, ids = selection(target)
    topology = protein.topology
    owners = np.array([a.residue_index for a in topology.atoms])
    wanted = np.unique(owners[ids])
    keep = []
    for ri in wanted:
        r = topology.residues[ri]
        # Only amino acids on a traced chain have a side chain hanging off the cartoon.
        if r.is_nucleic or r.ca < 0 or topology.residue_categories[ri] != "polymer":
            continue
        # Proline's side chain closes a ring onto its own N, so N stays with the sticks.
        backbone = BACKBONE - {"N"} if r.name == "PRO" else BACKBONE
        members = [
            int(i)
            for i in np.flatnonzero(owners == ri)
            if i != r.ca and topology.atoms[i].name not in backbone
        ]
        # A residue needs at least one heavy side-chain atom; this skips glycine, whose Cα
        # hydrogens HA2 and HA3 count as backbone.
        if any(topology.atoms[i].element not in ("H", "D") for i in members):
            keep.extend((r.ca, *members))
    if not keep:
        raise ValueError("Selection contains no amino acid side chains")
    return Region(protein, np.array(keep, dtype=int))


class RegionAnimate:
    """Compatibility factory for the shared fluent animation builder."""

    # RegionAnimate(region) returns an Animate built on the region. Because that is not a
    # RegionAnimate, Python skips __init__, and region.animate chains go through Animate;
    # the methods below are not reached that way.

    def __new__(cls, region):
        from .animation import Animate

        return Animate(region)

    def __init__(self, region):
        self.region = region

    def set_color(self, color, **kwargs):
        return Colorize(self.region, color, **kwargs)

    def set_opacity(self, opacity, **kwargs):
        return SetOpacity(self.region, opacity, **kwargs)

    def show_atoms(self, **kwargs):
        """Animate these atoms in as ball-and-stick detail (see ShowAtoms)."""
        return ShowAtoms(self.region, **kwargs)

    def hide_atoms(self, **kwargs):
        """Animate these detail atoms out (see HideAtoms)."""
        return HideAtoms(self.region, **kwargs)
