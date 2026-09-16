"""Independent atom/residue appearance tracks, evaluated identically on CPU and GPU."""

import numpy as np

from .animation import Animation
from .math3d import color as parse_color
from .rates import linear


def initial_appearance(count):
    # tint before/after (RGB + strength), timing, and opacity before/after/timing.
    data = np.zeros((count, 16), np.float32)
    data[:, 9] = data[:, 13] = data[:, 15] = 1
    data[:, 12] = 1
    data.flags.writeable = False
    return data


def progress(clock, start, span, smooth):
    t = np.clip((clock - start) / np.maximum(span, 1e-8), 0, 1)
    return np.where(smooth, t * t * t * (10 + t * (-15 + 6 * t)), t)


def current_tints(protein):
    s = protein._appearance
    t = progress(protein._color_mix, s[:, 8], s[:, 9], s[:, 10] > 0.5)[:, None]
    return (1 - t) * s[:, :4] + t * s[:, 4:8]


def current_opacities(protein):
    s = protein._appearance
    t = progress(protein._opacity_mix, s[:, 14], s[:, 15], s[:, 11] > 0.5)
    return (1 - t) * s[:, 12] + t * s[:, 13]


def selection(target):
    from .protein import Protein
    from .regions import Region

    if isinstance(target, Region):
        target._validate()
        return target.protein, target.atom_indices
    if isinstance(target, Protein):
        return target, np.arange(len(target.topology.atoms))
    raise TypeError("Styling requires a Protein or Region")


def set_color(target, value):
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


class _AppearanceAnimation(Animation):
    requires_linear_timeline = True

    def __init__(self, target, value, *, residue_delay=0, reverse=False, easing="smooth"):
        self.protein, self.ids = selection(target)
        super().__init__(self.protein, rate_func=linear)
        self.channels = frozenset((self.kind, int(i)) for i in self.ids)
        if not np.isfinite(residue_delay) or residue_delay < 0:
            raise ValueError("residue_delay must be finite and nonnegative")
        if easing not in ("smooth", "linear"):
            raise ValueError("easing must be smooth or linear")
        self.value, self.residue_delay, self.reverse, self.easing = (
            value,
            float(residue_delay),
            reverse,
            easing,
        )

    def bind(self):
        super().bind()
        p = self.protein
        residues = np.array([p.topology.atoms[i].residue_index for i in self.ids])
        order = np.unique(residues)
        ranks = np.searchsorted(order, residues)
        if self.reverse:
            ranks = len(order) - 1 - ranks
        duration = getattr(self, "run_time", 1.0)
        span = duration - (len(order) - 1) * self.residue_delay
        if span <= 0:
            raise ValueError("run_time must exceed total residue delay")
        self.start_time = getattr(self, "start_time", 0.0)
        self.duration = duration
        self.begin, self.span = self.start_time + ranks * self.residue_delay, span
        self.track = p._appearance.copy()
        if self.kind == "color":
            self.track[self.ids, :4] = current_tints(p)[self.ids]
            self.track[self.ids, 4:8] = self.value
            self.track[self.ids, 8] = self.begin
            self.track[self.ids, 9] = self.span
            self.track[self.ids, 10] = self.easing == "smooth"
        else:
            self.track[self.ids, 12] = current_opacities(p)[self.ids]
            self.track[self.ids, 13] = self.value
            self.track[self.ids, 14] = self.begin
            self.track[self.ids, 15] = self.span
            self.track[self.ids, 11] = self.easing == "smooth"
        self.track.flags.writeable = False
        columns = np.arange(11) if self.kind == "color" else np.arange(11, 16)
        self._write_indices = np.ix_(self.ids, columns)
        self._other_mask = np.ones(self.track.shape, bool)
        self._other_mask[self._write_indices] = False
        self._combined = None

    def apply(self, alpha):
        # Separate tracks can run together without replacing one another's fields.
        p = self.protein
        other = p._appearance
        if self._combined is None or not np.array_equal(
            self._combined[self._other_mask], other[self._other_mask]
        ):
            data = other.copy()
            data[self._write_indices] = self.track[self._write_indices]
            data.flags.writeable = False
            self._combined = data
        p._appearance = self._combined
        clock = self.start_time + float(alpha) * self.duration
        if self.kind == "color":
            p._color_mix = clock
        else:
            p._opacity_mix = clock


class Colorize(_AppearanceAnimation):
    """Fade to a color across every representation; None restores the base palette."""

    kind = "color"
    channels = frozenset({"color"})

    def __init__(self, target, color, **kwargs):
        tint = np.r_[parse_color(color), 1] if color is not None else np.zeros(4)
        super().__init__(target, tint, **kwargs)


class SetOpacity(_AppearanceAnimation):
    """Animate atom/residue opacity, optionally staggered in N-to-C order."""

    kind = "opacity"
    channels = frozenset({"residue_opacity"})

    def __init__(self, target, opacity, **kwargs):
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("Opacity must be finite and in [0, 1]")
        super().__init__(target, float(opacity), **kwargs)


class RegionAnimate:
    def __init__(self, region):
        self.region = region

    def set_color(self, color, **kwargs):
        return Colorize(self.region, color, **kwargs)

    def set_opacity(self, opacity, **kwargs):
        return SetOpacity(self.region, opacity, **kwargs)
