"""Contact-guided, staggered backbone morphing between different protein topologies."""

import numpy as np

from .animation import Animation
from .matching import ContactMatch, fit_transform, match_backbones
from .rates import linear
from .structure import coordinates


class BackboneMorph(Animation):
    """Move selected CAs N-to-C; crossfade unmatched source/destination residues.

    residue_delay is seconds between successive matched residue starts.
    run_time comes from Scene.play; each residue moves for run_time-(K-1)*delay.
    fade_out and fade_in are normalized intervals for unmatched residues.
    align=True rigidly aligns the destination's matched CAs to the source first.
    In ball-and-stick mode, every atom in a residue translates with its CA and
    source/target atom sets crossfade. Side-chain atoms are not individually mapped.
    This animation owns both objects' geometry/transform/opacity channels. Animate
    the camera concurrently, rather than independently rotating either endpoint.
    """

    channels = frozenset({"geometry", "transform", "opacity", "controls"})
    requires_linear_timeline = True

    def __init__(
        self,
        source,
        destination,
        *,
        match=None,
        residue_delay=0.008,
        fade_out=(0.0, 0.35),
        fade_in=(0.65, 1.0),
        align=True,
        motion_easing="smooth",
        **match_options,
    ):
        super().__init__(source, rate_func=linear)
        if source is destination:
            raise ValueError("BackboneMorph needs two different Protein objects")
        if not np.isfinite(residue_delay) or residue_delay < 0:
            raise ValueError("residue_delay must be finite and nonnegative")
        for interval in (fade_out, fade_in):
            if len(interval) != 2 or not 0 <= interval[0] < interval[1] <= 1:
                raise ValueError("Fade intervals must satisfy 0 <= start < end <= 1")
        if motion_easing not in ("smooth", "linear"):
            raise ValueError("motion_easing must be 'smooth' or 'linear'")
        self.destination, self.match = destination, match
        self.residue_delay, self.fade_out, self.fade_in = residue_delay, fade_out, fade_in
        self.align, self.motion_easing, self.match_options = align, motion_easing, match_options

    @property
    def targets(self):
        return (self.target, self.destination)

    def bind(self):
        super().bind()
        source, dest = self.target, self.destination
        if self.match is None:
            self.match = match_backbones(source, dest, **self.match_options)
        if not isinstance(self.match, ContactMatch):
            raise TypeError("match must be a ContactMatch")
        a, b = np.asarray(self.match.source_indices), np.asarray(self.match.target_indices)
        if len(a) < 3 or len(a) != len(b) or a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
            raise ValueError("Match must contain at least three integer residue pairs")
        if (np.diff(a) <= 0).any() or (np.diff(b) <= 0).any():
            raise ValueError("Matched residue indices must increase strictly N-to-C in both proteins")
        for p, ids, keys in [(source, a, self.match.source_keys), (dest, b, self.match.target_keys)]:
            if ids.min() < 0 or ids.max() >= len(p.topology.residues):
                raise ValueError("Matched residue index is out of bounds")
            actual = tuple(
                (
                    p.topology.residues[i].chain,
                    p.topology.residues[i].resid,
                    p.topology.residues[i].icode,
                    p.topology.residues[i].name,
                )
                for i in ids
            )
            if keys and actual != keys:
                raise ValueError("Match residue identities do not belong to these proteins")
            if any(p.topology.residues[i].ca < 0 for i in ids):
                raise ValueError("Each matched residue must have a C-alpha")
            if len({p.topology.residues[i].chain for i in ids}) != 1:
                raise ValueError("A BackboneMorph correspondence must describe one chain in each protein")
        self.run_time = getattr(self, "run_time", 1.0)
        span = self.run_time - (len(a) - 1) * self.residue_delay
        if span <= 0:
            raise ValueError(
                f"run_time must exceed {(len(a) - 1) * self.residue_delay:.3f}s for this residue_delay; "
                "reduce delay or increase run_time"
            )
        self.move_time = span
        self.source_start = coordinates(source.positions)
        destination_original = coordinates(dest.positions)
        sm, dm = source.model_matrix, dest.model_matrix
        sw = self.source_start @ sm[:3, :3].T + sm[:3, 3]
        tw = destination_original @ dm[:3, :3].T + dm[:3, 3]
        ac = np.array([source.topology.residues[i].ca for i in a])
        bc = np.array([dest.topology.residues[i].ca for i in b])
        if self.align:
            rotation, translation = fit_transform(tw[bc], sw[ac])
            tw = tw @ rotation + translation
        self.target_world = tw[bc].copy()
        self.source_world = sw[ac].copy()
        source_end_world, destination_start_world = sw.copy(), tw.copy()
        sr = np.array([atom.residue_index for atom in source.topology.atoms])
        dr = np.array([atom.residue_index for atom in dest.topology.atoms])
        self.source_controls = self._controls(source, 1, 0, self.fade_out)
        self.destination_controls = self._controls(dest, 0, 1, self.fade_in)
        easing = 2 if self.motion_easing == "smooth" else 1
        for rank, (si, ti, satom, tatom) in enumerate(zip(a, b, ac, bc)):
            s_mask, d_mask = sr == si, dr == ti
            delta = tw[tatom] - sw[satom]
            # Translate whole residues with their CA, retaining each endpoint's internal geometry.
            source_end_world[s_mask] += delta
            destination_start_world[d_mask] -= delta
            begin = rank * self.residue_delay / self.run_time
            length = span / self.run_time
            self.source_controls[s_mask, :4] = [begin, length, easing, 0]
            self.destination_controls[d_mask, :4] = [begin, length, easing, 0]
            self.source_controls[s_mask, 4:] = [1, 0, begin, length]
            self.destination_controls[d_mask, 4:] = [0, 1, begin, length]
        inv_s, inv_d = np.linalg.inv(sm[:3, :3]), np.linalg.inv(dm[:3, :3])
        self.source_end = coordinates((source_end_world - sm[:3, 3]) @ inv_s.T)
        self.destination_start = coordinates((destination_start_world - dm[:3, 3]) @ inv_d.T)
        self.destination_end = coordinates((tw - dm[:3, 3]) @ inv_d.T)
        for c in (self.source_controls, self.destination_controls):
            c.flags.writeable = False
        self.source_opacity, self.destination_opacity = source.opacity, dest.opacity
        self.destination_final_controls = dest._controls
        self.delay_schedule = np.arange(len(a)) * self.residue_delay

    @staticmethod
    def _controls(protein, start, end, fade):
        controls = np.zeros((len(protein.topology.atoms), 8), np.float32)
        controls[:, 1] = 1
        controls[:, 4:] = [start, end, fade[0], fade[1] - fade[0]]
        return controls

    def apply(self, alpha):
        source, dest = self.target, self.destination
        source._pair(self.source_start, self.source_end, alpha)
        dest._pair(self.destination_start, self.destination_end, alpha)
        source._controls = self.source_controls
        dest._controls = self.destination_final_controls if alpha >= 1 else self.destination_controls
        source.opacity = self.source_opacity if alpha < 1 else 0.0
        dest.opacity = self.destination_opacity if alpha > 0 else 0.0
