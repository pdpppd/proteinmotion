"""Contact-guided, staggered backbone morphing between molecular topologies.

``BackboneMorph`` morphs one structure into another when the two have different
atoms, for example two homologous proteins. ``matching.match_backbones`` pairs up
residues whose contact maps agree (a ``ContactMatch``). During the clip each matched
residue travels rigidly with its Cα (C1′ for nucleotides) from its place in the source
to its place in the destination, residue after residue along the chain. Both objects
are drawn: the source copy of a residue fades out while the destination copy fades in,
and unmatched residues crossfade in place. Most of the work happens on the GPU through
per-atom motion and opacity controls (``Protein._controls``) set up in ``bind``.
"""

import numpy as np

from .animation import Animation
from .matching import ContactMatch, fit_transform, match_backbones, morph_anchors
from .rates import evaluate, linear, resolve, smooth
from .structure import coordinates


class BackboneMorph(Animation):
    """Move matched Cα/C1′ anchors in chain order; crossfade unmatched residues.

    residue_delay is seconds between successive matched residue starts.
    run_time comes from Scene.play; each residue moves for run_time-(K-1)*delay.
    fade_out and fade_in are normalized intervals for unmatched residues.
    C1′ anchors are used for DNA/RNA, Cα for proteins. Delays follow deposited
    residue order, normally 5′ to 3′ for nucleotides and N to C for amino acids.
    align=True rigidly aligns the destination's matched anchors to the source first.
    Every atom in a residue translates with its anchor; source/target geometry
    crossfades. Base and side-chain atoms are not individually mapped.
    This animation owns both objects' geometry/transform/opacity channels. Animate
    the camera concurrently, rather than independently rotating either endpoint.
    """

    channels = frozenset({"geometry", "transform", "opacity", "controls"})
    # Per-residue timing is baked into the GPU controls as fractions of a linear clip.
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
        easing=None,
        delay_seconds=None,
        stagger_fraction=None,
        **match_options,
    ):
        super().__init__(source, rate_func=linear)
        if source is destination:
            raise ValueError("BackboneMorph needs two different Protein objects")
        # delay_seconds and easing are aliases of residue_delay and motion_easing.
        residue_delay = residue_delay if delay_seconds is None else delay_seconds
        if stagger_fraction is not None and (
            not np.isfinite(stagger_fraction) or not 0 <= stagger_fraction < 1
        ):
            raise ValueError("stagger_fraction must be in [0, 1)")
        self.stagger_fraction = stagger_fraction
        if not np.isfinite(residue_delay) or residue_delay < 0:
            raise ValueError("residue_delay must be finite and nonnegative")
        for interval in (fade_out, fade_in):
            if len(interval) != 2 or not 0 <= interval[0] < interval[1] <= 1:
                raise ValueError("Fade intervals must satisfy 0 <= start < end <= 1")
        motion_easing = resolve(motion_easing if easing is None else easing)
        self.destination, self.match = destination, match
        self.residue_delay, self.fade_out, self.fade_in = residue_delay, fade_out, fade_in
        self.align, self.motion_easing, self.match_options = align, motion_easing, match_options

    @property
    def result(self):
        """The destination object to select, label and animate after this morph."""
        return self.destination

    def _set_easing(self, easing):
        self.motion_easing = resolve(easing)

    @property
    def targets(self):
        return (self.target, self.destination)

    def bind(self):
        super().bind()
        source, dest = self.target, self.destination
        # Matched residue indices, increasing along one chain in each structure, and the
        # Cα or C1′ atom that carries each of them.
        source_ids, target_ids = self._matched_residues(source, dest)
        source_anchors, target_anchors = self._anchor_atoms(source, dest, source_ids, target_ids)
        self.source_anchor_indices, self.target_anchor_indices = source_anchors, target_anchors
        # Timing: matched residue k starts k × residue_delay seconds into the clip, and all
        # of them move for the same span, so the last one arrives as the clip ends.
        self.run_time = getattr(self, "run_time", 1.0)
        if self.stagger_fraction is not None:
            self.residue_delay = self.run_time * self.stagger_fraction / max(1, len(source_ids) - 1)
        span = self.run_time - (len(source_ids) - 1) * self.residue_delay
        if span <= 0:
            raise ValueError(
                f"run_time must exceed {(len(source_ids) - 1) * self.residue_delay:.3f}s for this "
                "residue_delay; reduce delay or increase run_time"
            )
        self.move_time = span
        # Work in world space, since the two objects may have different transforms.
        self.source_start = coordinates(source.positions)
        destination_original = coordinates(dest.positions)
        source_model, dest_model = source.model_matrix, dest.model_matrix
        source_xyz = self.source_start @ source_model[:3, :3].T + source_model[:3, 3]
        dest_xyz = destination_original @ dest_model[:3, :3].T + dest_model[:3, 3]
        if self.align:
            # Superpose the destination's matched anchors on the source's (Kabsch), so the
            # morph shows the change in shape rather than a rigid move across the scene.
            rotation, translation = fit_transform(dest_xyz[target_anchors], source_xyz[source_anchors])
            dest_xyz = dest_xyz @ rotation + translation
        self.target_world = dest_xyz[target_anchors].copy()
        self.source_world = source_xyz[source_anchors].copy()
        source_end_world, destination_start_world = source_xyz.copy(), dest_xyz.copy()
        source_residue = np.array([atom.residue_index for atom in source.topology.atoms])
        dest_residue = np.array([atom.residue_index for atom in dest.topology.atoms])
        # Unmatched residues stay in place and crossfade over fade_out and fade_in.
        self.source_controls = self._controls(source, 1, 0, self.fade_out)
        self.destination_controls = self._controls(dest, 0, 1, self.fade_in)
        # Shader easing mode: 2 is quintic smooth, 1 linear. apply() handles other rate
        # functions on the CPU.
        easing = 2 if self.motion_easing is smooth else 1
        for rank, (si, ti, satom, tatom) in enumerate(
            zip(source_ids, target_ids, source_anchors, target_anchors)
        ):
            s_mask, d_mask = source_residue == si, dest_residue == ti
            delta = dest_xyz[tatom] - source_xyz[satom]
            # Translate whole residues with Cα/C1′, retaining endpoint internal geometry.
            # The source copy ends on the destination anchor; the destination copy starts
            # on the source anchor.
            source_end_world[s_mask] += delta
            destination_start_world[d_mask] -= delta
            begin = rank * self.residue_delay / self.run_time
            length = span / self.run_time
            # Both copies move over the same window of the clip, while the source copy
            # fades out and the destination copy fades in.
            self.source_controls[s_mask, :4] = [begin, length, easing, 0]
            self.destination_controls[d_mask, :4] = [begin, length, easing, 0]
            self.source_controls[s_mask, 4:] = [1, 0, begin, length]
            self.destination_controls[d_mask, 4:] = [0, 1, begin, length]
        # Back to each object's local frame, where its keyframes live.
        inv_s, inv_d = np.linalg.inv(source_model[:3, :3]), np.linalg.inv(dest_model[:3, :3])
        self.source_end = coordinates((source_end_world - source_model[:3, 3]) @ inv_s.T)
        self.destination_start = coordinates((destination_start_world - dest_model[:3, 3]) @ inv_d.T)
        self.destination_end = coordinates((dest_xyz - dest_model[:3, 3]) @ inv_d.T)
        for c in (self.source_controls, self.destination_controls):
            c.flags.writeable = False
        self.source_opacity, self.destination_opacity = source.opacity, dest.opacity
        self.destination_final_controls = dest._controls
        self.delay_schedule = np.arange(len(source_ids)) * self.residue_delay

    def _matched_residues(self, source, dest):
        """Check the residue correspondence (computing it if needed) and return its indices."""
        if self.match is None:
            self.match = match_backbones(source, dest, **self.match_options)
        if not isinstance(self.match, ContactMatch):
            raise TypeError("match must be a ContactMatch")
        a, b = np.asarray(self.match.source_indices), np.asarray(self.match.target_indices)
        if len(a) < 3 or len(a) != len(b) or a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
            raise ValueError("Match must contain at least three integer residue pairs")
        if (np.diff(a) <= 0).any() or (np.diff(b) <= 0).any():
            raise ValueError(
                "Matched residue indices must increase strictly in chain order in both structures"
            )
        for protein, ids, keys in [(source, a, self.match.source_keys), (dest, b, self.match.target_keys)]:
            if ids.min() < 0 or ids.max() >= len(protein.topology.residues):
                raise ValueError("Matched residue index is out of bounds")
            # A saved match records residue identities; they must name these proteins' residues.
            actual = tuple(
                (
                    protein.topology.residues[i].chain,
                    protein.topology.residues[i].resid,
                    protein.topology.residues[i].icode,
                    protein.topology.residues[i].name,
                )
                for i in ids
            )
            if keys and actual != keys:
                raise ValueError("Match residue identities do not belong to these proteins")
            if len({protein.topology.residues[i].chain for i in ids}) != 1:
                raise ValueError("A BackboneMorph correspondence must describe one chain in each structure")
        return a, b

    def _anchor_atoms(self, source, dest, source_ids, target_ids):
        """Atom indices of the matched residues' anchors (Cα or C1′) in each structure."""
        ac, source_anchor = morph_anchors(source, source_ids)
        bc, target_anchor = morph_anchors(dest, target_ids)
        if source_anchor != target_anchor:
            raise ValueError("Source and target must use the same morph anchor (CA or C1')")
        for endpoint, anchor in (("source", source_anchor), ("target", target_anchor)):
            stored = self.match.report.get(f"{endpoint}_anchor_atom", anchor)
            if stored != anchor:
                raise ValueError(f"Saved {endpoint} morph anchor {stored!r} does not match {anchor!r}")
        return ac, bc

    @staticmethod
    def _controls(protein, start, end, fade):
        """Per-atom controls that leave atoms in place and fade them from start to end.

        The (atoms, 8) float32 layout is that of ``Protein._controls`` (``struct Control``
        in molecule.wgsl): columns 0:4 are the motion start and length as fractions of
        the clip, the easing mode (0 follows the global tween, 1 linear, 2 smooth) and an
        unused slot; columns 4:8 are the opacity at the start and end and the start and
        length of the fade. ``fade`` is a (start, end) interval of the clip.
        """
        controls = np.zeros((len(protein.topology.atoms), 8), np.float32)
        controls[:, 1] = 1
        controls[:, 4:] = [start, end, fade[0], fade[1] - fade[0]]
        return controls

    def apply(self, alpha):
        source, dest = self.target, self.destination
        # The keyframes span the whole clip; each atom's controls pick its window within it.
        source._pair(self.source_start, self.source_end, alpha)
        dest._pair(self.destination_start, self.destination_end, alpha)
        source._controls = self.source_controls
        # Once the morph is over, the destination gets its own controls back.
        dest._controls = self.destination_final_controls if alpha >= 1 else self.destination_controls
        if self.motion_easing not in (smooth, linear) and 0 < alpha < 1:
            # The shader eases only linearly or with smooth. For any other rate function,
            # evaluate each atom's progress here and give the GPU finished coordinates.
            for protein, start, end, controls in (
                (source, self.source_start, self.source_end, self.source_controls),
                (dest, self.destination_start, self.destination_end, self.destination_controls),
            ):
                local = np.clip((alpha - controls[:, 0]) / controls[:, 1], 0, 1)
                fraction = np.array([evaluate(self.motion_easing, t) for t in local])[:, None]
                points = coordinates((1 - fraction) * start + fraction * end)
                protein._pair(points, points, alpha)
                # The motion is already in the coordinates; keep only the opacity timing.
                custom = controls.copy()
                custom[:, 2] = 0
                custom.flags.writeable = False
                protein._controls = custom
        # Both objects show during the morph; the source hides at its end, and the
        # destination before its start.
        source.opacity = self.source_opacity if alpha < 1 else 0.0
        dest.opacity = self.destination_opacity if alpha > 0 else 0.0
