"""Order-preserving Cα/C1′ correspondence via contact-map compatibility.

Objective: maximize cardinality under a pairwise soft-contact error bound, then
minimize squared contact error at that cardinality. Bounded search reports its
optimality status; candidate pruning is never disguised as a global proof.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


def morph_anchors(protein, ids):
    """Validate one polymer type and return its required Cα or C1′ atom indices."""
    residues = [protein.topology.residues[i] for i in ids]
    names = {r.morph_atom_name for r in residues}
    if len(names) != 1:
        raise ValueError("A morph correspondence must select one polymer type per endpoint")
    name = next(iter(names))
    if any(r.morph_atom < 0 for r in residues):
        raise ValueError(f"Each matched residue must have a {name} morph anchor")
    return np.array([r.morph_atom for r in residues], dtype=int), name


def backbone_selection(protein, chain=None):
    ids = np.array(
        [
            i
            for i, r in enumerate(protein.topology.residues)
            if r.morph_atom >= 0 and (chain is None or r.chain == chain)
        ],
        dtype=int,
    )
    if len(ids) < 3:
        raise ValueError(
            "Select a chain containing at least three morph anchors: C1' for DNA/RNA or CA for proteins"
        )
    chains = {protein.topology.residues[i].chain for i in ids}
    if len(chains) != 1:
        raise ValueError(
            "Contact matching requires one chain from each structure; set source_chain/target_chain"
        )
    indices, anchor = morph_anchors(protein, ids)
    return ids, protein.positions[indices].astype(float), next(iter(chains)), anchor


def ca_selection(protein, chain=None):
    """Legacy protein-only selection helper; nucleotide matching uses backbone_selection."""
    ids, xyz, chain, anchor = backbone_selection(protein, chain)
    if anchor != "CA":
        raise ValueError("ca_selection requires protein C-alpha atoms")
    return ids, xyz, chain


def contact_map(xyz, cutoff=8.0, softness=1.5):
    """Soft contacts: sigmoid((cutoff - anchor distance)/softness), diagonal zero."""
    if not np.isfinite(cutoff) or cutoff <= 0 or not np.isfinite(softness) or softness <= 0:
        raise ValueError("cutoff and softness must be finite and positive")
    d = cdist(xyz, xyz)
    c = 1 / (1 + np.exp(np.clip((d - cutoff) / softness, -50, 50)))
    np.fill_diagonal(c, 0)
    return c


def fit_transform(moving, reference):
    """Return a proper row-vector rotation R and translation t: moving @ R + t."""
    a, b = np.asarray(moving, float), np.asarray(reference, float)
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return r, bc - ac @ r


def _dp(scores):
    """Maximum-score monotone matching, with unmatched residues freely skippable."""
    n, m = scores.shape
    dp = np.zeros((n + 1, m + 1))
    trace = np.zeros((n + 1, m + 1), np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = dp[i - 1, j - 1] + scores[i - 1, j - 1]
            if diagonal > dp[i - 1, j] and diagonal > dp[i, j - 1]:
                dp[i, j], trace[i, j] = diagonal, 1
            elif dp[i - 1, j] >= dp[i, j - 1]:
                dp[i, j], trace[i, j] = dp[i - 1, j], 2
            else:
                dp[i, j], trace[i, j] = dp[i, j - 1], 3
    pairs, i, j = [], n, m
    while i and j:
        action = trace[i, j]
        if action == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == 2:
            i -= 1
        else:
            j -= 1
    return np.array(pairs[::-1], dtype=int).reshape(-1, 2)


def _error(ca, cb, pairs):
    if len(pairs) < 2:
        return 0.0, 0.0, 0.0
    a, b = pairs.T
    difference = ca[np.ix_(a, a)] - cb[np.ix_(b, b)]
    upper = difference[np.triu_indices(len(a), 1)]
    return float(np.sum(upper**2)), float(np.sqrt(np.mean(upper**2))), float(np.abs(upper).max())


def _filter(pairs, ca, cb, tolerance):
    """Feasible seed, removing high-conflict pairs. Final optimization is separate."""
    pairs = pairs.copy()
    if not len(pairs):
        return pairs
    a, b = pairs.T
    error = np.abs(ca[np.ix_(a, a)] - cb[np.ix_(b, b)])
    active = np.ones(len(pairs), bool)
    while True:
        bad = (error > tolerance + 1e-10) & active[:, None] & active[None, :]
        degree = bad.sum(1)
        if not degree.max():
            return pairs[active]
        score = degree + np.sum(error * bad, axis=1) / (len(pairs) + 1)
        active[int(np.argmax(score))] = False


def _seeds(x, y, ca, cb, tolerance):
    n, m = len(x), len(y)
    seeds, candidates = [], set()

    # Distance/contact fingerprints do not use residue sequence identity.
    def features(c, xyz):
        d = cdist(xyz, xyz)
        order = np.arange(len(c))
        before = order[:, None] > order[None, :]
        after = order[:, None] < order[None, :]
        offsets = []
        for k in (2, 3, 4, 6, 8):
            offsets.extend([d[order, np.minimum(order + k, len(c) - 1)], d[order, np.maximum(order - k, 0)]])
        return np.column_stack([c.sum(1), (c * before).sum(1), (c * after).sum(1), *offsets])

    fa, fb = features(ca, x), features(cb, y)
    scale = np.maximum(np.std(np.concatenate((fa, fb)), axis=0), 1.0)
    descriptor = cdist(fa / scale, fb / scale) / np.sqrt(fa.shape[1])
    mappings = [_dp(1.2 - descriptor)]
    # Broad length-scaled and offset seeds cover insertions and shifted domains.
    for offset in np.linspace(-m // 2, m // 2, 11).astype(int):
        a = np.arange(n)
        b = a + offset
        keep = (b >= 0) & (b < m)
        if keep.sum() >= 3:
            mappings.append(np.column_stack((a[keep], b[keep])))
    width = min(8, n, m)
    da, db = cdist(x, x), cdist(y, y)
    features_a = np.array([da[i : i + width, i : i + width].ravel() for i in range(n - width + 1)])
    features_b = np.array([db[j : j + width, j : j + width].ravel() for j in range(m - width + 1)])
    fragment_error = cdist(features_a, features_b) / width
    occupied = set()
    for flat in np.argsort(fragment_error, axis=None):
        i, j = np.unravel_index(flat, fragment_error.shape)
        bucket = (i // 8, j // 8)
        if bucket in occupied:
            continue
        occupied.add(bucket)
        mappings.append(np.column_stack((np.arange(i, i + width), np.arange(j, j + width))))
        if len(occupied) >= 18:
            break
    for mapping in mappings:
        if len(mapping) < 3:
            continue
        for _ in range(3):
            r, t = fit_transform(y[mapping[:, 1]], x[mapping[:, 0]])
            distance = cdist(x, y @ r + t)
            scores = 1 / (1 + (distance / 4.0) ** 2) - 0.12
            if len(mapping) > width:
                ia, ib = mapping.T
                # Contact disagreement against all current anchors refines each possible match.
                disagreement = np.mean(np.abs(ca[:, ia][:, None, :] - cb[:, ib][None, :, :]), axis=2)
                scores -= 1.5 * disagreement
            mapping = _dp(scores)
            if len(mapping) < 3:
                break
        if len(mapping) >= 3:
            seeds.append(_filter(mapping, ca, cb, tolerance))
            for i, j in mapping:
                for delta in (-1, 0, 1):
                    if 0 <= j + delta < m:
                        candidates.add((int(i), int(j + delta)))
    for i in range(n):
        for j in np.argsort(descriptor[i])[:8]:
            candidates.add((i, int(j)))
    return seeds, candidates, descriptor


class _Timeout(Exception):
    pass


class _CliqueSearch:
    """Bitset branch-and-bound with greedy coloring upper bounds (two-level objective)."""

    def __init__(self, pairs, ca, cb, tolerance, seeds, seconds):
        self.pairs, self.ca, self.cb = pairs, ca, cb
        self.seconds = seconds
        self.nodes = 0
        self.best = []
        self.best_error = float("inf")
        lookup = {tuple(pair): k for k, pair in enumerate(pairs)}
        for seed in seeds:
            ids = [lookup[tuple(p)] for p in seed if tuple(p) in lookup]
            self.consider(ids)
        # Build adjacency in blocks; never materialize an O((n*m)^2) float tensor.
        count = len(pairs)
        self.adj = []
        ai, bi = pairs.T
        for start in range(0, count, 128):
            aa, bb = ai[start : start + 128, None], bi[start : start + 128, None]
            ordered = ((aa < ai) & (bb < bi)) | ((aa > ai) & (bb > bi))
            error = np.abs(ca[aa, ai] - cb[bb, bi])
            compatible = ordered & (error <= tolerance + 1e-10)
            packed = np.packbits(compatible, axis=1, bitorder="little")
            self.adj.extend(int.from_bytes(row.tobytes(), "little") for row in packed)
        self.completed = False

    def consider(self, ids):
        error = _error(self.ca, self.cb, self.pairs[ids])[0] if len(ids) else 0.0
        if len(ids) > len(self.best) or (len(ids) == len(self.best) and error < self.best_error):
            self.best, self.best_error = list(ids), error

    def check_time(self):
        if time.perf_counter() >= self.deadline:
            raise _Timeout

    def color_sort(self, p):
        order, bounds, color = [], [], 0
        while p:
            self.check_time()
            color += 1
            q = p
            while q:
                bit = q & -q
                v = bit.bit_length() - 1
                order.append(v)
                bounds.append(color)
                p ^= bit
                q &= ~bit & ~self.adj[v]
        return order, bounds

    def expand(self, chosen, p, error):
        self.nodes += 1
        self.check_time()
        order, bounds = self.color_sort(p)
        for k in range(len(order) - 1, -1, -1):
            upper = len(chosen) + bounds[k]
            if upper < len(self.best) or (upper == len(self.best) and error >= self.best_error - 1e-14):
                return
            v = order[k]
            bit = 1 << v
            extension = chosen + [v]
            new_p = p & self.adj[v]
            increment = 0.0
            if chosen:
                a, b = self.pairs[v]
                aa, bb = self.pairs[chosen].T
                increment = float(np.sum((self.ca[a, aa] - self.cb[b, bb]) ** 2))
            new_error = error + increment
            if len(extension) > len(self.best) or (
                len(extension) == len(self.best) and new_error < self.best_error
            ):
                self.best, self.best_error = extension, new_error
            if new_p:
                self.expand(extension, new_p, new_error)
            p &= ~bit

    def solve(self):
        self.deadline = time.perf_counter() + self.seconds
        try:
            self.expand([], (1 << len(self.pairs)) - 1, 0.0)
            self.completed = True
        except _Timeout:
            pass
        return self.pairs[self.best][np.argsort(self.pairs[self.best, 0])]


@dataclass(frozen=True)
class ContactMatch:
    """Topology residue indices in chain order, and an auditable optimization report."""

    source_indices: np.ndarray
    target_indices: np.ndarray
    report: dict
    source_keys: tuple
    target_keys: tuple

    def __post_init__(self):
        for name in ("source_indices", "target_indices"):
            values = np.asarray(getattr(self, name))
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError("Residue index arrays must be one-dimensional integers")
            values = values.copy()
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @classmethod
    def from_pairs(cls, source, target, source_indices, target_indices):
        """Use explicit topology residue-index correspondences instead of automatic search."""
        a, b = np.asarray(source_indices), np.asarray(target_indices)
        if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) < 3:
            raise ValueError("Supply equal-length lists of at least three residue pairs")

        def keys(protein, indices):
            if (
                indices.dtype.kind not in "iu"
                or indices.min() < 0
                or indices.max() >= len(protein.topology.residues)
            ):
                raise ValueError("Invalid topology residue index")
            return tuple(
                (
                    protein.topology.residues[i].chain,
                    protein.topology.residues[i].resid,
                    protein.topology.residues[i].icode,
                    protein.topology.residues[i].name,
                )
                for i in indices
            )

        source_keys, target_keys = keys(source, a), keys(target, b)
        _, source_anchor = morph_anchors(source, a)
        _, target_anchor = morph_anchors(target, b)
        if source_anchor != target_anchor:
            raise ValueError("Source and target must use the same morph anchor (CA or C1')")
        return cls(
            a,
            b,
            {
                "selected_by": "user",
                "matched_count": len(a),
                "global_optimal": False,
                "source_anchor_atom": source_anchor,
                "target_anchor_atom": target_anchor,
            },
            source_keys,
            target_keys,
        )

    @classmethod
    def load(cls, path):
        """Load saved correspondence. BackboneMorph validates identities against the actual inputs."""
        data = json.loads(Path(path).read_text())
        a, b = (
            np.array(data.pop("source_residue_indices")),
            np.array(data.pop("target_residue_indices")),
        )
        ak, bk = (
            tuple(map(tuple, data.pop("source_residue_keys"))),
            tuple(map(tuple, data.pop("target_residue_keys"))),
        )
        return cls(a, b, data, ak, bk)

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    **self.report,
                    "source_residue_indices": self.source_indices.tolist(),
                    "target_residue_indices": self.target_indices.tolist(),
                    "source_residue_keys": self.source_keys,
                    "target_residue_keys": self.target_keys,
                },
                indent=2,
            )
            + "\n"
        )


def match_backbones(
    source,
    target,
    *,
    source_chain=None,
    target_chain=None,
    cutoff=8.0,
    softness=1.5,
    max_contact_error=0.30,
    search_seconds=8.0,
    max_candidates=6000,
):
    """Maximize a monotone matched subset under a per-pair contact discrepancy bound.

    Uses C1′ for DNA/RNA and Cα for proteins, in deposited chain order (normally
    5′ to 3′ or N to C). Residues missing the required atom remain unmatched.
    Set max_candidates=None to search all anchor pairings (limited to 30,000 candidate
    vertices). The report distinguishes full-space proofs from candidate-restricted
    or time-limited solutions. The cutoff/softness are ångströms; max_contact_error
    is in [0,1]. The time budget covers branch-and-bound, not seed/graph preparation.
    """
    if not np.isfinite(max_contact_error) or not 0 <= max_contact_error <= 1:
        raise ValueError("max_contact_error must be in [0, 1]")
    if not np.isfinite(search_seconds) or search_seconds <= 0:
        raise ValueError("search_seconds must be finite and positive")
    if max_candidates is not None and (not isinstance(max_candidates, int) or max_candidates < 3):
        raise ValueError("max_candidates must be None or an integer >= 3")
    started = time.perf_counter()
    source_ids, x, sc, source_anchor = backbone_selection(source, source_chain)
    target_ids, y, tc, target_anchor = backbone_selection(target, target_chain)
    if source_anchor != target_anchor:
        raise ValueError("Source and target must use the same morph anchor (CA or C1')")
    if min(len(x), len(y)) > 800:
        raise ValueError("Select chains/domains with at most 800 anchored residues in the shorter structure")
    ca, cb = contact_map(x, cutoff, softness), contact_map(y, cutoff, softness)
    if len(x) * len(y) > 30000 and max_candidates is None:
        raise ValueError(
            "Full candidate search exceeds 30,000 vertices; set max_candidates for bounded search"
        )
    seeds, candidates, descriptor = _seeds(x, y, ca, cb, max_contact_error)
    full = max_candidates is None or len(x) * len(y) <= max_candidates
    if full:
        pairs = np.array([(i, j) for i in range(len(x)) for j in range(len(y))], int)
    else:
        # Keep the best feasible seed intact if candidate budget truncation is needed.
        seeds.sort(key=lambda p: (-len(p), _error(ca, cb, p)[0]))
        required = {tuple(p) for p in seeds[0]} if seeds else set()
        if len(required) > max_candidates:
            raise ValueError("max_candidates is smaller than the best seed; increase it")
        ranked = sorted(candidates - required, key=lambda p: (descriptor[p], p))
        pairs = np.array(sorted(required | set(ranked[: max_candidates - len(required)])), int)
    solver = _CliqueSearch(pairs, ca, cb, max_contact_error, seeds, search_seconds)
    selected = solver.solve()
    if len(selected) < 3:
        raise ValueError("No compatible subset of at least three residues was found; relax max_contact_error")
    a, b = selected.T
    sse, rms, worst = _error(ca, cb, selected)
    r, t = fit_transform(y[b], x[a])
    anchor_rmsd = float(np.sqrt(np.mean(np.sum((y[b] @ r + t - x[a]) ** 2, axis=1))))
    cardinality_proved = (full and solver.completed) or len(a) == min(len(x), len(y))
    optimal = (full and solver.completed) or (cardinality_proved and sse < 1e-14)
    src_idx, dst_idx = source_ids[a], target_ids[b]

    def keys(protein, ids):
        return tuple(
            (
                protein.topology.residues[i].chain,
                protein.topology.residues[i].resid,
                protein.topology.residues[i].icode,
                protein.topology.residues[i].name,
            )
            for i in ids
        )

    report = dict(
        objective="maximize matched count under pairwise contact-error bound, then minimize sum squared error",
        source_chain=sc,
        target_chain=tc,
        source_anchor_atom=source_anchor,
        target_anchor_atom=target_anchor,
        source_anchor_count=len(x),
        target_anchor_count=len(y),
        matched_count=len(a),
        source_coverage=len(a) / len(x),
        target_coverage=len(b) / len(y),
        cutoff_angstrom=cutoff,
        softness_angstrom=softness,
        max_contact_error=max_contact_error,
        contact_rms_error=rms,
        contact_max_error=worst,
        aligned_anchor_rmsd_angstrom=anchor_rmsd,
        full_candidate_space=full,
        candidate_count=len(pairs),
        search_completed=solver.completed,
        cardinality_proved=cardinality_proved,
        global_optimal=optimal,
        cardinality_upper_bound=len(a) if cardinality_proved else min(len(x), len(y)),
        search_nodes=solver.nodes,
        seconds=time.perf_counter() - started,
        search_seconds=search_seconds,
    )
    if source_anchor == "CA":
        # Preserve report keys used by saved protein workflows.
        report.update(source_ca_count=len(x), target_ca_count=len(y), aligned_ca_rmsd_angstrom=anchor_rmsd)
    return ContactMatch(src_idx, dst_idx, report, keys(source, src_idx), keys(target, dst_idx))
