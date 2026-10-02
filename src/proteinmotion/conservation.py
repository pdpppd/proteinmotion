"""Per-residue sequence conservation from a multiple sequence alignment."""

from pathlib import Path

import gemmi
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_alignment(source):
    """[(name, aligned sequence)] from FASTA, A3M, Stockholm or Clustal text or a file path.

    When rows differ in length (A3M), insertions (lowercase letters and ".") are
    removed so every row has the query's columns. A list of sequences or
    (name, sequence) pairs is also accepted.
    """
    if not isinstance(source, (str, Path)):
        rows = [
            (f"seq{i}", s) if isinstance(s, str) else (str(s[0]), str(s[1])) for i, s in enumerate(source)
        ]
    else:
        text = Path(source).read_text() if Path(str(source)).is_file() else str(source)
        lines = text.splitlines()
        first = next((line for line in lines if line.strip()), "")
        rows = []
        if first.startswith("# STOCKHOLM"):
            order, chunks = [], {}
            for line in lines:
                if not line.strip() or line.startswith("#") or line.startswith("//"):
                    continue
                name, sequence = line.split()[:2]
                if name not in chunks:
                    order.append(name)
                    chunks[name] = []
                chunks[name].append(sequence)
            rows = [(name, "".join(chunks[name])) for name in order]
        elif first.upper().startswith("CLUSTAL"):
            order, chunks = [], {}
            for line in lines[1:]:
                if not line.strip() or line.startswith(" ") or line.startswith("\t"):
                    continue
                name, sequence = line.split()[:2]
                if name not in chunks:
                    order.append(name)
                    chunks[name] = []
                chunks[name].append(sequence)
            rows = [(name, "".join(chunks[name])) for name in order]
        else:
            name, parts = None, []
            for line in lines + [">"]:
                if line.startswith(">"):
                    if name is not None:
                        rows.append((name, "".join(parts)))
                    name, parts = line[1:].strip() or f"seq{len(rows)}", []
                elif not line.startswith("#"):
                    parts.append(line.strip())
    rows = [(name, sequence) for name, sequence in rows if sequence]
    if len({len(sequence) for _, sequence in rows}) > 1:
        # A3M: lowercase letters and dots are insertions relative to the query's columns.
        rows = [
            (name, "".join(c for c in sequence if not c.islower() and c != ".")) for name, sequence in rows
        ]
    rows = [(name, sequence.upper().replace(".", "-")) for name, sequence in rows]
    if not rows:
        raise ValueError("The alignment contains no sequences")
    width = len(rows[0][1])
    if any(len(sequence) != width for _, sequence in rows):
        raise ValueError("Aligned sequences must all have the same length after removing insertions")
    return rows


def column_conservation(sequences, *, weighting=True):
    """Conservation in [0, 1] for each alignment column: (1 − H/ln 20) × (1 − gap fraction)."""
    letters = np.array([list(sequence) for sequence in sequences])
    codes = np.full(letters.shape, -1, dtype=int)
    for k, letter in enumerate(AMINO_ACIDS):
        codes[letters == letter] = k
    count = len(sequences)
    if weighting:
        # Henikoff & Henikoff position-based weights: rare residues give a sequence weight.
        weights = np.zeros(count)
        for column in codes.T:
            kinds = column[column >= 0]
            if not len(kinds):
                continue
            counts = np.bincount(kinds, minlength=20)
            types = np.count_nonzero(counts)
            weights[column >= 0] += 1 / (types * counts[column[column >= 0]])
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(count, 1 / count)
    else:
        weights = np.full(count, 1 / count)
    scores = np.zeros(codes.shape[1])
    for c, column in enumerate(codes.T):
        mask = column >= 0
        filled = weights[mask].sum()
        if filled <= 0:
            continue
        frequencies = np.bincount(column[mask], weights=weights[mask], minlength=20) / filled
        frequencies = frequencies[frequencies > 0]
        entropy = -np.sum(frequencies * np.log(frequencies))
        scores[c] = (1 - entropy / np.log(20)) * filled / weights.sum()
    return np.clip(scores, 0, 1)


def _identity(a, b):
    result = gemmi.align_string_sequences(list(a), list(b), [])
    return result.calculate_identity() / 100, result


def _mapping(query, target):
    """Index into ``query`` for each position of ``target`` (−1 where unaligned)."""
    _, result = _identity(query, target)
    q, t = result.add_gaps(query, 1), result.add_gaps(target, 2)
    mapping, qi, ti = np.full(len(target), -1), 0, 0
    for a, b in zip(q, t):
        if a != "-" and b != "-":
            mapping[ti] = qi
        qi += a != "-"
        ti += b != "-"
    return mapping


def residue_conservation(protein, alignment, *, chain=None, query=None, weighting=True):
    """One conservation value per topology residue; NaN where the alignment has no column."""
    rows = read_alignment(alignment)
    topology = protein.topology
    chains = {}
    for ri, residue in enumerate(topology.residues):
        wanted = chain is None or residue.chain in ([chain] if isinstance(chain, str) else chain)
        if residue.ca >= 0 and not residue.is_nucleic and wanted:
            chains.setdefault(residue.chain, []).append(ri)
    if not chains:
        raise ValueError("The structure has no amino-acid chain to map the alignment onto")

    def letters(indices):
        codes = [gemmi.find_tabulated_residue(topology.residues[i].name) for i in indices]
        return "".join((c.one_letter_code.upper() if c is not None else "X") or "X" for c in codes)

    sequences = {name: letters(ids) for name, ids in chains.items()}
    if query is None:
        reference = max(sequences.values(), key=len)
        # The first sequence is usually the query (A3M, ColabFold); otherwise pick the closest.
        candidates = range(min(len(rows), 500))
        scores = [_identity(rows[k][1].replace("-", ""), reference)[0] for k in candidates]
        query = int(np.argmax(scores)) if scores[0] < 0.9 else 0
    elif isinstance(query, str):
        names = [name for name, _ in rows]
        if query not in names:
            raise ValueError(f"No sequence named {query!r} in the alignment")
        query = names.index(query)
    aligned = rows[query][1]
    columns = np.array([k for k, c in enumerate(aligned) if c != "-"])
    ungapped = aligned.replace("-", "")
    scores = column_conservation([sequence for _, sequence in rows], weighting=weighting)
    values = np.full(len(topology.residues), np.nan)
    identity = {name: _identity(ungapped, sequence)[0] for name, sequence in sequences.items()}
    if chain is not None:
        used = list(sequences)
    else:
        # Every chain the query matches closely (homo-oligomers), else the closest homolog.
        used = [name for name, value in identity.items() if value >= 0.9]
        best = max(identity, key=identity.get)
        used = used or ([best] if identity[best] >= 0.3 else [])
    if not used:
        raise ValueError("The alignment's query does not match any chain; pass chain= or query=")
    for name in used:
        mapping = _mapping(ungapped, sequences[name])
        for ri, position in zip(chains[name], mapping):
            if position >= 0:
                values[ri] = scores[columns[position]]
    return values
