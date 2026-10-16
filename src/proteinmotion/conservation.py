"""Per-residue sequence conservation from a multiple sequence alignment.

``ResidueValues.conservation()`` (properties.py) calls ``residue_conservation``. It reads
the alignment (FASTA, A3M, Stockholm or Clustal), scores every column, aligns the query
row to each matching protein chain of the structure, and returns one score per topology
residue. The score is the Shannon-entropy conservation 1 − H/ln 20 of the column's amino
acids, with Henikoff position-based sequence weights, multiplied by the weighted fraction
of sequences that have a residue rather than a gap there.
"""

from pathlib import Path

import gemmi
import numpy as np

# One-letter codes of the 20 standard amino acids. Any other letter (X, B, Z, U) counts as a gap.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _interleaved_rows(lines):
    """(name, sequence) rows from "name sequence ..." lines.

    Long alignments come in blocks, so a name appears once per block; its pieces are joined
    in file order, and rows keep the order in which names first appear.
    """
    order, chunks = [], {}
    for line in lines:
        name, sequence = line.split()[:2]
        if name not in chunks:
            order.append(name)
            chunks[name] = []
        chunks[name].append(sequence)
    return [(name, "".join(chunks[name])) for name in order]


def _fasta_rows(lines):
    """(name, sequence) rows from FASTA or A3M lines; an empty header is named seqN."""
    rows = []
    name, parts = None, []
    # The extra ">" at the end flushes the last record.
    for line in lines + [">"]:
        if line.startswith(">"):
            if name is not None:
                rows.append((name, "".join(parts)))
            name, parts = line[1:].strip() or f"seq{len(rows)}", []
        # "#" lines are comments, such as the length line at the top of ColabFold A3M files.
        elif not line.startswith("#"):
            parts.append(line.strip())
    return rows


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
        # A string names a file when that file exists; otherwise it is the alignment text.
        text = Path(source).read_text() if Path(str(source)).is_file() else str(source)
        lines = text.splitlines()
        first = next((line for line in lines if line.strip()), "")
        if first.startswith("# STOCKHOLM"):
            # Stockholm: "#" lines are the header and markup (#=GF, #=GS, #=GR, #=GC) and
            # "//" is the terminator; every other line is "name sequence".
            rows = _interleaved_rows(
                line
                for line in lines
                if line.strip() and not line.startswith("#") and not line.startswith("//")
            )
        elif first.upper().startswith("CLUSTAL"):
            # Clustal: after the header line, skip blank lines and the indented consensus
            # line of "*", ":" and "." marks. Rows are "name sequence [running count]".
            rows = _interleaved_rows(
                line
                for line in lines[1:]
                if line.strip() and not line.startswith(" ") and not line.startswith("\t")
            )
        else:
            rows = _fasta_rows(lines)
    rows = [(name, sequence) for name, sequence in rows if sequence]
    if len({len(sequence) for _, sequence in rows}) > 1:
        # A3M: lowercase letters and dots are insertions relative to the query's columns.
        rows = [
            (name, "".join(c for c in sequence if not c.islower() and c != ".")) for name, sequence in rows
        ]
    # Uppercase, and write "." gaps (Stockholm and A3M use them in insert columns) as "-".
    # When the rows already had equal lengths, insert columns stay and score like any other.
    rows = [(name, sequence.upper().replace(".", "-")) for name, sequence in rows]
    if not rows:
        raise ValueError("The alignment contains no sequences")
    width = len(rows[0][1])
    if any(len(sequence) != width for _, sequence in rows):
        raise ValueError("Aligned sequences must all have the same length after removing insertions")
    return rows


def column_conservation(sequences, *, weighting=True):
    """Conservation in [0, 1] for each alignment column: (1 − H/ln 20) × (1 − gap fraction)."""
    # codes[sequence, column]: 0–19 for the amino acids in AMINO_ACIDS order, −1 for a gap.
    letters = np.array([list(sequence) for sequence in sequences])
    codes = np.full(letters.shape, -1, dtype=int)
    for k, letter in enumerate(AMINO_ACIDS):
        codes[letters == letter] = k
    count = len(sequences)
    if weighting:
        # Henikoff & Henikoff position-based weights (J. Mol. Biol. 243, 574–578, 1994). In
        # each column a sequence gains 1 / (r × s), where r is the number of distinct residue
        # types in the column and s the number of sequences sharing this sequence's residue.
        # Sequences with rare residues weigh more, and near-duplicates share their weight.
        weights = np.zeros(count)
        for column in codes.T:
            kinds = column[column >= 0]
            if not len(kinds):
                continue
            counts = np.bincount(kinds, minlength=20)
            types = np.count_nonzero(counts)
            weights[column >= 0] += 1 / (types * counts[column[column >= 0]])
        # Normalize to sum 1. An alignment of gaps alone falls back to equal weights.
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(count, 1 / count)
    else:
        weights = np.full(count, 1 / count)
    scores = np.zeros(codes.shape[1])
    for c, column in enumerate(codes.T):
        mask = column >= 0
        # Total weight of the sequences that have a residue in this column.
        filled = weights[mask].sum()
        if filled <= 0:
            continue
        # Weighted residue frequencies among those sequences. Shannon entropy H (in nats) is
        # 0 for an invariant column and ln 20 when all 20 amino acids are equally common.
        frequencies = np.bincount(column[mask], weights=weights[mask], minlength=20) / filled
        frequencies = frequencies[frequencies > 0]
        entropy = -np.sum(frequencies * np.log(frequencies))
        # filled / weights.sum() is the weighted fraction of sequences without a gap here.
        scores[c] = (1 - entropy / np.log(20)) * filled / weights.sum()
    return np.clip(scores, 0, 1)


def _identity(a, b):
    """(identity in [0, 1], gemmi AlignmentResult) for a global alignment of two sequences.

    Identity counts matched positions relative to the shorter sequence.
    """
    result = gemmi.align_string_sequences(list(a), list(b), [])
    return result.calculate_identity() / 100, result


def _mapping(query, target):
    """Index into ``query`` for each position of ``target`` (−1 where unaligned)."""
    _, result = _identity(query, target)
    # Walk the two gapped rows together, counting residues in each. Where both rows have a
    # residue, target position ti pairs with query position qi.
    q, t = result.add_gaps(query, 1), result.add_gaps(target, 2)
    mapping, qi, ti = np.full(len(target), -1), 0, 0
    for a, b in zip(q, t):
        if a != "-" and b != "-":
            mapping[ti] = qi
        qi += a != "-"
        ti += b != "-"
    return mapping


def _query_index(rows, query, sequences):
    """Row index of the alignment's query: ``query`` as an index or name, or found by identity.

    ``sequences`` maps chain IDs to the structure's one-letter sequences.
    """
    if query is None:
        reference = max(sequences.values(), key=len)
        # The first sequence is usually the query (A3M, ColabFold); otherwise pick the closest.
        # Only the first 500 rows are compared, which bounds the cost for deep alignments.
        candidates = range(min(len(rows), 500))
        scores = [_identity(rows[k][1].replace("-", ""), reference)[0] for k in candidates]
        return int(np.argmax(scores)) if scores[0] < 0.9 else 0
    if isinstance(query, str):
        names = [name for name, _ in rows]
        if query not in names:
            raise ValueError(f"No sequence named {query!r} in the alignment")
        return names.index(query)
    return query


def residue_conservation(protein, alignment, *, chain=None, query=None, weighting=True):
    """One conservation value per topology residue; NaN where the alignment has no column."""
    rows = read_alignment(alignment)
    topology = protein.topology
    # Amino-acid residues (those with a Cα) of the requested chains, by chain ID.
    chains = {}
    for ri, residue in enumerate(topology.residues):
        wanted = chain is None or residue.chain in ([chain] if isinstance(chain, str) else chain)
        if residue.ca >= 0 and not residue.is_nucleic and wanted:
            chains.setdefault(residue.chain, []).append(ri)
    if not chains:
        raise ValueError("The structure has no amino-acid chain to map the alignment onto")

    def letters(indices):
        # One-letter codes from gemmi's residue table. Modified residues have their parent's
        # code in lower case, so upper() maps MSE to M; unknown names become X.
        codes = [gemmi.find_tabulated_residue(topology.residues[i].name) for i in indices]
        return "".join((c.one_letter_code.upper() if c is not None else "X") or "X" for c in codes)

    sequences = {name: letters(ids) for name, ids in chains.items()}
    query = _query_index(rows, query, sequences)
    # columns[k] is the alignment column of the query's k-th residue.
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
        # 30% identity is a common lower bound for a reliable sequence match.
        used = [name for name, value in identity.items() if value >= 0.9]
        best = max(identity, key=identity.get)
        used = used or ([best] if identity[best] >= 0.3 else [])
    if not used:
        raise ValueError("The alignment's query does not match any chain; pass chain= or query=")
    # Align the query to each chain, then give each residue the score of its query column.
    for name in used:
        mapping = _mapping(ungapped, sequences[name])
        for ri, position in zip(chains[name], mapping):
            if position >= 0:
                values[ri] = scores[columns[position]]
    return values
