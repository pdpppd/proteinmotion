"""AlphaFold confidence files: predicted aligned error (PAE) and pLDDT.

Readers for the per-residue pLDDT and the PAE matrix written by AlphaFold DB, ColabFold and
AlphaFold 3. ``ResidueValues.plddt()`` (properties.py) calls ``read_plddt``, and
``Heatmap.pae()`` (plots.py) calls ``pae_for`` to get a matrix ordered like the protein's
residues. Residues are matched to the structure by position along the polymer, by residue
number, or for AlphaFold 3 by (chain, residue number) per token.
"""

import json
from pathlib import Path

import numpy as np


def _load(source):
    """Parsed JSON object, or an array from a ``.npy`` file, for a path or in-memory data."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=False)
        data = json.loads(path.read_text())
    else:
        data = source
    # AlphaFold DB wraps its JSON object in a one-element list.
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    return data


def read_pae(source):
    """(matrix, maximum, token ids) from AlphaFold DB, ColabFold or AlphaFold 3 output.

    Accepts a JSON or .npy file, a parsed JSON object, or an array. Rows are the
    residues the structures are aligned on; columns the residues whose error is
    reported. AlphaFold 3 ``full_data`` files also return (chain, residue) ids per
    token so ligand atoms can be averaged onto their residues.
    """
    data = _load(source)
    tokens = None
    maximum = None
    if isinstance(data, dict):
        if "predicted_aligned_error" in data:
            # AlphaFold DB's current format: the matrix as a nested list of rows.
            matrix = data["predicted_aligned_error"]
            maximum = data.get("max_predicted_aligned_error")
        elif "pae" in data:
            # ColabFold scores files and AlphaFold 3 full_data files. AlphaFold 3 has one
            # token per standard residue but one per atom of a ligand or modified residue,
            # so it also lists the (chain, residue number) that owns each token.
            matrix = data["pae"]
            maximum = data.get("max_pae")
            if "token_chain_ids" in data and "token_res_ids" in data:
                tokens = list(zip(data["token_chain_ids"], data["token_res_ids"]))
        elif {"residue1", "residue2", "distance"} <= set(data):
            # AlphaFold DB v1/v2: flattened (residue1, residue2, distance) triples, 1-based.
            r1, r2 = np.asarray(data["residue1"]) - 1, np.asarray(data["residue2"]) - 1
            matrix = np.full((r1.max() + 1, r2.max() + 1), np.nan)
            matrix[r1, r2] = data["distance"]
            maximum = data.get("max_predicted_aligned_error")
        else:
            raise ValueError("No PAE matrix found; expected predicted_aligned_error or pae")
    else:
        # A bare matrix: a parsed list of rows or an array loaded from .npy.
        matrix = data
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("PAE must be a square matrix")
    # Without a stated maximum, the largest value in the matrix sets the color range.
    maximum = float(maximum) if maximum is not None else float(np.nanmax(matrix))
    return matrix, maximum, tokens


def pae_for(protein, source):
    """(matrix, residue indices, maximum): PAE ordered like the protein's residues."""
    matrix, maximum, tokens = read_pae(source)
    residues = protein.topology.residues
    if tokens is not None:
        # Find the topology residue that owns each token (−1 if the structure lacks it).
        index = {(r.chain, r.resid): i for i, r in enumerate(residues)}
        owners = np.array([index.get((str(c), int(n)), -1) for c, n in tokens])
        ids = np.array(sorted({i for i in owners if i >= 0}), dtype=int)
        if not len(ids):
            raise ValueError("PAE tokens do not match the structure's chains and residue numbers")
        # groups[r, k] = 1 / (tokens of residue r) when token k belongs to residue r, so
        # groups @ PAE @ groups.T averages every block of token pairs into one residue pair.
        # Tokens without an owner get no weight.
        rows = np.searchsorted(ids, owners)
        keep = owners >= 0
        groups = np.zeros((len(ids), len(owners)))
        groups[rows[keep], np.flatnonzero(keep)] = 1
        groups /= groups.sum(1, keepdims=True)
        return groups @ matrix @ groups.T, ids, maximum
    # Without tokens, rows follow residue order. Try the polymer residues, then every residue.
    polymer = [i for i, r in enumerate(residues) if r.trace_atom >= 0]
    for ids in (polymer, list(range(len(residues)))):
        if len(ids) == len(matrix):
            return matrix, np.array(ids, dtype=int), maximum
    raise ValueError(
        f"The PAE matrix has {len(matrix)} rows, but the structure has {len(polymer)} polymer residues; "
        "load the model file that the PAE file describes"
    )


def read_plddt(protein, source):
    """pLDDT per topology residue from an AlphaFold DB confidence or ColabFold scores file."""
    data = _load(source)
    residues = protein.topology.residues
    values = np.full(len(residues), np.nan)
    if isinstance(data, dict) and "confidenceScore" in data:
        # AlphaFold DB confidence_v*.json: scores keyed by residue number, numbered from 1 when
        # the file omits the numbers. AlphaFold DB models are single chains, so the number
        # identifies the residue.
        numbers = data.get("residueNumber", range(1, len(data["confidenceScore"]) + 1))
        lookup = dict(zip(numbers, data["confidenceScore"]))
        for i, r in enumerate(residues):
            if r.trace_atom >= 0 and r.resid in lookup:
                values[i] = lookup[r.resid]
        return values
    # ColabFold scores file ({"plddt": [...]}) or a bare list: one score per polymer residue,
    # in order.
    scores = data["plddt"] if isinstance(data, dict) else data
    scores = np.asarray(scores, dtype=float)
    polymer = [i for i, r in enumerate(residues) if r.trace_atom >= 0]
    if len(scores) != len(polymer):
        raise ValueError(f"{len(scores)} pLDDT values for {len(polymer)} polymer residues")
    values[polymer] = scores
    return values
