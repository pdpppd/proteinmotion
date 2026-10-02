"""AlphaFold confidence files: predicted aligned error (PAE) and pLDDT."""

import json
from pathlib import Path

import numpy as np


def _load(source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=False)
        data = json.loads(path.read_text())
    else:
        data = source
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
            matrix = data["predicted_aligned_error"]
            maximum = data.get("max_predicted_aligned_error")
        elif "pae" in data:
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
        matrix = data
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("PAE must be a square matrix")
    maximum = float(maximum) if maximum is not None else float(np.nanmax(matrix))
    return matrix, maximum, tokens


def pae_for(protein, source):
    """(matrix, residue indices, maximum): PAE ordered like the protein's residues."""
    matrix, maximum, tokens = read_pae(source)
    residues = protein.topology.residues
    if tokens is not None:
        index = {(r.chain, r.resid): i for i, r in enumerate(residues)}
        owners = np.array([index.get((str(c), int(n)), -1) for c, n in tokens])
        ids = np.array(sorted({i for i in owners if i >= 0}), dtype=int)
        if not len(ids):
            raise ValueError("PAE tokens do not match the structure's chains and residue numbers")
        rows = np.searchsorted(ids, owners)
        keep = owners >= 0
        groups = np.zeros((len(ids), len(owners)))
        groups[rows[keep], np.flatnonzero(keep)] = 1
        groups /= groups.sum(1, keepdims=True)
        return groups @ matrix @ groups.T, ids, maximum
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
        numbers = data.get("residueNumber", range(1, len(data["confidenceScore"]) + 1))
        lookup = dict(zip(numbers, data["confidenceScore"]))
        for i, r in enumerate(residues):
            if r.trace_atom >= 0 and r.resid in lookup:
                values[i] = lookup[r.resid]
        return values
    scores = data["plddt"] if isinstance(data, dict) else data
    scores = np.asarray(scores, dtype=float)
    polymer = [i for i, r in enumerate(residues) if r.trace_atom >= 0]
    if len(scores) != len(polymer):
        raise ValueError(f"{len(scores)} pLDDT values for {len(polymer)} polymer residues")
    values[polymer] = scores
    return values
