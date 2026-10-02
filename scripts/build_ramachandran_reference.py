"""Build the Ramachandran plot's favored and allowed regions from high-resolution PDB entries.

The reference set is one representative protein chain per 30% sequence-identity
cluster among X-ray entries at 1.4 Å resolution or better with R-free at most
0.20, a single protein entity and 80–700 residues (RCSB search, best resolution
per cluster). Residues are kept when their backbone and flanking peptide atoms
have no alternate locations, full occupancy and B factors below 30 Å².

Each class (general, glycine, trans proline, pre-proline) is histogrammed on a
2° grid, smoothed with a periodic Gaussian, and thresholded at the densities
that enclose 98% (favored) and 99.8% (allowed) of its residues. Isolated patches
smaller than 12 cells are dropped.

Run from a checkout:
    python scripts/build_ramachandran_reference.py --cache ~/.cache/proteinmotion-rama
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import gemmi
import numpy as np
from scipy.ndimage import gaussian_filter, label

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/proteinmotion/data/ramachandran.npz"
PROVENANCE = ROOT / "src/proteinmotion/data/ramachandran.json"
QUERY = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": 1.4,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": "X-RAY DIFFRACTION",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "equals",
                    "value": 1,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid",
                    "operator": "equals",
                    "value": 0,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                    "operator": "range",
                    "value": {"from": 80, "to": 700},
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "refine.ls_R_factor_R_free",
                    "operator": "less_or_equal",
                    "value": 0.2,
                },
            },
        ],
    },
    "request_options": {
        "group_by": {
            "aggregation_method": "sequence_identity",
            "similarity_cutoff": 30,
            "ranking_criteria_type": {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"},
        },
        "group_by_return_type": "representatives",
        "paginate": {"start": 0, "rows": 1000},
    },
    "return_type": "polymer_entity",
}
CLASSES = ("general", "glycine", "proline", "preproline")
STEP = 2.0
SIGMA = 5.0  # degrees
MIN_ISLAND = 12  # grid cells; smaller isolated patches are dropped
LEVELS = {"favored": 0.98, "allowed": 0.998}
STANDARD = set(gemmi.find_tabulated_residue(n).name for n in (
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
))  # fmt: skip


def entries():
    request = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(QUERY).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    return sorted({item["identifier"].split("_")[0] for item in result["result_set"]})


def download(identifier, cache):
    path = cache / f"{identifier}.cif.gz"
    if not path.exists() or not path.stat().st_size:
        with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{identifier}.cif.gz", timeout=120
        ) as response:
            path.write_bytes(response.read())
    return path


def clean(atom):
    return atom is not None and atom.occ >= 0.999 and atom.b_iso < 30 and not atom.has_altloc()


def torsions(path):
    """(class, φ, ψ) for well-ordered residues of the first model."""
    structure = gemmi.read_structure(str(path))  # gemmi reads .gz directly
    structure.setup_entities()
    rows = []
    for chain in structure[0]:
        polymer = chain.get_polymer()
        for i in range(1, len(polymer) - 1):
            prev, res, nxt = polymer[i - 1], polymer[i], polymer[i + 1]
            if res.name not in STANDARD or nxt.name not in STANDARD or prev.name not in STANDARD:
                continue
            atoms = [prev.find_atom("C", "*"), res.find_atom("N", "*"), res.find_atom("CA", "*")]
            atoms += [res.find_atom("C", "*"), res.find_atom("O", "*"), nxt.find_atom("N", "*")]
            if not all(clean(a) for a in atoms):
                continue
            if atoms[0].pos.dist(atoms[1].pos) > 1.5 or atoms[3].pos.dist(atoms[5].pos) > 1.5:
                continue
            phi = np.degrees(gemmi.calculate_dihedral(*(a.pos for a in atoms[:4])))
            psi = np.degrees(gemmi.calculate_dihedral(atoms[1].pos, atoms[2].pos, atoms[3].pos, atoms[5].pos))
            if res.name == "GLY":
                kind = "glycine"
            elif res.name == "PRO":
                omega = np.degrees(
                    gemmi.calculate_dihedral(
                        prev.find_atom("CA", "*").pos, atoms[0].pos, atoms[1].pos, atoms[2].pos
                    )
                )
                if abs(omega) < 90:  # cis proline
                    continue
                kind = "proline"
            elif nxt.name == "PRO":
                kind = "preproline"
            else:
                kind = "general"
            rows.append((CLASSES.index(kind), phi, psi))
    return rows


def regions(phi, psi):
    bins = np.arange(-180, 180 + STEP, STEP)
    counts, _, _ = np.histogram2d(psi, phi, bins=[bins, bins])
    density = gaussian_filter(counts, SIGMA / STEP, mode="wrap")
    index = lambda values: np.clip(((np.asarray(values) + 180) // STEP).astype(int), 0, len(bins) - 2)  # noqa: E731
    at_points = density[index(psi), index(phi)]
    levels = np.zeros(density.shape, np.uint8)
    thresholds = {}
    for value, (name, fraction) in enumerate(sorted(LEVELS.items(), key=lambda item: -item[1]), start=1):
        threshold = float(np.quantile(at_points, 1 - fraction))
        inside = density >= threshold
        # Drop tiny isolated patches; the plot wraps at ±180°, so islands are labeled periodically.
        tiled, count = label(np.tile(inside, (3, 3)))
        center = tiled[len(inside) : 2 * len(inside), len(inside) : 2 * len(inside)]
        sizes = np.bincount(tiled.ravel())
        keep = (center > 0) & (sizes[center] >= MIN_ISLAND * 9)
        levels[keep] = value
        thresholds[name] = threshold
    return levels, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/proteinmotion-rama")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    identifiers = entries()
    with ThreadPoolExecutor(16) as pool:
        paths = list(pool.map(lambda identifier: download(identifier, args.cache), identifiers))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(torsions, paths))
    data = np.array([row for rows in results for row in rows], dtype=float).reshape(-1, 3)
    grids, counts, thresholds = {}, {}, {}
    for k, name in enumerate(CLASSES):
        subset = data[data[:, 0] == k]
        grids[name], thresholds[name] = regions(subset[:, 1], subset[:, 2])
        counts[name] = int(len(subset))
        print(f"{name}: {len(subset)} residues")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, step=STEP, **grids)
    PROVENANCE.write_text(
        json.dumps(
            {
                "description": "Favored (98%) and allowed (99.8%) φ/ψ regions; level 2 = favored, 1 = allowed.",
                "grid": "Rows ψ, columns φ, from −180° in 2° cells.",
                "smoothing_sigma_degrees": SIGMA,
                "residue_filter": "No altloc, occupancy 1, B < 30 Å² for C(i−1), N, Cα, C, O, N(i+1); cis Pro excluded.",
                "query": QUERY,
                "date": date.today().isoformat(),
                "entries": len(identifiers),
                "residues": counts,
                "pdb_ids": identifiers,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"Wrote {OUTPUT} and {PROVENANCE}")


if __name__ == "__main__":
    main()
