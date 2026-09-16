"""Reproduce the contact-map figure and auditable residue-pair table."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from proteinmotion import ContactMatch, Protein
from proteinmotion.matching import ca_selection, contact_map


def report(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = Path(__file__).parent / "data"
    source = Protein.from_file(data / "1cll.cif", chains="A")
    target = Protein.from_file(data / "1ncx.cif", chains="A")
    match = ContactMatch.load(data / "calmodulin-troponin-match.json")
    a, x, _ = ca_selection(source)
    b, y, _ = ca_selection(target)
    ia = np.searchsorted(a, match.source_indices)
    ib = np.searchsorted(b, match.target_indices)
    params = match.report
    ca = contact_map(x, params["cutoff_angstrom"], params["softness_angstrom"])
    cb = contact_map(y, params["cutoff_angstrom"], params["softness_angstrom"])
    error = np.abs(ca[np.ix_(ia, ia)] - cb[np.ix_(ib, ib)])
    k = len(ia)
    delay, duration = 0.025, 6.0
    with (output / "backbone-residue-pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "matched_rank",
                "source_topology_index",
                "source_chain",
                "source_residue_number",
                "source_insertion_code",
                "source_residue_name",
                "target_topology_index",
                "target_chain",
                "target_residue_number",
                "target_insertion_code",
                "target_residue_name",
                "motion_start_seconds",
                "motion_end_seconds",
            ]
        )
        for rank, (i, j, sk, tk) in enumerate(
            zip(match.source_indices, match.target_indices, match.source_keys, match.target_keys)
        ):
            start = rank * delay
            writer.writerow(
                [rank + 1, i, *sk, j, *tk, f"{start:.3f}", f"{start + duration - (k - 1) * delay:.3f}"]
            )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "text.color": "#dce8f4",
            "axes.labelcolor": "#b7c9da",
            "axes.edgecolor": "#3c4e64",
            "xtick.color": "#9cb0c6",
            "ytick.color": "#9cb0c6",
            "axes.facecolor": "#101e30",
            "figure.facecolor": "#0b1422",
        }
    )
    fig = plt.figure(figsize=(17, 10.5), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.75])
    cmaps = [
        LinearSegmentedColormap.from_list("source", ["#0b1422", "#56d8c0"]),
        LinearSegmentedColormap.from_list("target", ["#0b1422", "#f2ba67"]),
        "magma",
    ]
    for col, (matrix, cmap, title, maximum) in enumerate(
        [
            (ca, cmaps[0], f"Calmodulin · 1CLL\n{k}/{len(x)} Cα residues selected", 1),
            (cb, cmaps[1], f"Troponin C · 1NCX\n{k}/{len(y)} Cα residues selected", 1),
            (
                error,
                cmaps[2],
                f"Matched contact disagreement\nRMS {params['contact_rms_error']:.4f} · max {error.max():.4f}",
                params["max_contact_error"],
            ),
        ]
    ):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            matrix,
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=maximum,
            extent=(0.5, len(matrix) + 0.5, 0.5, len(matrix) + 0.5),
            interpolation="nearest",
        )
        ax.set_title(title, loc="left", fontsize=14, pad=16, color="#edf4fc")
        label = "Matched rank (N → C)" if col == 2 else "Cα order (N → C)"
        ax.set(xlabel=label, ylabel=label)
        if col < 2:
            ids, color = (ia, "#56d8c0") if col == 0 else (ib, "#f2ba67")
            ax.scatter(
                ids + 1,
                np.full(len(ids), -0.02),
                transform=ax.get_xaxis_transform(),
                marker="|",
                s=40,
                c=color,
                clip_on=False,
            )
        fig.colorbar(
            im, ax=ax, shrink=0.72, pad=0.015, label="Absolute error" if col == 2 else "Soft contact"
        )
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(ia + 1, ib + 1, color="#657c98", linewidth=1, zorder=0)
    ax.scatter(
        ia + 1,
        ib + 1,
        c=np.arange(k),
        cmap=LinearSegmentedColormap.from_list("pair", ["#56d8c0", "#f2ba67"]),
        s=22,
    )
    ax.set(
        title="Selected correspondence · one-to-one, strictly N-to-C",
        xlabel="Calmodulin Cα order",
        ylabel="Troponin C Cα order",
        xlim=(0, len(x) + 1),
        ylim=(0, len(y) + 1),
    )
    ax.grid(alpha=0.10)
    ax = fig.add_subplot(gs[1, 2])
    t = np.linspace(0, duration, 400)
    for rank, color in zip([0, k // 3, 2 * k // 3, k - 1], ["#56d8c0", "#90bbc6", "#cda37e", "#f2ba67"]):
        u = np.clip((t - rank * delay) / (duration - (k - 1) * delay), 0, 1)
        ax.plot(t, u**3 * (10 + u * (-15 + 6 * u)), label=f"Residue rank {rank + 1}", color=color)
    ax.set(title="Staggered, eased movement", xlabel="Seconds into the morph", ylabel="Movement progress")
    ax.legend(frameon=False, fontsize=9, labelcolor="#b7c9da")
    ax.grid(alpha=0.10)
    fig.suptitle(
        f"Contact-guided backbone morph  /  {k} matched residue pairs",
        x=0.03,
        ha="left",
        fontsize=24,
        weight="bold",
        color="#edf4fc",
    )
    fig.supxlabel(
        f"Soft contacts: sigmoid((8 Å − distance)/1.5 Å). Pairwise error ≤ {params['max_contact_error']:.2f}. "
        f"Delay = {delay * 1000:.0f} ms/residue.\n"
        "Bounded search found this feasible mapping; global optimality was not proved. "
        "This is a visual morph, not a simulated transition.",
        fontsize=11,
        color="#9cb0c6",
    )
    fig.savefig(output / "backbone-contact-map.png", dpi=170)
    fig.savefig(output / "backbone-contact-map.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="contact-report")
    report(parser.parse_args().output)
