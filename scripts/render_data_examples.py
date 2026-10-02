"""Render ligand, numerical-property, plot, density, and torsion examples and verify every frame."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import av

from proteinmotion.cli import load_scene

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "website/public/media/docs"
EXAMPLES = [
    (
        "depth-tunnels",
        "depth_tunnels",
        "DepthTunnels",
        38.5,
        "How deep is it?",
        "1AON: tunnels with a ring every 5 Å to the ADP pocket and the ring's inner wall.",
        ["1aon.cif"],
    ),
    (
        "thread-hemoglobin",
        "thread_hemoglobin",
        "ThreadHemoglobin",
        6.2,
        "Hemoglobin, one chain at a time",
        "4HHB: four wires thread the α and β subunits from their C termini.",
        ["4hhb.cif"],
    ),
    (
        "ligands-and-side-chains",
        "ligands_and_side_chains",
        "LigandsAndSideChains",
        6.5,
        "Ions and coordinating side chains",
        "1CLL Ca²⁺ ions on the cartoon, and the side chains within 3 Å of each ion.",
        ["1cll.cif"],
    ),
    (
        "binding-sites",
        "binding_sites_film",
        "BindingSites",
        36,
        "Calcium sites and a nucleotide pocket",
        "1CLL Ca²⁺ sites, then the ADP and Mg²⁺ pocket of a GroEL subunit (1AON chain A).",
        ["1cll.cif", "1aon.cif"],
    ),
    (
        "troponin-sites",
        "troponin_sites",
        "TroponinSites",
        14,
        "Troponin C metal sites",
        "1NCX: two Cd²⁺ EF-hand sites and a sulfate held by Arg47.",
        ["1ncx.cif"],
    ),
    (
        "side-chain-ensemble",
        "side_chain_ensemble",
        "SideChainEnsemble",
        19,
        "Side chains in an NMR ensemble",
        "2K39 ubiquitin: the Leu8/Ile44/Val70 patch, then every side chain, across conformers.",
        ["2k39.cif"],
    ),
    (
        "nucleic-ions",
        "nucleic_ions",
        "NucleicIons",
        14,
        "Ions and ligands on nucleic acids",
        "1EHZ tRNA with Mg²⁺ and Mn²⁺ ions, then spermine across 2DCG Z-DNA.",
        ["1ehz.cif", "2dcg.cif"],
    ),
    (
        "numerical-properties",
        "numerical_properties",
        "NumericalProperties",
        3,
        "B factors across representations",
        "1UBQ B factors control residue color and cartoon thickness.",
        ["1ubq.cif"],
    ),
    (
        "synchronized-plots",
        "synchronized_plots",
        "SynchronizedPlots",
        4,
        "NMR states with linked plots",
        "2K39 playback with a contact map, sequence strip, and Cα distance trace.",
        ["2k39.cif"],
    ),
    (
        "density-maps",
        "density_maps",
        "DensityMaps",
        6,
        "Electron density and slices",
        "1UBQ PDBe density: animate the contour and move a slice past helix 23–34.",
        ["1ubq.cif", "1ubq.ccp4"],
    ),
    (
        "torsions",
        "torsion_basics",
        "TorsionBasics",
        7.5,
        "Torsion angles",
        "A 12-residue alanine peptide: ψ of Ala6 turns, then every residue moves to α-helix angles.",
        [],
    ),
    (
        "confidence-and-conservation",
        "confidence_and_conservation",
        "ConfidenceAndConservation",
        3,
        "Confidence and conservation",
        "AlphaFold calmodulin pLDDT and PAE, then the ubiquitin surface by hydropathy and Pfam conservation.",
        [
            "AF-P0DP23-F1-model_v6.cif",
            "AF-P0DP23-F1-predicted_aligned_error_v6.json",
            "1ubq.cif",
            "pf00240-seed.sto",
        ],
    ),
    (
        "hemoglobin-morph",
        "structure_morph",
        "HemoglobinMorph",
        11,
        "Hemoglobin from T to R",
        "2DN2 deoxy to 2DN1 oxy (assembly 1), superposed on α1β1, with the change in Cα distances.",
        ["2dn2.cif", "2dn1.cif"],
    ),
    (
        "adenylate-kinase-morph",
        "structure_morph",
        "AdenylateKinaseMorph",
        11.5,
        "Adenylate kinase closes",
        "4AKE open to 1AKE closed around Ap5A, superposed on the CORE.",
        ["4ake.cif", "1ake.cif"],
    ),
    (
        "torsions-film",
        "torsions",
        "TorsionsFilm",
        32,
        "Two angles per residue",
        "φ and ψ of one residue, an α helix and its hydrogen bonds, the 1PGA β-hairpin, and ubiquitin.",
        ["1pga.cif", "1ubq.cif"],
    ),
]


# Longer gallery films use a lower bitrate to keep the committed previews small.
BITRATE = {
    "depth-tunnels": "4M",
    "thread-hemoglobin": "4M",
    "binding-sites": "4M",
    "troponin-sites": "4M",
    "side-chain-ensemble": "4M",
    "nucleic-ions": "4M",
    "torsions-film": "4M",
    "hemoglobin-morph": "4M",
    "adenylate-kinase-morph": "4M",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[entry[0] for entry in EXAMPLES])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest_path = MEDIA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key, file, cls, poster_time, title, caption, data in EXAMPLES:
        if args.only and key not in args.only:
            continue
        source = ROOT / "examples" / (file + ".py")
        scene = load_scene(source, cls, width=1280, height=720, fps=60).build()
        video, poster = MEDIA / (key + ".mp4"), MEDIA / (key + ".jpg")
        if not args.validate_only:
            scene.render(video, bitrate=BITRATE.get(key, "5M"), progress=False)
        count = 0
        with av.open(video) as container:
            assert container.streams.video[0].average_rate == 60
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (1280, 720)
                assert abs(float(frame.pts * frame.time_base) - count / 60) < 0.001
                if count == round(poster_time * 60):
                    frame.to_image().save(poster, quality=92)
                count += 1
        assert count == max(1, math.ceil(scene.duration * scene.fps))
        dependencies = [source, *[ROOT / "examples/data" / p for p in data]]
        manifest[key] = dict(
            title=title,
            caption=caption,
            source=str(source.relative_to(ROOT)),
            scene=cls,
            duration=scene.duration,
            frames=count,
            width=1280,
            height=720,
            fps=60,
            video=f"media/docs/{key}.mp4",
            poster=f"media/docs/{key}.jpg",
            poster_time=poster_time,
            video_sha256=digest(video),
            poster_sha256=digest(poster),
            dependencies={str(p.relative_to(ROOT)): digest(p) for p in dependencies},
            code=source.read_text().strip(),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        doc = ROOT / "docs" / (key + ".md")
        if doc.exists():
            doc.write_text(
                re.sub(
                    rf"(```python output={key}\n).*?\n```",
                    lambda m: m[1] + source.read_text().strip() + "\n```",
                    doc.read_text(),
                    flags=re.DOTALL,
                )
            )
        print(f"{key}: {count} frames decoded at 1280×720, 60 fps", flush=True)


if __name__ == "__main__":
    main()
