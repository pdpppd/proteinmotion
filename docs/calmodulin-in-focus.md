# Calmodulin in focus

A 68-second film rendered with Blender EEVEE at 1080p and 60 fps. The camera follows one calmodulin structure through helix close-ups, atomic detail, a molecular surface, and a closing orbit. Focus moves between two selected helices.

[Watch the film](https://pdpppd.github.io/proteinmotion/gallery/#calmodulin-in-focus) · [Download the script](calmodulin_in_focus.py) · [Download the structure](data/1cll.cif)

## Chapters

| Time | View |
|---|---|
| 00:00–00:08 | Whole-protein reveal and camera orbit |
| 00:08–00:17 | Helix 5–19, with the surrounding structure at 10% opacity |
| 00:17–00:27 | Backbone atoms and hydrogen-bond candidates |
| 00:27–00:35 | Pull back and mark helices 5–19 and 118–128 |
| 00:35–00:45 | Move lens focus from the gold helix to the teal helix |
| 00:45–00:55 | Molecular surface colored by deposited Cα B factors |
| 00:55–01:08 | Return through ribbon and cartoon views, then fade out |

## Render the film

Install [Blender 4.5 or later](https://www.blender.org/download/), then use the current repository checkout:

```bash
git clone https://github.com/pdpppd/proteinmotion.git
cd proteinmotion
python -m pip install -e .
python examples/calmodulin_in_focus.py --output calmodulin-in-focus.mp4
```

The script selects EEVEE and sets the background, lens settings, and output quality. It reads `examples/data/1cll.cif`. If you download the files separately, place the structure in a `data` folder beside the script.

The export uses 64 EEVEE samples, 1.25× spatial supersampling, and a 28-pixel blur limit. To make a smaller preview:

```bash
python examples/calmodulin_in_focus.py \
  --width 960 --height 540 --samples 32 --supersampling 1 \
  --output calmodulin-preview.mp4
```

See [EEVEE and depth of field](eevee.md) for Blender setup and residue focus controls.

## Full script and output

The video beside this script is the full 68-second film. Edit the residue selections, `FocusPull` calls, or animation durations to change the sequence.

{{CALMODULIN_FOCUS_SOURCE}}

## Structure and measurements

The input is [PDB 1CLL](https://www.rcsb.org/structure/1CLL), chain A. Coordinates stay fixed throughout the film. Camera movement, lens focus, colors, and representations change over time.

Gold lines show five backbone hydrogen-bond candidates: donor residues 9, 10, 15, 16, and 17 connect to acceptor residues 5, 6, 11, 12, and 13. The geometric detector uses inferred amide hydrogen positions; each line connects donor N to acceptor O.

Surface colors use the deposited Cα B factor for each residue, with a fixed 0–50 Å² scale. Each lens target is the world-space centroid of its selected atoms. A residue range defines one focus plane, so atoms at different depths can have different amounts of blur.
