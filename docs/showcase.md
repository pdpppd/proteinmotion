# Feature demo

The homepage video is a **100.5-second scene at 60 fps**. It demonstrates representations, styling, labels, and state playback with calmodulin. A backbone morph changes calmodulin into troponin C, which is then used for the interaction measurements.

[Watch the film](https://pdpppd.github.io/proteinmotion/gallery/#showcase) · [Download the scene script](feature_showcase.py) · [Source and input files on GitHub](https://github.com/pdpppd/proteinmotion/blob/main/examples/feature_showcase.py)

![Calmodulin in the continuous ProteinMotion feature tour](showcase.png)

## Chapters

The chapter times mark caption changes within the scene.

| Start | Chapter | What you see |
|---|---|---|
| 00:00.0 | Representations | Calmodulin crossfades between cartoon, ribbon, ball-and-stick and solvent-excluded surface while the camera orbits. |
| 00:11.8 | Color + transparency | Residue colors and opacity change in sequence on the surface and cartoon. |
| 00:21.0 | Focus + annotation | The camera zooms into calmodulin's central helix. Sphere, box, and atom highlights mark the region, with labels for Lys75 and Glu84. |
| 00:33.1 | States + deformation | The same calmodulin model interpolates slowly through three deposited NMR conformers via an XTC reader, returns to its starting coordinates, then demonstrates procedural deformation. |
| 00:52.2 | Contact-guided morph | 114 matched Cα positions move from calmodulin to troponin C with a 25 ms N-to-C delay. Unmatched residues fade out and in during an 8.5-second morph. |
| 01:05.7 | Hydrogen bonds | The camera moves directly into the morphed troponin C's helix, showing ten geometrically detected backbone N···O contacts. |
| 01:13.9 | Live distances | A ruler between Cα84 and Cα97 changes from a depth-tested 3D line to a 2D overlay. |
| 01:21.5 | Electrostatics | The camera follows the same helix into the charged residues 91 and 95; imported charges drive screened-Coulomb highlighting. |
| 01:29.7 | Overview | The camera pulls back to show troponin C, then the scene fades out. |

The gallery also includes a [GroEL/GroES assembly example](https://pdpppd.github.io/proteinmotion/gallery/#groel).

## Render the video

Use the repository checkout so the included structures and fixtures are available. This example uses the optional MD reader:

```bash
python -m pip install -e '.[md]'
```

```bash output=gallery-showcase
proteinmotion render examples/feature_showcase.py FeatureShowcase \
  --fps 60 -o proteinmotion-showcase.mp4
```

`FeatureShowcase` uses standard `ProteinScene` methods: `play()`, `wait()`, and `seek()`. It contains one camera and two protein objects. After `BackboneMorph`, the scene uses the target object for hydrogen bonds, distances, and electrostatics.

`scene.chapter_manifest()` returns the chapter times. Edit the script to change camera margins, residue delays, representation transitions, and line styles. ProteinMotion renders the titles and transitions with the molecular scene.

## Data and interpretation

- The starting model is calmodulin **1CLL**. [**1CFC**](https://www.rcsb.org/structure/1CFC) supplies 25 deposited calcium-free calmodulin NMR conformers. All **1,130 protein heavy atoms** in the 1CLL display topology map by exact atom identity to 1CFC and are aligned on common Cαs. The six nonprotein atoms absent from 1CFC are hidden during playback. The display retains its original topology and secondary-structure assignments.
- The XTC fixture contains all 25 mapped conformers. The film uses only states **1–3**, with 3.5 seconds per inter-state transition. These are NMR conformations stored in XTC format to demonstrate the reader. The animation interpolates into and out of the ensemble.
- Run `from examples.feature_showcase import prepare_inputs; prepare_inputs()` to regenerate the PDB/XTC files and charge array. The explicit procedural deformation is also illustrative.
- The **1CLL→1NCX** morph uses the included contact-map correspondence. The search returns a feasible match within its time and candidate limits. The animation interpolates matched coordinates. See [matching methods](morphing.md).
- Troponin C's hydrogen bonds are detected using N···O ≤ 3.5 Å and N–H···O ≥ 150°. The angle uses **inferred backbone amide H**, and the drawn line connects the measured donor N and acceptor O. The analysis uses coordinates from the deposited troponin C structure.
- The imported 1NCX charge array contains **example formal side-chain charges**. Screened-Coulomb parameters are dielectric 80 and screening length 8 Å. The contact energy is an approximation under those settings.

The master is **1920×1080 at 60 fps**, rendered through Metal with 4× MSAA and VideoToolbox encoding. The homepage/gallery preview is **1280×720 at 60 fps**, with muted playback, standard video controls, and respect for reduced-motion preferences on the homepage. [Render measurements, continuity checks and input verification](showcase-report.json).
