# Alpha-helix hydrogen-bond test

This controlled example makes the backbone hydrogen-bond pattern visible: **O(i)···H–N(i+4)**. It uses a fixed, idealized 16-residue backbone with explicit amide hydrogens, so the view can rotate without contacts changing between conformers.

[Watch the film](https://pdpppd.github.io/proteinmotion/gallery/#alpha-helix) · [Download the complete script](alpha_helix_hbonds.py) · [Source on GitHub](https://github.com/pdpppd/proteinmotion/blob/main/examples/alpha_helix_hbonds.py)

![A close-up distinguishing the hydrogen bond from the covalent N–H bond](alpha-helix.png)

## What the test establishes

The detector receives coordinates and the normal geometry criteria, **N···O ≤ 3.5 Å and N–H···O ≥ 150°**. It is not told to connect residues four positions apart. It finds exactly the **12 expected pairs**: O1–H5 through O12–H16, with no extra pairs. The i→i+4 pattern is the defining backbone hydrogen-bond arrangement of an [α helix](https://www.ebi.ac.uk/training/online/courses/foundations-protein-structure/principles-of-protein-folding-and-architecture/secondary-structure-%CE%B1-helices-and-%CE%B2-sheets/%CE%B1-helix/).

The backbone is constructed with φ = −57°, ψ = −47° and trans peptide bonds (ω = 180°). It contains N, Cα, C, O and amide H; side chains, other hydrogens and terminal caps are omitted. This is a synthetic geometry fixture, not a deposited structure, MD trajectory or energy-minimized peptide. Coordinates are held fixed throughout the film.

| Quantity | Value in this idealized fixture |
|---|---:|
| Expected / detected bonds | 12 / 12 |
| H···O distance | 2.10 Å |
| N···O distance | 3.09 Å |
| N–H···O angle | 165.6° |

The film draws **gold dashed H···O segments** and keeps covalent N–H bonds solid. It first shows the entire network, then isolates O5···H–N9 and labels the two different distances. Short bond lines do not carry inline numbers; the measurements sit beside the close-up so a label does not hide the bond.

The zoom begins at 8.4 seconds and fades the surrounding residues. In v0.6.2, covalent sticks are clipped at each atom surface before blending, so the fade no longer exposes cylinders inside the balls. This applies automatically to global and residue-wise opacity; no scene-code changes are needed.

[All measured pairs as CSV](alpha-helix-hbonds.csv) · [Geometry and render report](alpha-helix-hbonds-report.json)

## Run it

From the repository root:

```bash
proteinmotion render examples/alpha_helix_hbonds.py AlphaHelixHBonds -o alpha-helix-hbonds.mp4 --fps 60
```

The core authoring code is:

```python
from proteinmotion import HydrogenBonds, Write

# p is the backbone with explicit H from the complete example.
hb = HydrogenBonds(p, hydrogens="explicit", max_distance=3.5, min_angle=150)
network = hb.highlight(
    mode="3d",
    endpoints="hydrogen_acceptor",
    show_distances=False,
    color="#f2ba67",
    radius=0.055,
    dash_count=5,
)
self.play(Write(network), run_time=2)
```

For a structure file, load explicit H with `Protein.from_file(path, include_hydrogens=True)`. `endpoints="hydrogen_acceptor"` draws H···A and reports that distance if `show_distances=True`. With `hydrogens="backbone"` or inferred H in `"auto"` mode, the endpoint follows the virtual hydrogen position; that does not itself create a visible H atom. `hb.hydrogen_position(pair)` returns the current explicit or inferred H coordinates in model Å.

The backward-compatible default `endpoints="donor_acceptor"` draws D···A and labels donor–acceptor distance. `Interaction.distance` always remains D···A, whichever line endpoints you choose. Hydrogen endpoints apply only to hydrogen-bond analyses.

## Why the earlier example was hard to read

The NMR example displayed one donor–acceptor ruler at a time inside a much larger model, without visible hydrogens. Its inline caption occupied much of the short line. Changing conformers could also change which contacts passed the geometry cutoff. That view did not make the repeating helix network clear.

This diagnostic draws all qualifying helix contacts, shows the H atoms, moves only the camera and separates measurements from the short lines. No angle or distance thresholds were relaxed to obtain the idealized network.

## Regression checks and a real structure

The checks verify the generated φ/ψ angles independently, the entire expected residue-pair set, explicit and virtual H endpoints, transformed anchors, and reproducible GPU rendering. Reversing one amide H removes its bond; an extended backbone with φ = −135° / ψ = 135° does not acquire forced i→i+4 bonds.

A separate check uses the included **1UBQ residues 23–34** with inferred backbone H. At the default 150° angle cutoff it detects six local helix bonds. Two further i→i+4 pairs have angles approximately 149.8° and 145.4°, so they are excluded; setting the documented cutoff to 140° includes all eight. This demonstrates real-coordinate cutoff sensitivity, rather than assuming every annotated helix must pass one geometric threshold. [General hydrogen-bond methods and limitations](interactions.md).
