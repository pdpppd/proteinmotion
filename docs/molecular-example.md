# Complete molecular tools script

Two scripts use the [2K39 ubiquitin ensemble](https://www.rcsb.org/structure/2K39). The first shows residue colors, transparency, and surfaces. The second shows hydrogen bonds, electrostatic estimates, and distance labels.

[Download molecular_tools.py](molecular_tools.py) · [View source on GitHub](https://github.com/pdpppd/proteinmotion/blob/main/examples/molecular_tools.py)

Run from the repository root:

```bash output=gallery-surfaces
proteinmotion render examples/molecular_tools.py StylingAndSurface -o styling-and-surface.mp4 --fps 60
```

```bash output=gallery-interactions
proteinmotion render examples/molecular_tools.py InteractionsAndDistances -o interactions-and-distances.mp4 --fps 60
```

Both videos interpolate between NMR conformations. The interaction scene uses inferred backbone hydrogens and example formal side-chain charges. To use prepared charges, supply an array or call `Electrostatics.from_pqr()`. See [interactions](interactions.md) for charge preparation and [surfaces](styling.md) for mesh settings.

{{MOLECULAR_SOURCE}}
