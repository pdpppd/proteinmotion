# Complete molecular tools script

Two runnable films using the included [2K39 ubiquitin ensemble](https://www.rcsb.org/structure/2K39): residue color/transparency and moving surfaces, followed by hydrogen bonds and screened electrostatic contacts with live distance labels.

[Download molecular_tools.py](molecular_tools.py) · [View source on GitHub](https://github.com/pdpppd/proteinmotion/blob/main/examples/molecular_tools.py)

Run from the repository root:

```bash
proteinmotion render examples/molecular_tools.py StylingAndSurface -o styling-and-surface.mp4 --fps 60
proteinmotion render examples/molecular_tools.py InteractionsAndDistances -o interactions-and-distances.mp4 --fps 60
```

These films use interpolated NMR conformers as a visualization, not physical time. The interaction scene explicitly labels virtual backbone hydrogens and illustrative formal side-chain charges. For prepared charges, use an atom-ordered array or `Electrostatics.from_pqr()` as described in the [interaction guide](interactions.md). Surface meshes rebuild as coordinates change; the [surface guide](styling.md) explains quality and speed options.

{{MOLECULAR_SOURCE}}
