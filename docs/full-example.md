# Complete NMR example

`examples/nmr_regions.py` animates a ubiquitin NMR ensemble with 3D highlights, camera focus, and a transition to ball-and-stick.

PDB 2K39 contains 116 models of ubiquitin. This example interpolates between models to show their structural variation. Model order is not a time sequence.

## Run a scene

```bash
proteinmotion render examples/nmr_regions.py RegionTour -o regions.mp4 --fps 60
proteinmotion render examples/nmr_regions.py NMRStates -o cartoon.mp4 --fps 60
proteinmotion render examples/nmr_regions.py NMRAtoms -o atoms.mp4 --fps 60
```

## Full source

{{NMR_SOURCE}}

## Adapt the scene

Change `residues=(23, 34)` for an inclusive author-number range, or use `[8, 44, 70]` for specific residues. `start` and `end` are zero-based model indices. `run_time` is the clip duration in seconds. `margin` controls camera framing; `padding` controls the 3D annotation size in ångströms.

Download [the Python script](nmr_regions.py) or browse [the examples on GitHub](https://github.com/pdpppd/proteinmotion/tree/main/examples).
