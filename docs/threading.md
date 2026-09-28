# Threading animations

`Thread` brings the actual cartoon, ribbon, or ball-and-stick representation into view. Each traced chain follows a seeded flight path through the C terminus toward the N terminus, with quintic easing over the full clip. `Unthread` reverses the motion. Surface representations raise `ValueError`.

This is an illustrative entrance, not protein folding or molecular dynamics. Each residue and bound ligand moves as a rigid group; bonds between residues can stretch during travel. The final coordinates are restored exactly. Both native and Studio render the transported geometry.

## Thread and unthread

```python
p = Protein.from_file("2dri.cif", chains="A").cartoon().center()
self.camera.frame(p, aspect=self.width / self.height)
self.play(
    Thread(p, self.camera, swirl=0.25,
           aspect=self.width / self.height),
    run_time=8,
)
self.wait(2)
self.play(Unthread(p, self.camera), run_time=8)
```

Use `.ball_and_stick()` instead of `.cartoon()` to thread atoms and bonds. The current representation stays visible throughout; there is no wire-to-protein fade. The animation changes coordinates, so it cannot run concurrently with another coordinate or representation animation such as `Morph`, `Deform`, `PlayTrajectory`, or `Representation`. Coloring and camera movement can run alongside it.

Pass the scene camera and aspect ratio so the entrance starts outside the frame. Without a camera, the starting distance is based on the molecule's size. Frame the resting protein before constructing the animation.

| Option | Effect |
|---|---|
| `easing=None` | Defaults to quintic `smooth` for geometry; accepts a function returning a value in `[0, 1]` |
| `stagger=0.0` | Delay per chain as a fraction of the clip, in chain order; each chain settles independently |
| `swirl=1.0` | Corkscrew strength along the flight path; `0` gives smooth arcs |
| `seed=0` | Varies the flight paths |
| `aspect=16/9` | Scene width divided by height, for off-screen entry positioning |
| `mode="geometry"` | Transport the current representation; `"wire"` selects the original wire-and-crossfade effect |

`Unthread` takes the same options. With `stagger`, the last chain leaves first. Both animations support deterministic seeking.

## Studio review movies

This script renders matching cartoon and ball-and-stick movies of the ribose-binding protein, PDB 2DRI, with a fixed camera, no glow, and an eight-second entrance:

```bash
python examples/studio_threading.py
```

The two 1080p, 60 fps movies are written to `output/studio-threading-eased/`.

## Glowing tips

Optional halos follow each leading terminus and dim as it settles. Set `glow=0` to focus on the geometry.

| Option | Effect |
|---|---|
| `radius=0.45` | Base radius in Å for glow sizing; also the wire radius in wire mode |
| `glow=12.0` | Halo radius as a multiple of `radius`; `0` turns glow off |
| `glow_color=None` | Halo color; defaults to the chain color blended with warm white |
| `glow_brightness=1.0` | Halo intensity |

## Original wire effect

Use `mode="wire"` for the original entrance: a wire flies in, traces the backbone, and crossfades into the protein over the final `settle=0.15` fraction. `settle` applies only to wire mode. In this mode, `easing=None` defaults to `ease_in_out_sine` and `head=1.8` controls the tip radius as a multiple of `radius`. Wire colors follow the cartoon colors, including residue tints.

```python
self.play(Thread(p, self.camera, mode="wire", stagger=0.16), run_time=7)
```

The hemoglobin example explicitly uses wire mode, with rose α subunits, teal β subunits, and four hemes:

```bash output=gallery-thread-hemoglobin
proteinmotion render examples/thread_hemoglobin.py ThreadHemoglobin --fps 60 -o hemoglobin.mp4
```

[Source on GitHub](https://github.com/pdpppd/proteinmotion/blob/main/examples/thread_hemoglobin.py)
