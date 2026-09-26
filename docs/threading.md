# Threading animations

`Thread` brings a protein into view as wires. Each chain gets a wire that flies in from its own side of the screen, enters at the C terminus, and traces the backbone to the N terminus. The wire then lies along the cartoon's centerline, and the protein fades in. `Unthread` plays the same motion in reverse. The flight paths are seeded illustrations, not physical motion.

## Example and output

This film uses human deoxyhemoglobin, PDB 4HHB. The two α subunits are tinted rose and the two β subunits teal. Their wires arrive one after another from four sides, and the four hemes appear with the cartoon.

```bash output=gallery-thread-hemoglobin
proteinmotion render examples/thread_hemoglobin.py ThreadHemoglobin --fps 60 -o hemoglobin.mp4
```

[Source on GitHub](https://github.com/pdpppd/proteinmotion/blob/main/examples/thread_hemoglobin.py)

## Thread and unthread

```python
p = Protein.from_file("4hhb.cif").cartoon().center()
self.camera.frame(p, aspect=self.width / self.height)
self.play(Thread(p, self.camera, stagger=0.16), run_time=7)
self.wait(2)
self.play(Unthread(p, self.camera, stagger=0.16), run_time=5)
```

Pass the scene camera so each wire starts just outside the frame. Without it, wires start at a fixed distance from the molecule. The protein is hidden while its wires thread, and fades in over the last part of the clip.

| Option | Effect |
|---|---|
| `radius=0.45` | Wire radius in Å |
| `head=1.8` | Radius of the wire's tip, as a multiple of `radius` |
| `easing=ease_in_out_sine` | Maps clip time to wire travel; any rate function, such as `smooth` |
| `stagger=0.0` | Delay between chains, as a fraction of the threading time, in chain order |
| `settle=0.15` | Final fraction of the clip in which the protein fades in and the wires fade out |
| `swirl=1.0` | Strength of the corkscrew around each flight path; `0` gives smooth arcs |
| `seed=0` | Varies the flight paths |

`Unthread` takes the same options. With `stagger`, the last chain leaves first.

## Glowing tips

The tip of each wire glows while it moves and dims as it comes to rest.

| Option | Effect |
|---|---|
| `glow=12.0` | Halo radius, as a multiple of the wire radius; `0` turns the glow off |
| `glow_color=None` | Halo color; the default blends each chain's color with warm white |
| `glow_brightness=1.0` | Halo intensity |

The glow is drawn by the native renderer and is hidden behind geometry in front of it. EEVEE renders the wires without the glow.

## Colors

Wires take the colors that the cartoon will show, including residue tints. Tint subunits before the animation to give each its own wire color:

```python
p.select(chain=["A", "C"]).set_color("#ef8f8f")   # α subunits
p.select(chain=["B", "D"]).set_color("#5fc9c0")   # β subunits
```

A tint on a whole chain also colors its ligands. To keep the hemes in their own color, tint only the polymer atoms, as the example does.

Each wire follows the same spline through the Cα (or nucleotide trace) atoms that the cartoon uses, so it lands on the cartoon's centerline. Chains with gaps in their trace get one wire per continuous segment.
