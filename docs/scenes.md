# Scenes and motion

```python
from proteinmotion import *
import numpy as np


class MyScene(ProteinScene):
    def construct(self):
        p = Protein.from_file("examples/data/1ubq.cif").cartoon().center()
        self.add(p)
        self.camera.frame(p)
        self.play(FadeIn(p), run_time=1)
        self.play(Rotate(p, np.pi), run_time=3)
        self.play(Representation(p, "ribbon"), run_time=1)
        self.play(p.animate.shift((5, 0, 0)), self.camera.animate.orbit(theta=0.6), run_time=2)
        self.wait(0.5)


MyScene(width=1920, height=1080, fps=30).render("film.mp4")
```

Angles are radians; coordinates and radii are ångströms. `smooth` (quintic easing)
is the default for motion/morphs. `PlayTrajectory` defaults to `linear` so playback
does not change speed accidentally. Pass `rate_func=...` to `play()` to override all
animations in a clip, except `BackboneMorph`, which uses `motion_easing` to preserve
its per-residue delays in seconds. All animations in a `play` call run concurrently for `run_time`.
Animations in consecutive calls are sequential. Combine movement and shape animation
in one call; two concurrent writers to the same property are rejected.

## Representations

```python
p.cartoon(color="secondary")  # helix ribbons, sheet arrows, tubular coils
p.ribbon(color="rainbow", width=1.05)
p.ball_and_stick(atom_scale=0.30, bond_radius=0.14)
p.cartoon(color="chain")  # or a #RRGGBB color
p.shift((2, 0, 0)).rotate(0.5).scale(1.2)
self.play(p.animate.rotate(2).scale(1.1), run_time=3)
self.camera.depth_cue = 0.65  # 0 disables distance fog; default .65
```

Configure topology, representation colors, radii, and camera framing before the first
`play`/`wait`. Use animations for changes on the timeline. `Representation` smoothly
changes cartoon/ribbon cross sections and smoothly blends transitions to atoms.
Fades use weighted blended order-independent transparency, with no screen-door
dithering. Opaque surfaces write depth first; translucent surfaces accumulate color
and background transmittance without writing depth. Composition happens per MSAA
sample to preserve edges and opaque occlusion. Transparent depth ordering is
approximated, so dense overlaps may differ from exact sorted alpha compositing.
Ball-and-stick uses element colors. Cartoon/ribbon colors support secondary structure,
chain, rainbow, and a fixed color.

Camera framing is explicit; optional region tracking follows a moving center. Use
`self.camera.frame(p, margin=1.25, aspect=self.width/self.height)` for portrait renders.
Object rotations are around the molecular centroid; camera orbits are around its target.
