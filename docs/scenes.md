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

Angles are in radians. Coordinates and radii are in ångströms. Motion and morphs use quintic `smooth` easing by default. `PlayTrajectory` uses `linear` for a constant playback rate.

Set `rate_func` on `play()` to change easing for the call. `BackboneMorph` uses `motion_easing` instead, so residue delays keep their duration in seconds.

Animations in one `play()` call run together for `run_time` seconds. Consecutive calls run in sequence. You can combine movement with a shape change. Two concurrent animations that change the same property raise an error.

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

Set topology, colors, radii, and camera framing before the first `play()` or `wait()`. Use animations for later changes. `Representation` changes the cartoon or ribbon shape and crossfades when switching to atoms or a surface.

Fades use weighted blended transparency. Overlapping transparent objects have approximate depth ordering; see [rendering](rendering.md) for details. Ball-and-stick models use element colors. Cartoon and ribbon models accept secondary-structure, chain, rainbow, or fixed colors.

Set the camera frame explicitly. For portrait output, include the scene aspect ratio: `self.camera.frame(p, margin=1.25, aspect=self.width/self.height)`. Region tracking can keep a moving selection centered.

Protein rotations use the molecular centroid. Camera orbits use the camera target.