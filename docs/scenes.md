# Scenes and motion

The rendered excerpts use [docs_examples.py](docs_examples.py). Its `ubiquitin()` helper loads PDB 1UBQ, and `frame()` adds the model and sets the camera. Run each excerpt inside `construct()`; the full script includes the imports and setup. Run the full script from a repository checkout, which includes the input structures.

```python output=motion
p = ubiquitin()
frame(self, p)
self.play(FadeIn(p), run_time=1)
self.play(Rotate(p, np.pi), run_time=3)
self.play(Representation(p, "ribbon"), run_time=1)
self.play(
    p.animate.shift((5, 0, 0)),
    self.camera.animate.orbit(theta=0.6),
    run_time=2,
)
self.wait(0.5)
```

Angles are in radians. Coordinates and radii are in ångströms. Motion and morphs use quintic `smooth` easing by default. `PlayTrajectory` uses `linear` for a constant playback rate.

Set `rate_func` on `play()` to change easing for the call. `BackboneMorph` uses `motion_easing` instead, so residue delays keep their duration in seconds.

Animations in one `play()` call run together for `run_time` seconds. Consecutive calls run in sequence. You can combine movement with a shape change. Two concurrent animations that change the same property raise an error.

## Representations

```python output=representations
p = ubiquitin().surface(resolution=0.7).cartoon()
frame(self, p)
self.wait(1)
for name in ("ribbon", "ball_and_stick", "surface"):
    self.play(Representation(p, name), run_time=1.2)
    self.play(Rotate(p, 0.45), run_time=1.5)
```

Set topology, colors, radii, and camera framing before the first `play()` or `wait()`. Use animations for later changes. `Representation` changes the cartoon or ribbon shape and crossfades when switching to atoms or a surface.

Fades use weighted blended transparency. Overlapping transparent objects have approximate depth ordering; see [rendering](rendering.md) for details. `ball_and_stick()` defaults to element colors. Cartoon and ribbon models accept secondary-structure, base, chain, rainbow, or fixed colors. `Representation` keeps the current palette and residue overrides during transitions. See [DNA and RNA](dna-rna.md) for nucleotide base styles and surfaces.

Set the camera frame explicitly. For portrait output, include the scene aspect ratio: `self.camera.frame(p, margin=1.25, aspect=self.width/self.height)`. Region tracking can keep a moving selection centered.

Protein rotations use the molecular centroid. Camera orbits use the camera target.
