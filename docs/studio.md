# Studio looks and film effects

The studio renderer adds camera-relative lighting, material presets, ambient occlusion,
molecular surfaces, and optional film effects. Set the look with one object:

```python
from proteinmotion import StudioLook

look = StudioLook(
    lighting="dramatic",
    material="medium",
    grain="fine",
    bloom="soft",
    halation="warm",
)

scene.render("film.mp4", renderer="studio", look=look)
scene.render_frame(12, output="frame.png", renderer="studio", look=look)
```

Create the scene at `width=3840, height=2160` for a native 4K image. Text and callouts
are drawn after the effects and stay sharp. Materials apply to the molecular scene;
lighting directions follow the camera during an orbit.

## Rendered examples

These 720p/60 fps clips use the scene's own renderer and look settings. Download
[the runnable scenes](studio_examples.py), [2DRI](data/2dri.cif),
and [1UBQ](data/1ubq.cif). Put the structures in a `data/` directory
beside the script; `DATA = Path(__file__).parent / "data"` in the examples below.
Studio is included in ProteinMotion 0.13.0 and later. Install or upgrade before
running these examples:

```sh
python -m pip install --upgrade "proteinmotion>=0.13.0"
```

### Thread the cartoon

The actual helix bands, beta arrows, and loop tubes travel into place with eased
motion. This is an illustrative entrance, not a folding simulation.

```python output=studio-threading
class CartoonThreading(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="clean")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.cartoon(color="#e8cda5", atom_scale=0.22, bond_radius=0.1)
        self.camera.frame(protein, margin=1.12, screen_position=(0.62, 0.53))
        self.add(Text("Cartoon threading", position=(0.05, 0.09), font_size=48))
        self.add(Text("2DRI · Eased entrance", position=(0.05, 0.16), font_size=26))
        self.play(Thread(protein, easing="smooth", swirl=0.25, glow=0, seed=4), run_time=6)
        self.play(self.camera.animate.orbit(0.3), run_time=2)
        self.wait(1)
```

### Thread atoms and bonds

The same motion also works with ball-and-stick geometry. Residues and bound ligands
move as rigid groups, and bonds between residues can stretch during the entrance.
Surface threading raises an error.

```python output=studio-atoms
class AtomThreading(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="glossy", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.ball_and_stick(atom_scale=0.2, bond_radius=0.1)
        self.camera.frame(protein, margin=1.12, screen_position=(0.62, 0.53))
        self.add(Text("Atom threading", position=(0.05, 0.09), font_size=48))
        self.add(Text("2DRI · Ball and stick", position=(0.05, 0.16), font_size=26))
        self.play(Thread(protein, easing="smooth", swirl=0.25, glow=0, seed=4), run_time=6)
        self.play(self.camera.animate.orbit(0.3), run_time=2)
        self.wait(1)
```

### Open a buried pocket

Switch representation with its settings in the same call. The cutaway exposes
bound ribose (RIP A272) in 2DRI. `shape="tunnel", rings=5` draws depth rings every
5 Å, with brighter marks every 10 Å. The tunnel stays fixed to the protein as the
camera moves. The surface uses a voxel approximation of SES;
cutting the display does not change the deposited atomic coordinates.

```python output=studio-cutaway
class BuriedLigand(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="dramatic", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.cartoon(color="#8fe3c0", atom_scale=0.3, bond_radius=0.12)
        ligand = protein.select(chain="A", resname="RIP", residues=272)
        ligand.filter(element="C").set_color("#ffc46b")
        self.camera.theta, self.camera.phi = 1.5, 0.25
        self.camera.frame(protein, margin=1.1, screen_position=(0.63, 0.53))
        self.add(protein)
        self.add(Text("Buried ribose", position=(0.05, 0.09), font_size=48))
        caption = Text("2DRI · Cartoon", position=(0.05, 0.16), font_size=26)
        self.add(caption)
        self.wait(1)
        self.play(
            Representation(protein, "surface", kind="ses", grid_spacing=0.6),
            caption.animate.set_text("Solvent-excluded surface"),
            run_time=1.5,
        )
        self.play(
            Reveal(self.camera, ligand, window=1.6, shape="tunnel", rings=5, padding=2, surface_keep=0.15),
            caption.animate.set_text("RIP A272 · Depth rings every 5 Å"),
            run_time=2,
        )
        self.play(self.camera.animate.orbit(0.25), run_time=2)
        self.play(Conceal(self.camera), Representation(protein, "cartoon"), run_time=1.5)
        self.play(caption.animate.set_text("2DRI · Cartoon"), run_time=0.5)
        self.wait(1)
```

## Lighting and materials

| Lighting | Appearance |
| --- | --- |
| `studio` | Warm key, cool fill, mint rim; the default |
| `soft` | Broad, even lighting with gentle contrast |
| `dramatic` | Directional warm key, little fill, strong cool rim |
| `flat` | Front lighting and strong ambient for low-contrast illustrations |

| Material | Appearance |
| --- | --- |
| `rough` | Matte, roughness 0.85, no clearcoat |
| `medium` | Satin, roughness 0.55, light clearcoat |
| `glossy` | Polished, roughness 0.38, strong clearcoat; the default |

For finer control, supply values such as `roughness=0.65`, `specular=0.7`,
`rim=2.0`, or `rim_color=(1.0, 0.5, 0.3)` to `StudioLook`. Colors accept RGB triples or hex strings.
Light strengths and material values are compiled into the scene shader when the
renderer is created; create a new renderer when changing those controls.

## Global recipes

```python
look = StudioLook(lighting="studio", material="glossy", effects="film")
```

| Recipe | Grain | Bloom | Halation |
| --- | ---: | ---: | ---: |
| `clean` | 0 | 0 | 0 |
| `subtle` | 0.014 | 0.20 | 0.12 |
| `film` | 0.040 | 0.35 | 0.40 |
| `dreamy` | 0.025 | 0.80 | 0.50 |

Recipes also set suitable grain sizes, bloom radii, and halation thresholds/radii.
With no `effects` argument, grain and halation are off and bloom retains the
lighting preset's original value (0.35 for `studio`). Ambient occlusion is controlled
separately with `ambient_occlusion` from 0 to 1; `clean` disables the three film effects.

## Individual effects

Each effect accepts a preset name or a numeric strength. Zero or `"off"` disables it.

| Effect | Presets | Additional controls |
| --- | --- | --- |
| `grain` | `off`, `fine`, `film`, `coarse` | `grain_size`, `grain_seed`, `grain_fps` |
| `bloom` | `off`, `soft`, `strong` | `bloom_threshold`, `bloom_radius` |
| `halation` | `off`, `subtle`, `warm` | `halation_threshold`, `halation_radius`, `halation_color` |

```python
look = StudioLook(
    effects="film",
    grain=0.025,          # Numeric strength overrides the recipe.
    grain_size=1.2,       # 1080p design pixels; scales to 2.4 pixels at 4K.
    grain_seed=42,
    grain_fps=24,        # New pattern 24 times per second.
    bloom="soft",        # An individual preset overrides the global recipe.
    bloom_threshold=0.9, # This explicit value overrides the soft preset.
    halation="off",
)
```

**Grain** adds monochrome, centered noise after highlight roll-off. Its amplitude
depends on the pixel's luminance: midtones receive more texture; black and white
receive none. Its seed combines `grain_seed` with the scene time at `grain_fps`.
Seeking to the same time produces the same grain, regardless of render order.

**Bloom** spreads bright highlights through a multiscale blur. `bloom_threshold`
uses the renderer's display-referred HDR values, where 1 is display white.
`bloom_radius` scales the spread (default 1). Bloom keeps the source light's color.

**Halation** creates a local red/amber fringe around bright boundaries. A blurred
highlight mask minus the original mask avoids tinting uniform bright areas.
`halation_radius` is the blur sigma in 1080p design pixels and scales with output
height. The default `halation_color` is `(1.0, 0.22, 0.045)`. Halation is an artistic
screen-space approximation, not a physical film-stock model.

Effects run once on the combined image after transparency layers have been blended.
Bloom and halation are extracted independently, before either is added to the image.
They do not require each other to be enabled. Grain runs last, before the text overlay.

For many stills, reuse a renderer and let the scene supply the current time:

```python
from proteinmotion.studio import StudioRenderer

with StudioRenderer(3840, 2160, msaa=4, look=look) as renderer:
    for time in (2, 5, 8):
        scene.render_frame(time, renderer=renderer, output=f"frame-{time}.png")
```

When calling `renderer.render(...)` or `renderer.enqueue(...)` directly, call
`renderer.set_time(seconds)` first to animate grain. The default time is zero.
Pass `look` when constructing an existing renderer; the scene rejects a second
look alongside a renderer instance to avoid silently ignoring settings.

## Command line and examples

```bash
proteinmotion still examples/studio_buried_ligand.py BuriedRibose \
  --time 19.8 --renderer studio --lighting dramatic --material medium \
  --effects film --grain 0.025 --halation warm \
  --width 3840 --height 2160 -o ribose-4k.png
```

The same flags work with `proteinmotion render`. Preset flags require
`--renderer studio`. Invalid names, non-finite strengths, and invalid sizes fail
before GPU allocation.

Run `python examples/studio_presets.py` to render 14 full-resolution 4K examples
and four comparison sheets into `output/studio-presets-4k/`. The gallery keeps
the same 2DRI coordinates, camera, and cutaway while varying the look, and saves
every resolved setting in `manifest.json`.

## Molecular surfaces

Studio now uses the same molecular surface builder as the native renderer by
default, with Studio lighting, materials, ambient occlusion, and film effects.
The existing surface API works with either renderer:

```python
protein.surface(kind="vdw")  # Union of atom spheres with element-specific VDW radii.
protein.surface(kind="sas", probe_radius=1.4)  # Atom radii expanded by the solvent probe.
protein.surface(kind="ses", probe_radius=1.4, resolution=0.5)  # Solvent-excluded surface.
scene.render_frame(output="surface.png", renderer="studio")
```

These are alternative settings; each `surface()` call replaces the previous one.
Radii, probe size, and resolution are in ångströms. VDW uses Gemmi's element radii
(minimum 1 Å) and ignores `probe_radius`. The default is `kind="ses"`, a 1.4 Å
probe, and 0.7 Å grid spacing. SES is a voxel approximation using cavity filling
and distance-transform erosion, not an analytical solvent-excluded surface.
Lower `resolution` values give finer meshes at higher memory and computation cost.
`max_voxels` limits grid allocation (default 8,000,000); exceeding it raises an
error rather than silently reducing resolution.

The mesh includes every atom present in the Protein, including ligands, ions,
and any loaded water. Loading with `include_water=False` (the default) omits
waters. Colors, residue opacity, representation transitions,
and `Reveal` cutaways use the shared mesh shaders with Studio shading.

Meshes are generated on the CPU and drawn on the GPU. The default
`update="rebuild"` rebuilds when coordinates change, so moving surfaces can be
expensive. `update="deform"` reuses a reference mesh and deforms it on the GPU:

```python
protein.surface(kind="ses", probe_radius=1.4, resolution=0.7, update="deform")
```

Use deformation for small motions; it does not recompute the molecular boundary
and can fold or tear for large structural changes. Rigid transforms and styling
changes reuse geometry in both modes. See [surface settings](styling.md).

### Original density appearance

The original smooth GPU density surface remains available explicitly:

```python
look = StudioLook(
    surface_mode="density",
    surface_inflation=0.45,
    surface_blobbiness=1.6,
    surface_voxels=3_000_000,
)
scene.render_frame(output="density.png", renderer="studio", look=look)
```

Density mode blends Gaussian fields around polymer atoms and ray-marches them on
the GPU. It is an artistic surface, not VDW, SAS, or SES: `kind`, `probe_radius`,
`update`, and `max_voxels` do not control it. It uses `resolution` as the desired
grid spacing and can coarsen that spacing to fit `surface_voxels`. The inflation,
blobbiness, and voxel-budget controls apply only to density mode. Choose
`surface_mode` when creating the renderer; create a new renderer to change modes.

## Classic cartoons

Studio's default `cartoon_style="classic"` draws broad helix bands with rounded
edges, flat beta-strand arrows, and thinner round loop tubes. A cubic B-spline
smooths the displayed backbone; surface normals follow the curved geometry and
arrow shoulders. The sweep is evaluated from the current interpolated atom
positions on the GPU, including during trajectories and deformations.

```python
protein.cartoon(color="#e8cda5")
look = StudioLook(cartoon_style="classic", lighting="soft", material="medium", effects="clean")
scene.render_frame(output="cartoon.png", renderer="studio", look=look)
```

Use `cartoon_style="legacy"` to recover the original Studio elliptical cartoon
sweep. The native renderer retains its existing cartoon geometry. The separate
`protein.ribbon()` representation keeps its uniform-width profile, and nucleotide
backbones retain their tube dimensions. Classic mode uses denser sweep tessellation
(20 subdivisions per segment, 24 around the cross-section).

Cartoons follow the structure's helix/sheet assignments; files without those
annotations are assigned with DSSP. Cross-sections and sheet arrows blend when
the assignment changes, for example during `SetTorsions`.

In every renderer, ribbons and cartoons are oriented by the Cα trace: by the
local helix axis in helices and by the curvature of strands and loops, with the
carbonyl direction used only where the trace is straight. The orientation depends
only on nearby residues, so cartoons turn smoothly as atoms move in morphs,
trajectories, and torsion animations. Between two residues whose relative turn is
ambiguous (a few segments per structure, mostly in loops and at helix kinks), the
ribbon makes that turn through a short round section rather than picking a side,
so it cannot flip from one frame to the next. Smoothing moves only the display curve, not atomic coordinates,
side-chain geometry, interaction endpoints, or analysis data. This is a familiar
molecular cartoon style, not a reproduction of another program's renderer.
Choose the style when constructing a renderer, as with lighting and materials.

Run `python examples/studio_cartoons.py` for three 4K before/after plates and a
seven-second 60 fps comparison turntable in `output/studio-cartoons/`.

## Shared settings and preview

Define `renderer = "studio"` and `look = StudioLook(...)` on your scene class, or pass them to the scene constructor. `scene.preview()`, `render_frame()` and `render()` inherit both. CLI flags override only the values supplied. Studio is also available through `proteinmotion preview film.py Movie --renderer studio`.

Use `look.with_presets(material="rough", effects="clean")` or `dataclasses.replace()` to derive looks while preserving explicit overrides. Preset assignment re-resolves inherited fields. Light and halation colors accept hex strings as well as RGB triples. Create a new renderer after changing shader/geometry settings on an existing look; scene exports do this automatically.
