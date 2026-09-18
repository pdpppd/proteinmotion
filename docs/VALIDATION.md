# Validation and performance

Tested on 16–17 September 2026 on the local **Apple M3 Max, 40 GPU cores, 64 GB RAM**,
macOS 26.6.2, arm64 Python 3.12.8. The native adapter reports `backend_type=Metal`.
wgpu 0.32.0, Gemmi 0.7.5, PyAV 18.1.0, NumPy 2.5.3, and MDAnalysis 2.10.0 were used.

## Version 0.7: package installation and AI agent skill

**107 local tests pass.** Six new CPU cases cover starter movies with paths containing spaces, rendering-scene construction outside the checkout, overwrite protection, skill installation under a configured Codex directory, idempotent installation, explicit updates preserving unrelated files, symlink rejection, and CLI entry points.

The wheel built from the source distribution was installed into a fresh environment with `md` and `preview` extras. An isolated Python process outside the checkout verified the installed module location, all shader/font/license resources, and the complete bundled skill. The installed CLI created a self-contained movie project and installed the skill to a temporary destination. A 60-frame, 640×360 native Metal/VideoToolbox movie rendered from the installed package and every frame decoded successfully. CI also runs the installed-resource and CLI checks.

The `proteinmotion-movies` skill passed the skill-creator validator and an independent authoring trial: a 10-second 640×360, 30 fps ubiquitin movie with residue selection, a representation transition and a Cα distance ruler. All 300 frames decoded. The trial identified a sizing ambiguity; the skill now explicitly documents 1080p design pixels for text, label offsets and 2D line widths. [Skill guide](agent-skill.md) · [Install the release](getting-started.md).

## Version 0.6.3: continuous feature tour and stable fade endpoints

The calmodulin and troponin C feature demo uses one camera and a single timeline: calmodulin stays on screen until the contact-guided morph, and the actual troponin C destination carries the subsequent interaction demonstrations. All **55 animation boundaries** were checked for continuity of camera, visible atom coordinates, opacity, color and representation. Frames immediately before/after the styling-to-focus boundary at 21 seconds are pixel-identical at a ±1 μs interval. Backward seeking reproduces the same pixels, the troponin C pose is preserved after the morph, and the first/last frames meet on the same background for a gentle loop.

The GPU quintic easing function now mirrors its upper half, like the CPU implementation. This avoids floating-point cancellation near opacity 1 that could send apparently opaque cartoon fragments through the transparency pass and cause a brightness pop. Four new native GPU cases cover cartoon and ball-and-stick fades at 1×/4× MSAA and nonzero timeline offsets. All four fail against the previous shader and pass with the fix. **101 local tests pass.**

The continuous film contains **6,030 frames / 100.5 seconds** at 1920×1080, 60 fps and 4× MSAA. It exported in **23.54 seconds (256.2 fps)** using Metal and VideoToolbox on the local M3 Max. This is one end-to-end run including renderer initialization, surfaces, readback and hardware encoding; caches may be warm. Its content differs from the earlier videos, so the timings describe separate workloads. The 720p web preview retains **60 fps**. All frames of both encodes decoded successfully; representative frames and the transition around 20 seconds were inspected.

The MD-reader demonstration now uses 25 deposited **1CFC calmodulin NMR conformers**, mapped by exact identity to all **1,130 protein heavy atoms** in the displayed 1CLL topology. Six nonprotein atoms are hidden during playback. Atom identities and every XTC frame were compared with the mapped source, with maximum coordinate error **0.00501 Å** (XTC quantization). The video interpolates slowly through states 1–3 to display NMR structural variation. [Film, chapter guide and source](showcase.md) · [Measured render, continuity and input report](showcase-report.json).

## Version 0.6.2: clean ball-and-stick fades

Covalent cylinders are clipped at the analytic surfaces of their endpoint spheres in both opaque and transparent passes. The clipping uses the current GPU-interpolated positions and transformed atom radii. It removes internal sticks that became visible when the alpha-helix zoom started fading residues; bond-only annotation rulers keep their full endpoints.

**97 local tests pass.** Six added native GPU cases (1× and 4× MSAA) compare transparent atom cores with and without attached bonds, cover unequal atom sizes, transforms, global and residue opacity, overlapping/coincident endpoints, and ensure bond-only annotations retain their full lengths. The shaft between atoms remains visible.

The alpha-helix film and web preview were re-rendered. The 1,381-frame 1080p/60 fps export took **7.12 seconds (194.0 fps)** on the local M3 Max using Metal and VideoToolbox; all full-resolution and preview frames decoded successfully. Frames around the 8.4-second fade were visually inspected. This is a single local run with potentially warm shader caches. Earlier timings used separate runs. [Current render report](alpha-helix-hbonds-report.json).

## Version 0.6.1: an alpha-helix hydrogen-bond diagnostic

A fixed idealized 16-residue backbone with explicit amide H recovers exactly **12 of 12 expected O(i)···H–N(i+4) contacts**, using the geometry criteria alone. The new `endpoints="hydrogen_acceptor"` option draws H···A and optionally labels that distance; the default donor–acceptor display remains available. The example separates the full network from an annotated single-bond close-up with distance labels beside the short dashed bonds.

Four added tests bring the suite to **91 passing tests**. They independently check the fixture's φ/ψ angles, the complete pair set, a reversed-H negative control, an extended-backbone negative control, explicit/virtual endpoints, transformed anchors, native rendering and backward seeking. The existing 1UBQ test now checks six helix contacts at 150° and all eight at a declared 140° cutoff, documenting the two borderline angles rather than forcing them into the result.

The 1080p/60 fps Metal/VideoToolbox film contains **1,381 frames** (23.02 seconds encoded); every frame decoded successfully. The original v0.6.1 export took **19.48 seconds**, with other work running on the same machine. [Test guide](alpha-helix.md) · [Measured pairs](alpha-helix-hbonds.csv) · [Render and geometry report](alpha-helix-hbonds-report.json).

## Version 0.6: residue styling, surfaces and interactions

The package now has **87 passing local tests**. Twenty added cases cover independent residue color/opacity tracks, stagger timing and backward seeking, all four GPU representations, closed single-atom vdW/SAS/SES geometry and outward normals, surface caching/rebuilding, Coulomb energy/potential against analytic references, strict PQR charge mapping, hydrogen-bond distance/angle rejection, live ruler units and anchors, real NMR virtual backbone H, 3D occlusion versus 2D overlays at 1×/4× MSAA, and bounded interaction rendering through state changes.

Two films use **1920 × 1080, 60 fps, 4× MSAA, Metal and h264_videotoolbox**:

| Example | Frames / duration | Export time | Throughput | Scene construction |
|---|---:|---:|---:|---:|
| Residue styling and rebuilt SES, 2K39 | 1,488 / 24.8 s | 33.695 s | **44.2 fps** | 0.782 s |
| Hydrogen bonds, distances and charge contacts, 2K39 | 1,212 / 20.2 s | 8.455 s | **143.3 fps** | 0.970 s |

These are single local end-to-end runs, including renderer setup, surface construction where needed, readback and hardware encoding, excluding scene construction/loading. Caches may be warm, and other processes were running on the machine. All **2,700 encoded frames** decoded with the expected H.264 codec, dimensions, rate and frame count. Representative encoded color, transparency, surface and ruler frames were visually inspected. [Raw timings](molecular-tools-benchmark.json) · [Video checks](molecular-tools-video-verification.json).

The surface uses a 1.4 Å probe and 0.5 Å voxel spacing. It rebuilds during coordinate changes, which dominates this export; static surfaces reuse GPU triangles for rotation and styling. The alternative fast deformation mode was visually unsuitable for large NMR conformer changes and is documented for small-displacement previews only. The default remains `update="rebuild"`.

Hydrogen-bond examples use explicitly flagged virtual backbone H with D–A ≤ 3.5 Å and D–H–A ≥ 150°. Electrostatic examples use illustrative formal side-chain charges, dielectric 80 and screening length 8 Å. These checks compare the implemented geometry and formulas with their expected values and verify the rendered output. [Methods and limits](interactions.md) · [Complete film source](molecular-example.md).

Scikit-image 0.26.0 was used for marching cubes. Its array-shape assignments emit NumPy 2.5 deprecation warnings in the mesh tests (55 warnings), alongside the two existing MDAnalysis warnings described below; all numerical and rendering assertions pass.

## Version 0.5: text animation and labels

Native text uses HarfBuzz shaping, FontTools outlines and cached triangle/stroke geometry. `Write` draws glyph contours and then fills them with Manim-style lagged timing; `Unwrite` reverses it. Callout and amino-acid label anchors follow selected coordinates through camera motion, deformation, NMR state interpolation and transforms.

Both new films use **1920 × 1080, 60 fps, 4× MSAA, Metal and h264_videotoolbox**:

| Example | Frames / duration | Export time | Throughput |
|---|---:|---:|---:|
| Protein labels and callouts, 2K39 | 1,128 / 18.8 s | 3.742 s | **301.5 fps** |
| Close-up vector writing study | 600 / 10 s | 1.576 s | **380.7 fps** |

These are single local runs including initialization, rendering, readback and hardware encoding, excluding scene construction/font layout; shader caches may be warm. All **1,728 encoded frames** decoded successfully. Representative decoded frames were visually inspected. [Raw export measurements](text-render-report.json) and [video verification](text-video-verification.json).

Eleven new tests cover kerning/ligatures, Greek glyphs, empty text, contour holes/islands, automatic author-number labels, insertion codes, configurable writing/erasing, timeline conflicts, live anchors, viewport clipping, opacity inheritance, label placement and scaling. Native GPU checks run at 1×/4× sampling and verify contour-to-fill progression, glyph staggering/reversal, open letter counters, continuous fades, transparency composition, reproducible backward seeking and unchanged glyph buffers across frames.

Text is drawn over the protein image. The [text guide](text.md) describes supported fonts, placement options, and current limitations.

## Version 0.4: regions and NMR playback

The new example loads [PDB 2K39](https://www.rcsb.org/structure/2K39), an RDC-derived
solution NMR ensemble of ubiquitin. All **116 deposited models** have matching
selected atom identities: **602 heavy atoms, 76 residues, chain A**. Conformers are
rigidly aligned to the first model on the Cα atoms of residues 1–70. The original
mmCIF is included, with its checksum and derived coordinate summary in the
[ensemble report](nmr-ensemble-report.json).

The overview movies visit all 116 models in deposited order, using quintic easing
between adjacent states to show structural variation. The region tour
uses subsets of those states, first focusing on residues 23–34 with gold sphere/box
annotations, then following residues 71–76 with cyan halos and a box.

All three movies use **1920 × 1080, 60 fps, 4× MSAA**, native Metal, and
`h264_videotoolbox` with software encoding disabled:

| Example | Frames / movie duration | Export time | Export throughput | Scene construction |
|---|---:|---:|---:|---:|
| Region focus/highlight tour | 1,446 / 24.1 s | 4.521 s | **319.8 fps** | 0.390 s |
| All 116 states, cartoon | 1,440 / 24 s | 3.694 s | **389.8 fps** | 0.409 s |
| All 116 states, ball-and-stick | 1,440 / 24 s | 3.680 s | **391.3 fps** | 0.386 s |

These are individual local runs including renderer initialization, rendering,
readback, encoding, and finalization, but excluding scene construction/loading.
Driver/shader caches may be warm. Every encoded frame (**4,326 total**) decoded
successfully, with the expected dimensions, frame counts, H.264 codec, and 60 fps.
Decoded helix/tail close-ups and ensemble frames were visually inspected.
See [timings](nmr-region-benchmark.json) and [video checks](nmr-video-verification.json).

New tests cover PDB-number selections, unions, invalid/empty selections, all three
highlight styles through coordinate and transform changes, focus tracking with
constant zoom, custom field of view, concurrent molecular motion, portrait framing,
and backward seeking. Native GPU tests compare repeat renders after deformation
and seeking. The renderer retains references to cached CPU arrays so Python object
ID reuse cannot incorrectly skip a changed coordinate/metadata upload. Caching
remains bounded. A fractional timeline-boundary regression checks quintic easing
stays within [0, 1]. Real NMR tests verify every model is finite, selected exact
model endpoints are preserved, and decoded-frame caching remains bounded.

## Version 0.3: smooth fades

Screen-door dithering has been removed. Fades use weighted blended order-independent
transparency with an opaque depth pass, floating-point accumulation/transmittance
targets, and a per-MSAA-sample composite. Opaque-only frames retain the single-pass
path; extra targets and pipelines are allocated lazily on the first fade.

The calmodulin/troponin C examples were re-rendered at **1920 × 1080, 60 fps, 4× MSAA**.
Each contains 690 frames / 11.5 s. Native Metal plus VideoToolbox export measured:

| Representation | Export time | Export throughput | Scene construction |
|---|---:|---:|---:|
| Ball-and-stick | 2.647 s | **260.7 fps** | 0.024 s |
| Cartoon | 1.841 s | **374.8 fps** | 0.022 s |

These are individual local runs, including renderer initialization and encoding,
excluding loading/matching. Shader and driver caches may already be warm. Matching reuses the saved JSON.
Every encoded frame in both movies decoded successfully. Representative decoded
fade frames were inspected. See [raw timing data](smooth-transparency-benchmark.json)
and [video checks](smooth-transparency-video-verification.json).

New numerical GPU tests check five opacity levels against continuous alpha blending,
opaque occlusion of a transparent atom, reversed transparent-object draw order, and
single-surface bond fading. All run with both 1× and 4× sampling. The opaque fast path
is checked before transparency allocation, and returning to full opacity reproduces
the original frame. Transparent depth/color ordering remains an approximation;
overlapping fragments use weighted blending.

## Version 0.2: large structures and protein morphs

All three new movies were rendered at **1920 × 1080, 60 fps, 4× MSAA** through
native Metal and `h264_videotoolbox` with software encoding disabled:

| Workload | Selected heavy atoms | Cα residues | Video length | Export time | Export throughput |
|---|---:|---:|---:|---:|---:|
| Calmodulin → troponin C | 2,414 across both inputs | 144 → 162 | 11.5 s / 690 frames | 2.863 s | **241.0 fps** |
| GroEL chain A | 3,836 | 524 | 6 s / 360 frames | 1.216 s | **296.2 fps** |
| Full GroEL/GroES, 21 chains | 58,870 | 8,015 | 9 s / 540 frames | 2.791 s | **193.5 fps** |

These export times include renderer initialization, initial GPU geometry preparation,
rendering, full-frame readback, hardware encoding, and file finalization. They exclude
scene construction and loading. Construction took 0.021 s, 0.072 s, and 4.222 s,
respectively; the GroEL complex preparation includes bond inference and orientation.
The morph uses a saved correspondence, so matching is also excluded from export.
Each measurement comes from a single local run.

The GroEL/GroES example uses actual [1AON](https://www.rcsb.org/structure/1AON)
coordinates and changes from a rotating cartoon to ball-and-stick. Counts exclude
water and hydrogens. The 8,015 Cα atoms correspond to residues with coordinates in the deposited model.

The morph uses [1CLL](https://www.rcsb.org/structure/1CLL) chain A and
[1NCX](https://www.rcsb.org/structure/1NCX) chain A. The saved mapping contains
**114 monotone residue pairs**, covering 79.17% of the source and 70.37% of the target.
Contact parameters: cutoff 8 Å, softness 1.5 Å, maximum pairwise error 0.30.
Observed RMS error: **0.0274571**; maximum: **0.2936199**. A proper rigid fit has
Cα RMSD 7.0586 Å. The contact match therefore includes differences in domain arrangement.

The search considered 3,611 candidate pairings and stopped at its 15 s search budget
(16.354 s including seed/graph preparation). **The search returned a feasible match; optimality remains unproved.**
The known feasible count is 114; the conservative global count upper bound is 144.
The 114 pairs are reproducible from the saved JSON rather than depending on wall-clock
search termination. At a configurable 25 ms residue delay, motion proceeds from N to C;
30 unmatched source residues fade out and 48 unmatched target residues fade in.
The morph lasts 6 s, with a 3.175 s movement interval per selected residue.

All **1,590 frames** across the three encoded outputs were decoded successfully;
frame count, dimensions, H.264 codec, and 60 fps metadata match the requested outputs.
Representative decoded frames and the contact-map figure were visually inspected.
Raw records: [timings](large-morph-benchmark.json), [video checks](video-verification.json).
Reproduce the scenes using `examples/large_protein.py` and `examples/backbone_morph.py`;
`examples/contact_report.py` exports a contact-map PDF/PNG and residue-pair CSV.

## Version 0.2.1: calmodulin ball-and-stick morph

`BallAndStickDemo` reuses the 114 saved Cα pairs, 25 ms delay, fade intervals, camera,
and 11.5 s timeline from the cartoon example. Both endpoints use element-colored
ball-and-stick, including their different side-chain atom sets. Bonds disappear with
their less-visible endpoint. The original cartoon example remains available.

The **690-frame, 1920 × 1080, 60 fps** H.264 export took **2.626 s (262.7 fps)** on
the same M3 Max/Metal/VideoToolbox backend, with scene construction taking another
0.022 s. Matching is reused from JSON. This is one local measurement. Every encoded frame decoded successfully, and dimensions,
frame rate and frame count match the request. A decoded midpoint frame was inspected.
Raw records: [timing](ball-and-stick-morph-benchmark.json) and
[video validation](ball-and-stick-video-verification.json).

## Original v0.1 benchmark results

At 1920 × 1080 with 4× MSAA:

| Workload | Atoms | Throughput |
|---|---:|---:|
| Warm cartoon rotation, render only | 602 | 1,222 fps |
| Warm ball-and-stick rotation, render only | 602 | 1,369 fps |
| Warm cartoon rotation, replicated scale test | 60,200 | 883 fps |
| Warm ball-and-stick rotation, replicated scale test | 60,200 | 188 fps |
| Complete showcase export, including encoding | 602 | **316 fps** |

The original demo contains **1,230 frames / 20.5 seconds at 60 fps**, H.264,
1920 × 1080, limited-range BT.709. Export took **3.898 seconds** using
`h264_videotoolbox` with software encoding disabled. The scene includes all three
representations, rotations, camera movement, easing, morphing, deformation, and a
synthetic 61-state trajectory.

The render-only tests measure 180 frames after shader compilation and initial uploads;
they include Python submission and synchronize completion with a final one-pixel
readback. They exclude full-frame readback, encoding, file loading, and initial geometry
preparation. The scale test repeats ubiquitin in a spatial grid inside one GPU object. It measures rendering with duplicated geometry. Cartoon rendering draws the resulting 7,600-residue backbone.

These are individual measurements on a machine with other workloads. Geometry, camera framing, image coverage, frame decoding, and system load affect the results. [Raw timing data](benchmark.json) and `examples/benchmark.py` are included.

## Export optimization

An early RGBA-to-FFmpeg subprocess implementation measured only 21.4 fps for the
same output dimensions. Profiling separated rendering/readback from encoding and IPC.
The current exporter:

1. Renders through native Metal.
2. Converts the resolved RGBA texture to NV12 in a Metal compute shader.
3. Reads back through three reusable staging buffers.
4. Passes NV12 directly to PyAV's VideoToolbox encoder in the same process.

This reduces readback from 4 to 1.5 bytes per pixel and removes CPU color conversion
and the raw-video pipe. It still copies data from mapped staging memory into the
encoder. An IOSurface/CVPixelBuffer bridge could remove those copies in a future implementation.

## Verification

`pytest`: **107 passed** for v0.7.0, including 101 from v0.6.3 and 97 from v0.6.2. Coverage includes:

- Real ubiquitin mmCIF loading, helix/sheet annotations, alternate-location selection,
  multi-model PDB loading, model identity mismatch rejection, and chain gap splitting.
- Rigid alignment precision and chirality, reordered atom correspondence for morphs,
  deformation endpoints, representation blending, easing endpoints, and conflicting
  animation channels.
- Deterministic forward/backward seeking, concurrent rotation and coordinate motion,
  memory-mapped trajectory units, reverse playback, and bounded frame caching.
- Writing and reading actual DCD and XTC files via MDAnalysis (coordinates generated
  for the test). The tests verify atom counts, ångström units, stride, and alignment.
- Native Metal rendering of cartoon, ribbon, and ball-and-stick; visible foreground,
  distinct representations, reproducible frames, and ordered triple-buffer readback.
- GPU NV12 conversion against a numerical BT.709 reference, including widths that
  require packed-word padding.
- H.264 hardware, HEVC hardware, and explicit libx264 fallback exports; frame counts,
  dimensions, codec and color metadata; preservation of existing output on failure.
- Interactive preview orbit/zoom/seek controls. A native preview window was opened,
  rendered, and closed successfully in a smoke test.
- Contact-matching cardinality and secondary error objective checked against an
  independent exhaustive enumerator on three small random systems; rigid-invariant
  matching with an insertion, proof flags, correspondence serialization, and chirality.
- N-to-C delay in seconds, per-residue endpoint accuracy with nonidentity transforms,
  complementary visibility, unmatched fade intervals, backward seeking, validation
  of timing/crossing correspondences, and subsequent ordinary morph behavior.
- Native GPU morph endpoint coverage and replay; stable endpoint/control buffer keys
  during staggered playback, confirming that frames reuse the geometry buffers.
- Real calmodulin/troponin C ball-and-stick morphs: whole-residue translations preserve
  internal atom offsets, all atoms inherit their residue's motion/fade timing, Cα paths
  coincide, and GPU endpoint images and backward seeks reproduce exactly.
- A GPU regression verifies that hiding one atom removes its whole connecting bond.

Two expected upstream MDAnalysis warnings occur: the announced DCD timestep behavior
change and a missing time-step interval in the generated XTC fixture. Playback uses
explicit scene duration, so neither changes the coordinate/frame checks.

Ruff checks and formatting pass. Source distribution and wheel builds are included;
the packaged shaders are exercised by a wheel-install smoke check. The rendered
gallery and representative encoded video frames were visually inspected.

## Current limitations

Rendering and hardware export are tested on Apple silicon Macs. Other platform configurations remain untested. Backbone orientation guides are built on the CPU for each new state; large trajectories can be limited by this work or frame decoding.

Morphs interpolate coordinates, with residue order preserved in contact-guided matches. Bond lengths and atomic clashes can change during a morph. Ball-and-stick morphs move each residue as a group and crossfade the two atom sets. The matching report records whether the search completed and proved optimality.

Secondary structure stays fixed during playback. Prepare periodic boundaries before loading trajectories. Bond connectivity is estimated from geometry. Transparency uses approximate weighted blending.

See [rendering](rendering.md), [morphing](morphing.md), and [trajectory preparation](trajectories.md) for the method details and supported inputs.

## EEVEE and lens focus (v0.8.0)

The full test suite passed 116 tests on Apple M3 Max, including native GPU tests and the optional EEVEE integration test. The EEVEE run used Blender 5.2 with Metal. CPU tests cover residue selectors, focus tracking, eased focus pulls with concurrent motion, arbitrary timeline seeks, and mesh export for all four representations.

The Blender test renders actual frames. It checks matching stationary frames, visible lighting, focus changes, annotation composition, camera projection alignment, representation changes, coordinate changes, surface rendering, empty scenes, and opacity-layer errors. Run it with `PROTEINMOTION_TEST_EEVEE=1 pytest`; Blender must be installed. Ordinary CI runs the CPU tests.

The [EEVEE example](eevee.md) renders an eight-second calmodulin sequence at 960×540 and 60 fps. All 480 frames are decoded after export. The source uses PDB 1CLL, chain A, and highlights residues 5–19. It includes a focus pull to residues 82–92 and back. The downloadable wheel is also checked outside the source checkout, including a Blender still render.

## Calmodulin film and atom shading

[Calmodulin in focus](calmodulin-in-focus.md) was rendered with Blender 5.2 LTS on Metal at 1920×1080 and 60 fps. The 68-second scene uses 64 samples, 1.25× spatial supersampling, and a 28-pixel blur limit. All 4,080 frames in both the original export and the compressed web copy were decoded; dimensions and every presentation timestamp were checked. The camera and representation transitions were reviewed in rendered frames and motion samples.

The atomic close-up exposed reversed triangle winding in the EEVEE atom sphere mesh. The body triangles now face outward, matching their radial shading normals. A regression test checks the direction of every sphere face. The EEVEE tests passed with the Blender integration check enabled.
