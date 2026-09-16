# Validation and performance

Tested on 16 September 2026 on the local **Apple M3 Max, 40 GPU cores, 64 GB RAM**,
macOS 26.6.2, arm64 Python 3.12.8. The native adapter reports `backend_type=Metal`.
wgpu 0.32.0, Gemmi 0.7.5, PyAV 18.1.0, NumPy 2.5.3, and MDAnalysis 2.10.0 were used.

## Version 0.5: vector writing and live labels

Native text uses HarfBuzz shaping, FontTools outlines and cached triangle/stroke geometry. `Write` draws glyph contours and then fills them with Manim-style lagged timing; `Unwrite` reverses it. Callout and amino-acid label anchors follow selected coordinates through camera motion, deformation, NMR state interpolation and transforms.

Both new films use **1920 × 1080, 60 fps, 4× MSAA, Metal and h264_videotoolbox**:

| Example | Frames / duration | Export time | Throughput |
|---|---:|---:|---:|
| Protein labels and callouts, 2K39 | 1,128 / 18.8 s | 3.742 s | **301.5 fps** |
| Close-up vector writing study | 600 / 10 s | 1.576 s | **380.7 fps** |

These are single local runs including initialization, rendering, readback and hardware encoding, excluding scene construction/font layout; shader caches may be warm. All **1,728 encoded frames** decoded successfully. Representative decoded frames were visually inspected. [Raw export measurements](text-render-report.json) and [video verification](text-video-verification.json).

Eleven new tests cover kerning/ligatures, Greek glyphs, empty text, contour holes/islands, automatic author-number labels, insertion codes, configurable writing/erasing, timeline conflicts, live anchors, viewport clipping, opacity inheritance, label placement and scaling. Native GPU checks run at 1×/4× sampling and verify contour-to-fill progression, glyph staggering/reversal, open letter counters, continuous fades, transparency composition, reproducible backward seeking and unchanged glyph buffers across frames.

Text is a screen overlay, not depth-occluded geometry. There is no LaTeX/MathTex, markup, font fallback or general collision-free placement. See the [text guide](text.md) for precise behavior and limitations.

## Version 0.4: regions and a real NMR ensemble

The new example loads [PDB 2K39](https://www.rcsb.org/structure/2K39), an RDC-derived
solution NMR ensemble of ubiquitin. All **116 deposited models** have matching
selected atom identities: **602 heavy atoms, 76 residues, chain A**. Conformers are
rigidly aligned to the first model on the Cα atoms of residues 1–70. The original
mmCIF is included, with its checksum and derived coordinate summary in the
[ensemble report](nmr-ensemble-report.json).

The overview movies visit all 116 models in deposited order, using quintic easing
between adjacent states. Deposited model order is not physical time; the transitions
are visual interpolations, not MD or inferred molecular pathways. The region tour
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
excluding loading/matching. Shader/driver caches may already be warm; these are not
cold-start or repeated statistical benchmarks. Matching reuses the saved JSON.
Every encoded frame in both movies decoded successfully. Representative decoded
fade frames were inspected. See [raw timing data](smooth-transparency-benchmark.json)
and [video checks](smooth-transparency-video-verification.json).

New numerical GPU tests check five opacity levels against continuous alpha blending,
opaque occlusion of a transparent atom, reversed transparent-object draw order, and
single-surface bond fading. All run with both 1× and 4× sampling. The opaque fast path
is checked before transparency allocation, and returning to full opacity reproduces
the original frame. Transparent depth/color ordering remains an approximation;
the renderer does not implement refraction or exact per-pixel fragment sorting.

## Version 0.2: real larger structures and different-protein morph

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
The figures are single local runs, not repeated or isolated peak benchmarks.

The GroEL/GroES example uses actual [1AON](https://www.rcsb.org/structure/1AON)
coordinates and changes from a rotating cartoon to ball-and-stick. Counts exclude
water and hydrogens. Missing/unresolved residues are not invented; 8,015 is the count
of modeled Cα atoms, not the total deposited sequence length.

The morph uses [1CLL](https://www.rcsb.org/structure/1CLL) chain A and
[1NCX](https://www.rcsb.org/structure/1NCX) chain A. The saved mapping contains
**114 monotone residue pairs**, covering 79.17% of the source and 70.37% of the target.
Contact parameters: cutoff 8 Å, softness 1.5 Å, maximum pairwise error 0.30.
Observed RMS error: **0.0274571**; maximum: **0.2936199**. A proper rigid fit has
Cα RMSD 7.0586 Å; contact similarity does not imply one rigid domain arrangement.

The search considered 3,611 candidate pairings and stopped at its 15 s search budget
(16.354 s including seed/graph preparation). **Global optimality was not proved.**
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
0.022 s. Matching is reused from JSON. This is one local measurement, not a repeated
performance estimate. Every encoded frame decoded successfully, and dimensions,
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

The delivered demo contains **1,230 frames / 20.5 seconds at 60 fps**, H.264,
1920 × 1080, limited-range BT.709. Export took **3.898 seconds** using
`h264_videotoolbox` with software encoding disabled. The scene includes all three
representations, rotations, camera movement, easing, morphing, deformation, and a
synthetic 61-state trajectory.

The render-only tests measure 180 frames after shader compilation and initial uploads;
they include Python submission and synchronize completion with a final one-pixel
readback. They exclude full-frame readback, encoding, file loading, and initial geometry
preparation. The scale test repeats ubiquitin in a spatial grid inside one GPU object;
it is not a biological assembly or a large MD trajectory benchmark. Cartoon rendering
draws the 7,600-residue backbone rather than all 60,200 atoms as spheres.

These are individual local measurements on a machine with other workloads, not
isolated, repeated statistical benchmarks. Geometry, camera framing, screen coverage,
disk/frame decoding, and system load change results. No universal or peak-performance
claim is made. [Raw timing data](benchmark.json) and `examples/benchmark.py` are included.

## Export optimization

An early RGBA-to-FFmpeg subprocess implementation measured only 21.4 fps for the
same output dimensions. Profiling separated rendering/readback from encoding and IPC.
The delivered exporter instead:

1. Renders through native Metal.
2. Converts the resolved RGBA texture to NV12 in a Metal compute shader.
3. Reads back through three reusable staging buffers.
4. Passes NV12 directly to PyAV's VideoToolbox encoder in the same process.

This reduces readback from 4 to 1.5 bytes per pixel and removes CPU color conversion
and the raw-video pipe. It still copies data from mapped staging memory into the
encoder; a future IOSurface/CVPixelBuffer bridge could remove those copies. There is
no claim that this is the fastest possible implementation of every workload.

## Verification

`pytest`: **67 passed** for v0.5. Coverage includes:

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
  during staggered playback, confirming geometry is not uploaded again each frame.
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

## Known limits

See the README for supported formats and API examples. This is a standalone v0.5
package. In particular, it does not implement physically constrained morphing,
automatic sequence alignment, per-frame DSSP, periodic unwrapping, bond-order chemistry,
shadows/SSAO, refractive/transmissive materials, or direct Manim Mobject integration.
Other operating systems and GPUs have not been verified. Cartoon orientation guides
are still constructed on the CPU once per loaded coordinate state; very large MD
systems may become bound by frame decoding or that preparation step.
Contact-guided morphs support cartoon/ribbon/ball-and-stick, one selected chain per endpoint, and
order-preserving correspondences. The matcher is bounded and does not promise a global
optimum on large cases. Fades use approximate blended transparency; intermediate backbone shapes are
visual interpolations without bond-length or collision constraints.
Ball-and-stick uses residue-wise translation and crossfades the endpoint atom sets;
there is no side-chain atom correspondence or physical all-atom transition model.
