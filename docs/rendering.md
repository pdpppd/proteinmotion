# Rendering

Python defines the scene and animation timeline. The GPU renders the molecular geometry, text, and annotations. wgpu-native compiles the same shaders for Metal on macOS and Vulkan on Windows/Linux. VideoToolbox and NVIDIA NVENC provide hardware video encoding.

## Native GPU rendering

Coordinate interpolation, backbone curves, ribbons, spheres, lighting, and depth cueing run on the GPU. Rotation and styling reuse geometry buffers and update small parameter buffers.

Residue color and opacity apply across representations. A separate pass draws text and callout lines over the protein image. Text geometry is cached after layout. Callout lines update to follow the selected coordinates. See [text and labels](text.md).

The renderer uses 4× multisample antialiasing (MSAA) to smooth geometry edges. Sphere silhouettes use analytical intersections and receive partial antialiasing. Current rendering limits include shadows, screen-space ambient occlusion, ray tracing, and refractive materials.

Since version 0.14.0, stills and movies up to 1920 × 1080 are also supersampled: the native and Studio renderers draw each frame at twice the width and height and average each 2 × 2 block of pixels in linear light. This smooths edges that MSAA cannot, such as thin loop tubes and callout lines, specular highlights, and the edges of Studio's ambient occlusion. Text, line widths, and effects keep their size because they are defined in 1080p design pixels. Set `ProteinScene(supersampling=1)` or `--supersampling 1` to turn it off, or 3 or 4 for still finer edges at a higher GPU cost; larger outputs default to 1. The interactive preview does not supersample.

## Studio GPU rendering

Use `renderer="studio"` with a `StudioLook` for four lighting presets, three material
presets, smooth helix bands and sheet-arrow cartoons, ambient occlusion, and molecular surfaces. Luminance-based film grain,
bloom, and warm halation can be selected individually or through a global recipe:

```python
from proteinmotion import StudioLook

scene.render("film.mp4", renderer="studio", look=StudioLook(
    lighting="dramatic", material="medium", effects="film",
))
```

See [studio looks and film effects](studio.md) for the presets, numeric controls,
4K stills, and command-line options. The native renderer's limitations above do not
include the studio backend's ambient occlusion and post-processing features.
Studio uses `cartoon_style="classic"` by default; `StudioLook(cartoon_style="legacy")`
retains its original elliptical cartoon sweep. See [classic cartoons](studio.md#classic-cartoons).

## Blender EEVEE

Use `--renderer eevee` or `scene.render("film.mp4", renderer="eevee")` to export through Blender. Install Blender 4.5+ separately; EEVEE is part of Blender. ProteinMotion starts one background Blender process per export and reuses it for all frames. macOS uses Metal.

The scene timeline evaluates positions, representations, colors, and selections. ProteinMotion exports cartoon, ribbon, sphere, bond, and surface meshes to Blender. EEVEE draws them with studio lighting and optional lens depth of field. The existing vector overlay renderer adds text, Write animations, callouts, and 2D measurements afterward. These overlays stay sharp.

`camera.set_focus()` sets lens focus. `FocusPull` animates it. The native renderer's `camera.focus()` and `scene.focus()` continue to frame or zoom to a selection. See [EEVEE and lens focus](eevee.md) for code and output.

EEVEE uses 64 samples and 1.5× image dimensions by default. It renders each frame independently with fixed sampling settings. Shadows, camera jitter, and screen-space ray tracing are disabled. Output is downsampled with a Lanczos filter. `camera.depth_cue` distance fog, the background color, cutaways, depth tunnels, and wire glow match the native renderer. `msaa` applies only to the native renderer, and EEVEE uses its own lighting, so materials look brighter and more saturated. EEVEE rounds opacities to 1/48, so a frame never needs more than 48 opacity layers.

### EEVEE transparency

EEVEE opacity uses a weighted combination of opaque render layers. Each layer includes geometry at or above one opacity threshold. For a fully visible helix and a context at 6% opacity, the result is 94% isolated helix plus 6% complete protein, mixed in scene-linear color. Each layer receives depth of field before composition.

This produces smooth group fades and reveals a selected region through faded foreground geometry. Geometry with the same opacity retains its opaque visibility ordering. It is useful for region highlights and representation transitions; it approximates translucent overlap. [Blender's post-process DOF has limitations with blended materials](https://docs.blender.org/manual/en/latest/render/eevee/limitations/limitations.html).

Render cost grows with the number of distinct opacity levels. Opacities are rounded to 1/48; backbone strips use the nearest residue's opacity. A frame therefore needs at most 48 layers, within the default `EEVEEOptions(max_opacity_layers=64)`; a lower limit raises an error when a frame exceeds it. Native rendering is a faster option for long trajectories with many independently fading residues.

## Native transparency

Opaque objects draw first and record their depth. Transparent objects then accumulate color and transmittance in floating-point buffers. A final GPU pass combines these buffers with the opaque image.

This is weighted blended order-independent transparency, based on [McGuire and Bavoil](https://www.jcgt.org/published/0002/02/09/). It approximates the ordering of overlapping transparent objects. The renderer allocates the extra buffers at the first fade and reuses them afterward.

Ball-and-stick bonds end at their atom surfaces. This hides the part of each cylinder inside an atom during both opaque rendering and fades. The clipping updates with atom positions and sizes. Distance and interaction lines keep their specified endpoints.

## Surfaces and interactions

Native and Studio share VDW, SAS, and approximate SES surface meshes, built on the CPU from voxel fields using distance transforms and marching cubes. Both honor element-specific VDW radii, probe radius, grid resolution, and the voxel allocation limit. The default `update="rebuild"` creates a new mesh when coordinates change. Rotation, color, and opacity changes reuse the mesh. Rebuilding a moving surface can take most of the export time.

The optional `update="deform"` mode moves a reference mesh on the GPU. Use it for small displacements; large changes can fold or tear the mesh. See [surface settings](styling.md).

Studio also retains its original artistic GPU density surface through
`StudioLook(surface_mode="density")`; see [Studio surface modes](studio.md#molecular-surfaces).

Hydrogen-bond and screened Coulomb calculations use CPU spatial queries. Results are cached until coordinates or analysis settings change. Their 3D lines are drawn with the molecular geometry; 2D lines and all distance text are drawn over the image. See [interaction methods](interactions.md).

## Coordinates and connectivity

GPU playback holds two coordinate states at a time. New states require a CPU update of backbone orientation guides. GPU memory use depends on atom count and image size, so trajectory length primarily affects loading and playback time.

Bond detection uses covalent radii and a spatial index when the structure loads. The current method estimates connectivity from geometry; bond order, aromaticity, and explicit topology bonds are outside its scope.

Ribbons split at chain changes, missing Cα atoms, and initial Cα gaps greater than 4.8 Å. Connectivity and secondary-structure assignments stay fixed during playback.

DNA/RNA backbones use C4′ anchors, with P as a fallback, and split at chain changes, missing anchors, and broken phosphodiester links. Base slabs and rings use cached meshes that follow three ring atoms; base sticks follow each bond endpoint. The native shader reads the current interpolated coordinates on the GPU. EEVEE evaluates the same geometry for each frame. See [DNA and RNA](dna-rna.md) for base templates and incomplete structures.

Morphs and state playback interpolate coordinates. They illustrate structural changes. Use simulation trajectories and an appropriate analysis method when interpreting molecular dynamics.

## Video export

Both renderers encode with PyAV. Automatic encoding prefers VideoToolbox on macOS and H.264 NVENC on Windows/Linux, with a warning and `libx264` fallback if hardware initialization fails. Explicit codecs fail visibly if unavailable. EEVEE reads completed RGBA frames from Blender before encoding. The native renderer uses the transfer path below.

A GPU compute pass converts each frame to limited-range BT.709 NV12. Three staging buffers overlap rendering and readback. NV12 transfers 1.5 bytes per pixel, compared with 4 for RGBA, reducing transfer size by 62.5%. PyAV passes NV12 to VideoToolbox or NVENC without converting it to planar YUV on the CPU. The path still maps and copies staging buffers; NVENC then uploads the frame. It does not share a zero-copy texture with the renderer.

NVENC uses preset P4, high-quality tuning, variable bitrate and a quality target of 18. `--bitrate` sets its target bitrate (20 Mbit/s by default); this is not a fixed file-size guarantee. H.264 is the automatic codec. Use `--codec hevc_nvenc` or `--codec av1_nvenc` for supported NVIDIA cards, or `--codec libx264` for software H.264. Software CRF and NVENC CQ values are different quality scales. [FFmpeg NVENC options](https://www.ffmpeg.org/doxygen/trunk/nvenc__h264_8c_source.html).

Large scenes can be limited by GPU buffer size, geometry and pixel processing, CPU frame decoding, or encoder speed. See [tests and benchmarks](VALIDATION.md) for measured results and their conditions.

## Platforms

Rendering and hardware export are tested on Apple silicon Macs and Windows 11 with an NVIDIA RTX 5070 Ti (driver 595.97, wgpu 0.32.0). macOS defaults to Metal. Windows/Linux selection prefers a discrete GPU and then Vulkan; software adapters are excluded. `--gpu-backend` and `--gpu-adapter` override selection, as do `PROTEINMOTION_GPU_BACKEND` and `PROTEINMOTION_GPU_ADAPTER`. The existing wgpu environment selectors remain supported.

Detection happens when a renderer and video writer are created, including direct calls to `scene.render()` from Python. No platform flags are needed for normal use. Exports with progress enabled print the detected platform and selected backend/encoder. The returned render report includes `platform`, `adapter` and `codec`; `progress=False` suppresses console output.

Vulkan passes the Windows rendering tests. DirectX 12 produced device-loss errors during transparency passes on this setup and is exposed for diagnosis, not recommended for production exports. Linux, other Windows GPUs, and interactive preview have not been validated in this Windows run. See [validation results](VALIDATION.md).

ProteinMotion exports images and videos as a standalone package. Use the exported files when composing a larger Manim scene or editing a video.

## Structure provenance

The package uses the MIT license. Manim animation timing and bundled Source Sans 3 fonts retain their upstream [licenses and attribution](https://github.com/pdpppd/proteinmotion/blob/main/THIRD_PARTY.md).

The example data include RCSB mmCIF files for [1UBQ](https://www.rcsb.org/structure/1UBQ) (ubiquitin), [1CLL](https://www.rcsb.org/structure/1CLL) and [1CFC](https://www.rcsb.org/structure/1CFC) (calmodulin), [1NCX](https://www.rcsb.org/structure/1NCX) (troponin C), [1AON](https://www.rcsb.org/structure/1AON) (GroEL/GroES), and [2K39](https://www.rcsb.org/structure/2K39) (ubiquitin NMR ensemble).

The structure morphs use human hemoglobin [2DN2](https://www.rcsb.org/structure/2DN2) (deoxy) and [2DN1](https://www.rcsb.org/structure/2DN1) (oxy, expanded to its biological assembly), and *E. coli* adenylate kinase [4AKE](https://www.rcsb.org/structure/4AKE) (open) and [1AKE](https://www.rcsb.org/structure/1AKE) (closed with Ap5A). The torsion examples use [1PGA](https://www.rcsb.org/structure/1PGA) (protein G B1 domain) for the β-hairpin angles. The confidence example uses the [AlphaFold DB model of calmodulin](https://alphafold.ebi.ac.uk/entry/P0DP23) (AF-P0DP23-F1, version 6, CC BY 4.0) and its PAE file, and the conservation example uses the [Pfam PF00240](https://www.ebi.ac.uk/interpro/entry/pfam/PF00240/) seed alignment.

The DNA and RNA examples use [1BNA](https://www.rcsb.org/structure/1BNA), a B-DNA dodecamer, and [1EHZ](https://www.rcsb.org/structure/1EHZ), yeast phenylalanine tRNA. These examples animate the camera and representation around deposited coordinates.

The original showcase uses a synthetic ubiquitin deformation. The NMR examples interpolate deposited models. The feature demo stores mapped 1CFC NMR coordinates in an XTC file to demonstrate the trajectory reader. The [feature demo guide](showcase.md) records the mapping and animation settings.

Implementation references: [Manim scenes](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html), [wgpu backends](https://wgpu-py.readthedocs.io/en/stable/backends.html), [Gemmi structure I/O](https://gemmi.readthedocs.io/en/stable/mol.html), [MDAnalysis trajectory readers](https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html), and [PyAV video generation](https://pyav.org/docs/stable/cookbook/numpy.html#generating-video).
