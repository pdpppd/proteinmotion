# Rendering

Python defines the scene and animation timeline. The GPU renders the molecular geometry, text, and annotations. On macOS, wgpu-native compiles the shaders for Metal, and VideoToolbox encodes the video.

## GPU rendering

Coordinate interpolation, backbone curves, ribbons, spheres, lighting, and depth cueing run on the GPU. Rotation and styling reuse geometry buffers and update small parameter buffers.

Residue color and opacity apply across representations. A separate pass draws text and callout lines over the protein image. Text geometry is cached after layout. Callout lines update to follow the selected coordinates. See [text and labels](text.md).

The renderer uses 4× multisample antialiasing (MSAA) to smooth geometry edges. Sphere silhouettes use analytical intersections and receive partial antialiasing. Current rendering limits include shadows, screen-space ambient occlusion, ray tracing, and refractive materials.

## Transparency

Opaque objects draw first and record their depth. Transparent objects then accumulate color and transmittance in floating-point buffers. A final GPU pass combines these buffers with the opaque image.

This is weighted blended order-independent transparency, based on [McGuire and Bavoil](https://www.jcgt.org/published/0002/02/09/). It approximates the ordering of overlapping transparent objects. The renderer allocates the extra buffers at the first fade and reuses them afterward.

Ball-and-stick bonds end at their atom surfaces. This hides the part of each cylinder inside an atom during both opaque rendering and fades. The clipping updates with atom positions and sizes. Distance and interaction lines keep their specified endpoints.

## Surfaces and interactions

Surface meshes are built on the CPU from voxel fields using distance transforms and marching cubes. The default `update="rebuild"` creates a new mesh when coordinates change. Rotation, color, and opacity changes reuse the mesh. Rebuilding a moving surface can take most of the export time.

The optional `update="deform"` mode moves a reference mesh on the GPU. Use it for small displacements; large changes can fold or tear the mesh. See [surface settings](styling.md).

Hydrogen-bond and screened Coulomb calculations use CPU spatial queries. Results are cached until coordinates or analysis settings change. Their 3D lines are drawn with the molecular geometry; 2D lines and all distance text are drawn over the image. See [interaction methods](interactions.md).

## Coordinates and connectivity

GPU playback holds two coordinate states at a time. New states require a CPU update of backbone orientation guides. GPU memory use depends on atom count and image size, so trajectory length primarily affects loading and playback time.

Bond detection uses covalent radii and a spatial index when the structure loads. The current method estimates connectivity from geometry; bond order, aromaticity, and explicit topology bonds are outside its scope.

Ribbons split at chain changes, missing Cα atoms, and initial Cα gaps greater than 4.8 Å. Connectivity and secondary-structure assignments stay fixed during playback.

Morphs and state playback interpolate coordinates. They illustrate structural changes. Use simulation trajectories and an appropriate analysis method when interpreting molecular dynamics.

## Video export

A Metal compute pass converts each frame to limited-range BT.709 NV12. PyAV sends the result to VideoToolbox for encoding. NV12 transfers 1.5 bytes per pixel, compared with 4 for RGBA. The current export path maps and copies GPU staging buffers before encoding.

Large scenes can be limited by GPU buffer size, geometry and pixel processing, CPU frame decoding, or encoder speed. See [tests and benchmarks](VALIDATION.md) for measured results and their conditions.

## Platforms

Rendering and hardware export are tested on Apple silicon Macs. macOS uses Metal by default. The backend can select native wgpu adapters on other platforms, but those configurations remain untested.

ProteinMotion exports images and videos as a standalone package. Use the exported files when composing a larger Manim scene or editing a video.

## Structure provenance

The package uses the MIT license. Manim animation timing and bundled Source Sans 3 fonts retain their upstream [licenses and attribution](https://github.com/pdpppd/proteinmotion/blob/main/THIRD_PARTY.md).

The example data include RCSB mmCIF files for [1UBQ](https://www.rcsb.org/structure/1UBQ) (ubiquitin), [1CLL](https://www.rcsb.org/structure/1CLL) and [1CFC](https://www.rcsb.org/structure/1CFC) (calmodulin), [1NCX](https://www.rcsb.org/structure/1NCX) (troponin C), [1AON](https://www.rcsb.org/structure/1AON) (GroEL/GroES), and [2K39](https://www.rcsb.org/structure/2K39) (ubiquitin NMR ensemble).

The original showcase uses a synthetic ubiquitin deformation. The NMR examples interpolate deposited models. The feature demo stores mapped 1CFC NMR coordinates in an XTC file to demonstrate the trajectory reader. The [feature demo guide](showcase.md) records the mapping and animation settings.

Implementation references: [Manim scenes](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html), [wgpu backends](https://wgpu-py.readthedocs.io/en/stable/backends.html), [Gemmi structure I/O](https://gemmi.readthedocs.io/en/stable/mol.html), [MDAnalysis trajectory readers](https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html), and [PyAV video generation](https://pyav.org/docs/stable/cookbook/numpy.html#generating-video).
