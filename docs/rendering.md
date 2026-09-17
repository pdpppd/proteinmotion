# Rendering and limits

- Python controls the scene/timeline. wgpu-native compiles WGSL shaders to Metal.
- Interpolation, spline evaluation, ribbon sweeps, sphere intersections, lighting,
  and depth cueing run on the GPU. Rotation-only frames update small uniform buffers.
- Opaque frames retain one geometry pass. On the first fade, the renderer lazily
  allocates RGBA16F accumulation and R16F transmittance targets. Transparent fragments
  use a second geometry pass against opaque depth, then a per-sample GPU composite.
  The targets are reused. Swept surfaces accumulate their exterior face once to avoid
  counting both skins of a ribbon or bond. This follows the approach of
  [McGuire and Bavoil](https://www.jcgt.org/published/0002/02/09/) with bounded depth/opacity weights.
- Ball-and-stick cylinders stop at the surfaces of their endpoint atoms. GPU clipping
  removes their buried portions before both opaque and transparent shading, so residue
  fades do not reveal sticks inside the balls. The clipping follows animated positions
  and atom sizes; distance and interaction rulers retain their specified endpoints.
- Vector text and callout leaders render in a separate overlay pass after molecular transparency. HarfBuzz/FontTools shape and tessellate cached glyphs; GPU uniforms control contour drawing and fills. Completed text skips the contour pass. Leaders follow projected 3D centroids and are not depth-occluded. See [text and labels](text.md).
- Residue color/opacity tracks run in the molecular shader and update small clock uniforms. Atom styles carry across all representations. Geometry buffers remain cached during styling.
- Surface meshes use CPU voxel fields, distance transforms and marching cubes. The default `update="rebuild"` updates the mesh when coordinates change; static surfaces reuse cached GPU triangles. Optional GPU deformation is approximate and can fold under large displacements. See [surface quality and update modes](styling.md).
- Hydrogen-bond geometry and screened-Coulomb contacts use cached CPU spatial queries. Contact changes update a bounded pool of rulers; 3D rulers join the depth-tested molecular pass, while 2D rulers join the overlay. Distance text is an overlay in either mode. See [interaction methods and units](interactions.md).
- Video export runs a Metal compute pass for limited-range BT.709 NV12 conversion,
  then feeds PyAV/VideoToolbox directly. NV12 readback uses 1.5 bytes/pixel instead of
  RGBA's 4. No raw-video subprocess pipe or CPU RGB-to-YUV conversion is required.
- Keyframe uploads rebuild backbone orientation guides on the CPU once per new state.
  Bond discovery uses a spatial index at load time. Bonds are geometric covalent-radius
  estimates; bond orders, aromaticity, and explicit topology connectivity are not modeled.
- Ribbons split at chain changes, absent alpha carbons, and initial CA gaps >4.8 Å.
  Topology/chain connectivity stays fixed throughout playback.
- GPU memory is bounded by atom count, screen resolution, and two coordinate states,
  rather than trajectory length. Large atom counts remain limited by adapter binding
  limits and shader/fragment throughput. CPU frame decoding, guide construction,
  GPU readback, and encoder throughput can dominate export. This is not a zero-copy
  IOSurface/CVPixelBuffer pipeline: staging buffers are still mapped and copied.
- 4× MSAA smooths geometric edges; analytical sphere silhouettes do not get full
  per-sample antialiasing. No shadows, SSAO, ray tracing, refractive/transmissive materials,
  MathTex/LaTeX, sequence alignment, or direct Manim Mobject integration in v0.6.
- Modern native GPUs are required. macOS requires Metal by default; other platforms
  can use native wgpu adapters but have not been verified in this delivery.

See [validation and benchmarks](VALIDATION.md) for the actual checks and timings.

## Structure provenance

MIT license for this package. Manim animation timing and bundled Source Sans 3 fonts retain their upstream [licenses and attribution](https://github.com/pdpppd/proteinmotion/blob/main/THIRD_PARTY.md). Dependencies keep their respective licenses. The included
structures are PDB [1UBQ](https://www.rcsb.org/structure/1UBQ) (ubiquitin),
[1CLL](https://www.rcsb.org/structure/1CLL) (calmodulin),
[1NCX](https://www.rcsb.org/structure/1NCX) (troponin C),
[1AON](https://www.rcsb.org/structure/1AON) (GroEL/GroES), and
[2K39](https://www.rcsb.org/structure/2K39) (ubiquitin NMR ensemble), downloaded as
mmCIF files from RCSB. The original showcase trajectory is a synthetic deformation
of ubiquitin, explicitly not MD data. The NMR example uses deposited 2K39 models.

Design references: [Manim scenes](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html),
[wgpu native backends](https://wgpu-py.readthedocs.io/en/stable/backends.html),
[Gemmi structure I/O](https://gemmi.readthedocs.io/en/stable/mol.html),
[MDAnalysis trajectory readers](https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html),
and [PyAV video generation](https://pyav.org/docs/stable/cookbook/numpy.html#generating-video).
