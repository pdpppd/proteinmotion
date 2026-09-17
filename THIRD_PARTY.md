# Third-party notices

ProteinMotion's original code is MIT licensed. The following material retains its upstream notices.

## Manim animation timing

The lag calculation and two-stage outline/fill behavior of `Write` / `Unwrite` are adapted from Manim Community's `Write` and `DrawBorderThenFill`, with native GPU rendering written for ProteinMotion.

- Source: [creation.py at commit 485c226168e9c189512b22468de89b18dbc1780e](https://github.com/ManimCommunity/manim/blob/485c226168e9c189512b22468de89b18dbc1780e/manim/animation/creation.py)
- License: [MIT, copyright 2018 3Blue1Brown LLC](src/proteinmotion/licenses/Manim-LICENSE.txt)
- Adapted in `src/proteinmotion/annotations.py` and `src/proteinmotion/shaders/text.wgsl`.

This package does not bundle or import Manim/Pango. Text is shaped using HarfBuzz, flattened with FontTools, triangulated with mapbox-earcut and drawn through wgpu/Metal. It is not a full or pixel-identical implementation of Manim Text, MarkupText, or MathTex.

## Bundled Source Sans 3 fonts

The unmodified Regular and Semibold OpenType fonts are from [Adobe Source Sans, commit 87b37a2daaed80fcb8e8ccb0085c4d72ddade12e](https://github.com/adobe-fonts/source-sans/tree/87b37a2daaed80fcb8e8ccb0085c4d72ddade12e).

Copyright 2010–2024 Adobe, with Reserved Font Name “Source”. Distributed under the [SIL Open Font License 1.1](src/proteinmotion/fonts/OFL.txt), included beside the font binaries in source distributions and wheels. The font license does not apply to documents or videos created using the fonts.

Other dependencies retain their respective upstream licenses. PDB structure provenance is documented in [docs/rendering.md](docs/rendering.md).

## Starter structure

The installable starter movie and Codex skill include deposited ubiquitin coordinates, [PDB 1UBQ](https://www.rcsb.org/structure/1UBQ), copied unchanged from `examples/data/1ubq.cif`. The file retains its structural metadata and source citation. These scientific data are separate from the package's original MIT-licensed code.

## Blender (optional external application)

The EEVEE backend runs a separately installed [Blender](https://www.blender.org/) executable. Blender and EEVEE are not bundled with ProteinMotion. Blender is distributed under the [GNU GPL](https://www.blender.org/about/license/). ProteinMotion's exported meshes and frame requests cross a process boundary; the EEVEE integration code in this repository uses the repository's MIT license.
