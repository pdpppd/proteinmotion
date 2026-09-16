# ProteinMotion

**Protein animation, written in Python.**

A small, Manim-inspired API for molecular films, with native Metal rendering on Apple silicon. Compose cartoons, ribbons, ball-and-stick models, contact-guided morphs, and real structural ensembles in a Python scene.

[![Checks](https://github.com/pdpppd/proteinmotion/actions/workflows/checks.yml/badge.svg)](https://github.com/pdpppd/proteinmotion/actions/workflows/checks.yml)
[![Documentation](https://github.com/pdpppd/proteinmotion/actions/workflows/pages.yml/badge.svg)](https://pdpppd.github.io/proteinmotion/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-376e59)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/license-MIT-376e59)](LICENSE)

[**Documentation**](https://pdpppd.github.io/proteinmotion/) · [**Video gallery**](https://pdpppd.github.io/proteinmotion/gallery/) · [**API reference**](https://pdpppd.github.io/proteinmotion/docs/api/) · [**Complete NMR script**](examples/nmr_regions.py)

[![Ubiquitin rendered as a cartoon, ribbon, and ball-and-stick model](docs/assets/representations.png)](https://pdpppd.github.io/proteinmotion/gallery/)

## What you can make

- **Molecular representations:** secondary-structure cartoons, ribbons, and element-colored ball-and-stick models, with depth cueing and smooth transparency.
- **Authored motion:** eased rotation, translation, camera movement, deformation, and morphing with deterministic forward/backward seeking.
- **Different-protein morphs:** contact-map matching with configurable error bounds, staggered N-to-C movement, and fades for unmatched residues.
- **Focused explanations:** live residue selections, camera tracking, and 3D sphere, box, or atom-halo highlights.
- **Ensemble films:** multi-model PDB/mmCIF, memory-mapped NumPy trajectories, and lazy MDAnalysis readers for XTC, DCD, TRR, and more.
- **Fast export on Apple silicon:** native wgpu → Metal, GPU coordinate interpolation, 4× MSAA, GPU NV12 conversion, and VideoToolbox encoding.

ProteinMotion is a standalone package with Manim-inspired syntax. Export its clips for use in Manim or another editor; it is not a Manim Mobject plug-in.

## Install

Python 3.11+ and a native GPU are required. Apple silicon/macOS is the verified platform.

```bash
git clone https://github.com/pdpppd/proteinmotion.git
cd proteinmotion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[preview]'
proteinmotion doctor
```

Install `'.[preview,md]'` to add MD trajectory readers. The core install is `pip install -e .`. PyAV wheels bundle FFmpeg libraries; rendering needs no FFmpeg executable, browser, Blender, or Xcode build. This version is installed from source; no PyPI release is assumed.

## Your first film

Save this as `film.py` in the repository root:

```python
from proteinmotion import (
    FadeIn, FadeOut, Protein, ProteinScene, Representation, Rotate,
)


class MyFilm(ProteinScene):
    def construct(self):
        protein = Protein.from_file("examples/data/1ubq.cif").cartoon().center()
        self.add(protein)
        self.camera.frame(protein, aspect=self.width / self.height)

        self.play(FadeIn(protein), run_time=0.8)
        self.play(Rotate(protein, 1.2), run_time=2)

        helix = protein.select(chain="A", residues=(23, 34))
        highlight = helix.highlight(style="box", color="#f2ba67")
        self.play(FadeIn(highlight), run_time=0.6)
        self.focus(helix, margin=1.7, run_time=1.5)
        self.play(Rotate(protein, 0.5), run_time=1.5)
        self.play(FadeOut(highlight), run_time=0.6)
        self.focus(protein, run_time=1.2)

        self.play(Representation(protein, "ball_and_stick"), run_time=1)
        self.wait(0.8)


if __name__ == "__main__":
    MyFilm(width=1920, height=1080, fps=60).render("film.mp4")
```

```bash
python film.py
# Or render the included equivalent:
proteinmotion render examples/quickstart.py Quickstart -o film.mp4 --fps 60
# Interactive preview (requires the preview extra):
proteinmotion preview examples/quickstart.py Quickstart
```

Residue ranges are inclusive PDB author numbers; angles are radians and coordinates are ångströms. Animations inside one `play()` run together. Successive calls run in sequence.

## Explore the examples

| Example | What it demonstrates | Source |
|---|---|---|
| Region tour | Focus, moving 3D highlights, NMR playback | [nmr_regions.py](examples/nmr_regions.py) |
| 116-model ubiquitin ensemble | Cartoon and ball-and-stick state interpolation | [nmr_regions.py](examples/nmr_regions.py) |
| Calmodulin → troponin C | Contact-guided backbone morphing and unmatched fades | [backbone_morph.py](examples/backbone_morph.py) |
| GroEL/GroES | A 21-chain assembly with 58,870 selected heavy atoms | [large_protein.py](examples/large_protein.py) |
| Representation showcase | Cartoon, ribbon, atoms, morphs, and synthetic motion | [showcase.py](examples/showcase.py) |

```bash
proteinmotion render examples/nmr_regions.py RegionTour -o regions.mp4 --fps 60
proteinmotion render examples/nmr_regions.py NMRAtoms -o ensemble.mp4 --fps 60
proteinmotion render examples/backbone_morph.py BallAndStickDemo -o morph.mp4 --fps 60
proteinmotion render examples/large_protein.py GroELComplex -o groel.mp4 --fps 60
```

Watch these in the [video gallery](https://pdpppd.github.io/proteinmotion/gallery/). The included NMR ensemble is [PDB 2K39](https://www.rcsb.org/structure/2K39); the example aligns the core Cα atoms before playback.

## Performance and verification

On the tested **Apple M3 Max**, the 24-second NMR cartoon exported at 1080p/60 fps in **3.69 seconds**. The annotated region tour exported in 4.52 seconds. These are single local runs including rendering/readback/encoding and excluding loading and scene construction, with potentially warm driver caches—not universal performance guarantees.

**56 local tests pass**, including native Metal rendering, transparency, reproducible seeking, trajectory I/O, matching, and hardware encoding. All 4,326 frames of the three NMR examples decoded successfully. [Methods, raw measurements, and limits](https://pdpppd.github.io/proteinmotion/docs/validation/).

## Scientific and implementation limits

Morphs and interpolated states are visual transitions, not energy-minimized pathways or MD simulations. NMR model order is not a physical time sequence. Contact matching is a bounded optimization; large-input results do not promise a global optimum. Ball-and-stick morphs translate whole matched residues and crossfade endpoint atom sets, without a side-chain atom correspondence.

Prepare whole, unwrapped MD coordinates before loading. Secondary-structure assignments remain fixed during playback. Transparency is approximate weighted blending; there are no shadows, SSAO, molecular solvent surfaces, or text labels. Other GPUs/platforms have not been verified. [Full rendering notes](https://pdpppd.github.io/proteinmotion/docs/rendering/).

## Development

```bash
python -m pip install -e '.[dev,md]'
ruff check src tests examples
pytest
python -m build
```

The documentation lives in `docs/`; its static website lives in `website/`. Pushes to `main` build and deploy GitHub Pages automatically. See [CONTRIBUTING.md](CONTRIBUTING.md) for local docs preview, CI scope, and contribution guidance.

## License and provenance

[MIT](LICENSE) for the package and site code. Dependencies retain their own licenses. Included PDB structures: [1UBQ](https://www.rcsb.org/structure/1UBQ), [1CLL](https://www.rcsb.org/structure/1CLL), [1NCX](https://www.rcsb.org/structure/1NCX), [1AON](https://www.rcsb.org/structure/1AON), and [2K39](https://www.rcsb.org/structure/2K39). See [provenance](https://pdpppd.github.io/proteinmotion/docs/rendering/#structure-provenance) and the [ensemble report](docs/nmr-ensemble-report.json).
