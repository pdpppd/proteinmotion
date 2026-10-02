"""Supersampled native and Studio frames: defaults, sizes, and convergence of edges."""

import numpy as np
import pytest

from proteinmotion import ProteinScene
from proteinmotion.camera import Camera


def test_scene_supersampling_defaults_and_validation():
    assert ProteinScene(width=1920, height=1080)._supersampling() == 2
    assert ProteinScene(width=640, height=360)._supersampling() == 2
    assert ProteinScene(width=3840, height=2160)._supersampling() == 1
    assert ProteinScene(width=3840, height=2160, supersampling=2)._supersampling() == 2
    for bad in (0, 5, 1.5):
        with pytest.raises(ValueError, match="supersampling"):
            ProteinScene(supersampling=bad)


@pytest.mark.gpu
@pytest.mark.parametrize("studio", [False, True])
def test_supersampled_frames_keep_size_and_converge(protein, studio):
    from proteinmotion.renderer import Renderer
    from proteinmotion.studio import StudioRenderer

    cls = StudioRenderer if studio else Renderer
    protein.cartoon()
    camera = Camera().frame(protein, aspect=2)
    images = {}
    for factor in (1, 2, 4):
        with cls(256, 128, supersampling=factor) as renderer:
            images[factor] = renderer.render([protein], camera, (0.04, 0.07, 0.13)).astype(float)
            assert renderer.width == 256 * factor and renderer.output_width == 256
        assert images[factor].shape == (128, 256, 4)
    with cls(256, 128, supersampling=2, readback_format="nv12") as renderer:
        assert renderer.render([protein], camera, (0.04, 0.07, 0.13)).shape == (192, 256)
    # Supersampling changes only edges, and 2x lies closer to a 4x reference than 1x does.
    one, two = (np.abs(images[f][..., :3] - images[4][..., :3]).mean() for f in (1, 2))
    assert two < one < 2
    with pytest.raises(ValueError, match="supersampling"):
        cls(256, 128, supersampling=5)
