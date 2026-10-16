"""The same scene at the same time renders the same pixels, in any renderer and any process.

The website's gallery and docs previews are compared frame by frame, so a render must not
depend on what the renderer drew before or on the process it runs in.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from proteinmotion import Colorize, Protein, ProteinScene, StudioLook
from proteinmotion.studio import StudioRenderer

STRUCTURE = Path(__file__).resolve().parents[1] / "examples/data/1ubq.cif"


def red_group_scene(msaa=4):
    """A red-tinted cartoon at the end of a timed group, with bloom, halation and grain.

    A review once reported this kind of frame flipping between two images across runs.
    """
    protein = Protein.from_file(STRUCTURE).center()
    look = StudioLook(effects="film", grain=0.05)
    scene = ProteinScene(width=640, height=360, msaa=msaa, renderer="studio", look=look).add(protein)
    scene.camera.frame(protein)
    scene.play(protein.animate.shift((2, 0, 0)).during(2), Colorize(protein, "#ff0000").during(0.5, delay=1))
    return scene.build()


def digest(pixels):
    return hashlib.sha1(pixels.tobytes()).hexdigest()


@pytest.mark.gpu
@pytest.mark.parametrize("msaa", [1, 4])
def test_studio_frame_repeats_with_fresh_and_reused_renderers(msaa):
    scene = red_group_scene(msaa)
    end = scene.duration
    first = scene.render_frame(end)
    red, green, blue = np.moveaxis(first[:, :, :3].astype(int), -1, 0)
    assert np.mean((red > green + 40) & (red > blue + 40)) > 0.01  # the cartoon is red
    np.testing.assert_array_equal(scene.render_frame(end), first)
    size = scene.width, scene.height
    with StudioRenderer(*size, msaa=msaa, supersampling=scene._supersampling(), look=scene.look) as renderer:
        # Earlier frames leave other contents in the renderer's targets and caches.
        for time in (0.0, 1.2, end, 0.6, end):
            image = scene.render_frame(time, renderer=renderer)
        np.testing.assert_array_equal(image, first)
    native = [scene.render_frame(end, renderer="native") for _ in range(2)]
    np.testing.assert_array_equal(native[0], native[1])


@pytest.mark.gpu
def test_studio_frame_repeats_across_processes():
    # String hashes and memory addresses change from process to process.
    child = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_determinism import digest, red_group_scene\n"
        "print(digest(red_group_scene().render_frame(2.0)))\n"
    )
    digests = [
        subprocess.run(
            [sys.executable, "-c", child],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[-1]
        for seed in ("0", "1")
    ]
    assert digests == [digest(red_group_scene().render_frame(2.0))] * 2
