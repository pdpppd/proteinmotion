import json
import shutil
import subprocess

import numpy as np
import pytest

from proteinmotion import PlayTrajectory, ProteinScene, Rotate, Trajectory
from proteinmotion.renderer import Renderer


@pytest.fixture(scope="module")
def renderer():
    # Run these on an actual GPU. A missing GPU is a visible skip, never a fake fallback.
    import wgpu

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter available")
    with Renderer(384, 256) as r:
        yield r


@pytest.mark.gpu
def test_all_representations_distinct_finite_and_opaque(protein, renderer):
    from proteinmotion.camera import Camera

    camera = Camera().frame(protein, aspect=384 / 256)
    background = np.array([0.03, 0.05, 0.08])
    images = []
    for method in ("cartoon", "ribbon", "ball_and_stick"):
        getattr(protein, method)()
        pixels = renderer.render([protein], camera, background)
        assert pixels.shape == (256, 384, 4)
        assert (pixels[:, :, 3] == 255).all()
        foreground = np.max(np.abs(pixels[:, :, :3].astype(int) - pixels[0, 0, :3]), axis=2) > 15
        assert foreground.sum() > 1000
        images.append(pixels)
    assert np.abs(images[0].astype(int) - images[1]).sum() > 10000
    assert np.abs(images[1].astype(int) - images[2]).sum() > 10000


@pytest.mark.gpu
def test_gpu_seek_reproducibility_and_streaming(protein, renderer):
    s = ProteinScene(width=384, height=256)
    s.add(protein)
    s.camera.frame(protein)
    frames = np.array([protein.positions + [i * 0.5, np.sin(i), 0] for i in range(5)])
    s.play(PlayTrajectory(protein, Trajectory(frames)), Rotate(protein, 1), run_time=2)
    first = s.render_frame(0.6, renderer=renderer)
    s.render_frame(1.5, renderer=renderer)
    repeat = s.render_frame(0.6, renderer=renderer)
    np.testing.assert_array_equal(first, repeat)
    streamed = []
    for time in [0, 0.2, 0.4, 0.6]:
        result = renderer.enqueue(s.seek(time), s.camera, s.background)
        if result is not None:
            streamed.append(result)
    streamed.extend(renderer.drain())
    assert len(streamed) == 4
    np.testing.assert_array_equal(streamed[-1], first)
    assert len(renderer._staging) == 3


@pytest.mark.gpu
@pytest.mark.parametrize(
    "codec,expected", [("auto", "h264"), ("libx264", "h264"), ("hevc_videotoolbox", "hevc")]
)
def test_hardware_video_export_and_frame_count(protein, tmp_path, codec, expected):
    from proteinmotion.video import available_encoders

    if codec != "auto" and codec not in available_encoders():
        pytest.skip(f"{codec} not provided by this PyAV build")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe unavailable")
    s = ProteinScene(width=320, height=240, fps=10)
    s.add(protein)
    s.camera.frame(protein)
    s.play(Rotate(protein, 1), run_time=0.5)
    output = tmp_path / "tiny.mp4"
    result = s.render(output, progress=False, codec=codec)
    assert result["frames"] == 5
    info = json.loads(
        subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)])
    )
    stream = info["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_frames"])) == (320, 240, 5)
    assert stream["codec_name"] == expected
    assert stream["color_space"] == "bt709"
    assert stream["color_range"] == "tv"


def test_failed_video_export_preserves_existing_file(tmp_path):
    from proteinmotion.video import VideoWriter

    path = tmp_path / "existing.mp4"
    path.write_bytes(b"existing content")
    with pytest.raises(RuntimeError, match="deliberate"):
        with VideoWriter(path, 320, 240, 30):
            raise RuntimeError("deliberate failure")
    assert path.read_bytes() == b"existing content"
    assert not list(tmp_path.glob(".proteinmotion-*"))


@pytest.mark.gpu
def test_gpu_nv12_conversion_matches_bt709_and_handles_padded_width(protein):
    from proteinmotion.camera import Camera

    # Deliberately even but not divisible by four, to exercise packed-word row padding.
    width, height = 386, 258
    camera = Camera().frame(protein, aspect=width / height)
    background = np.array([0.03, 0.05, 0.08])
    with Renderer(width, height) as r:
        rgb = r.render([protein], camera, background)[:, :, :3].astype(float) / 255
    with Renderer(width, height, readback_format="nv12") as r:
        nv12 = r.render([protein], camera, background)
    assert nv12.shape == (height * 3 // 2, width)
    y = np.round(16 + 219 * np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    averaged = rgb.reshape(height // 2, 2, width // 2, 2, 3).mean(axis=(1, 3))
    u = np.round(128 + 224 * np.dot(averaged, [-0.114572, -0.385428, 0.5]))
    v = np.round(128 + 224 * np.dot(averaged, [0.5, -0.454153, -0.045847]))
    np.testing.assert_allclose(nv12[:height], y, atol=1)
    np.testing.assert_allclose(nv12[height:, 0::2], u, atol=1)
    np.testing.assert_allclose(nv12[height:, 1::2], v, atol=1)
