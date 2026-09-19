import platform
import sys
from fractions import Fraction

import av
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
    "codec,expected",
    [
        ("auto", "h264"),
        ("libx264", "h264"),
        ("h264_nvenc", "h264"),
        ("hevc_nvenc", "hevc"),
        ("av1_nvenc", "av1"),
        ("hevc_videotoolbox", "hevc"),
    ],
)
def test_hardware_video_export_and_frame_count(protein, tmp_path, codec, expected, encoder_checks, capsys):
    if codec not in ("auto", "libx264") and not encoder_checks.get(codec, {}).get("usable"):
        pytest.skip(f"{codec} unavailable: {encoder_checks.get(codec)}")

    s = ProteinScene(width=320, height=240, fps=10)
    s.add(protein)
    s.camera.frame(protein)
    s.play(Rotate(protein, 1), run_time=0.5)
    output = tmp_path / "tiny.mp4"
    # The ordinary API call must detect the encoder without CLI flags or doctor.
    result = s.render(output) if codec == "auto" else s.render(output, progress=False, codec=codec)
    assert result["frames"] == 5
    assert result["platform"] == platform.system()
    if codec != "auto":
        assert result["codec"] == codec
        assert capsys.readouterr().out == ""
    else:
        preferred = "h264_videotoolbox" if sys.platform == "darwin" else "h264_nvenc"
        if encoder_checks.get(preferred, {}).get("usable"):
            assert result["codec"] == preferred
        output_log = capsys.readouterr().out
        assert f"Platform: {platform.system()}" in output_log
        assert result["adapter"]["device"] in output_log
        assert result["adapter"]["backend_type"] in output_log
        assert result["codec"] in output_log
    with av.open(output) as movie:
        stream = movie.streams.video[0]
        assert (stream.width, stream.height) == (320, 240)
        assert stream.codec_context.codec.id == av.Codec(expected, "r").id
        assert stream.codec_context.colorspace == 1  # BT.709
        assert stream.codec_context.color_range == 1  # limited range
        frames = list(movie.decode(video=0))
        assert len(frames) == 5
        assert [f.pts * f.time_base for f in frames] == [Fraction(i, 10) for i in range(5)]
        assert (
            np.abs(
                frames[0].to_ndarray(format="rgb24").astype(int) - frames[-1].to_ndarray(format="rgb24")
            ).sum()
            > 1000
        )


@pytest.fixture(scope="module")
def encoder_checks():
    from proteinmotion.video import probe_encoders

    return probe_encoders()


@pytest.mark.gpu
@pytest.mark.parametrize("pixel_format", ["rgba", "nv12"])
def test_nvenc_padded_frames_and_fractional_timestamps(tmp_path, encoder_checks, pixel_format):
    from proteinmotion.video import VideoWriter

    if not encoder_checks.get("h264_nvenc", {}).get("usable"):
        pytest.skip("H.264 NVENC unavailable")
    # Even dimensions that are not aligned to encoder strides or GPU copy rows.
    width, height = 386, 258
    rate = Fraction(30000, 1001)
    pixels = np.full((height, width, 4), 255, np.uint8)
    if pixel_format == "nv12":
        pixels = np.full((height * 3 // 2, width), 128, np.uint8)
        pixels[:height] = 235
    path = tmp_path / "padded.mp4"
    with VideoWriter(path, width, height, rate, codec="h264_nvenc", pixel_format=pixel_format) as writer:
        assert writer.stream.pix_fmt == "nv12"
        for _ in range(4):
            writer.write(pixels)
    with av.open(path) as movie:
        frames = list(movie.decode(video=0))
    assert len(frames) == 4
    assert [f.pts * f.time_base for f in frames] == [i / rate for i in range(4)]
    for frame in frames:
        assert (frame.width, frame.height) == (width, height)
        assert frame.to_ndarray(format="rgb24").min() >= 250


def test_failed_video_export_preserves_existing_file(tmp_path):
    from proteinmotion.video import VideoWriter

    path = tmp_path / "existing.mp4"
    path.write_bytes(b"existing content")
    with pytest.raises(RuntimeError, match="deliberate"):
        with VideoWriter(path, 320, 240, 30, codec="libx264"):
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
