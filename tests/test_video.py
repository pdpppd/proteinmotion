from fractions import Fraction

import av
import numpy as np
import pytest

from proteinmotion import ProteinScene, video
from proteinmotion.renderer import Renderer


@pytest.mark.parametrize("pixel_format", ["rgba", "nv12"])
def test_auto_falls_back_when_hardware_is_unavailable(tmp_path, monkeypatch, pixel_format):
    monkeypatch.setattr(video.sys, "platform", "win32")
    monkeypatch.setattr(video, "available_encoders", lambda: ["h264_nvenc", "libx264"])
    configure = video._configure_encoder
    open_encoder = video.VideoWriter._open
    attempted = []

    def track_open(self, codec, bits):
        # PyAV may reject an unavailable codec before configuration, e.g. NVENC on macOS.
        attempted.append(codec)
        return open_encoder(self, codec, bits)

    def unavailable(ctx, codec, *args):
        if codec == "h264_nvenc":
            raise av.error.ExternalError(1, "NVENC driver unavailable")
        configure(ctx, codec, *args)

    monkeypatch.setattr(video, "_configure_encoder", unavailable)
    monkeypatch.setattr(video.VideoWriter, "_open", track_open)
    path = tmp_path / "movie with spaces.mp4"
    path.write_bytes(b"original")
    shape = (240, 320, 4) if pixel_format == "rgba" else (360, 320)
    with pytest.warns(RuntimeWarning, match="using libx264"):
        with video.VideoWriter(path, 320, 240, 29.97, pixel_format=pixel_format) as writer:
            assert path.read_bytes() == b"original"
            assert writer.codec == "libx264"
            for _ in range(3):
                writer.write(np.zeros(shape, np.uint8))
    assert attempted == ["h264_nvenc", "libx264"]
    assert not list(tmp_path.glob(".proteinmotion-*"))
    with av.open(path) as movie:
        frames = list(movie.decode(video=0))
        assert len(frames) == 3
        assert [f.pts * f.time_base for f in frames] == [i / Fraction("29.97") for i in range(3)]


def test_explicit_hardware_failure_is_not_silently_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "available_encoders", lambda: ["h264_nvenc", "libx264"])

    def unavailable(*args):
        raise av.error.ExternalError(1, "NVENC driver unavailable")

    monkeypatch.setattr(video, "_configure_encoder", unavailable)
    path = tmp_path / "existing.mp4"
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="Could not initialize encoder h264_nvenc"):
        video.VideoWriter(path, 320, 240, 30, codec="h264_nvenc")
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".proteinmotion-*"))


@pytest.mark.parametrize(
    "platform,expected", [("win32", "h264_nvenc"), ("linux", "h264_nvenc"), ("darwin", "h264_videotoolbox")]
)
def test_platform_auto_codec_priority(monkeypatch, platform, expected):
    monkeypatch.setattr(video.sys, "platform", platform)
    assert video._encoder_candidates("auto", ["libx264", "h264_nvenc", "h264_videotoolbox"]) == [
        expected,
        "libx264",
    ]


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_bad_fps_does_not_create_output(tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        video.VideoWriter(tmp_path / "out.mp4", 320, 240, fps)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("codec", ["libx265", "hevc_videotoolbox", "hevc_nvenc"])
def test_hevc_movies_use_the_hvc1_tag(tmp_path, codec):
    # QuickTime and Safari play HEVC from an MP4 only with the hvc1 sample entry. libx265
    # writes hev1 unless the tag is set, and its name does not start with "hevc_".
    if codec not in video.available_encoders():
        pytest.skip(f"PyAV does not provide {codec}")
    path = tmp_path / "hevc.mp4"
    try:
        writer = video.VideoWriter(path, 64, 48, 30, codec=codec)
    except RuntimeError as exc:
        pytest.skip(f"{codec} unavailable: {exc}")
    with writer:
        for _ in range(3):
            writer.write(np.zeros((48, 64, 4), np.uint8))
    with av.open(path) as movie:
        assert movie.streams.video[0].codec_context.codec_tag == "hvc1"
        assert len(list(movie.decode(video=0))) == 3


@pytest.mark.parametrize("msaa", [0, 2, 8])
def test_unsupported_msaa_is_rejected_before_the_gpu_opens(msaa):
    # WebGPU guarantees only 1 and 4 samples. Other counts used to fail inside wgpu's
    # pipeline creation; a scene changed after construction gets the same message.
    with pytest.raises(ValueError, match="msaa must be 1 or 4"):
        Renderer(64, 48, msaa=msaa)
    scene = ProteinScene(width=64, height=48)
    scene.msaa = msaa
    with pytest.raises(ValueError, match="msaa must be 1 or 4"):
        scene.render_frame()
    with pytest.raises(ValueError, match="msaa must be 1 or 4"):
        ProteinScene(msaa=msaa)
