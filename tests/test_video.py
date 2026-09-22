from fractions import Fraction

import av
import numpy as np
import pytest

from proteinmotion import video


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
