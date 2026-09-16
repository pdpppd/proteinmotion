"""Direct PyAV to VideoToolbox export, without a raw-video subprocess pipe."""

import re
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av


def available_encoders():
    result = []
    for name in ("h264_videotoolbox", "hevc_videotoolbox", "libx264", "libx265"):
        try:
            av.Codec(name, "w")
        except (ValueError, av.FFmpegError):
            continue
        result.append(name)
    return result


class VideoWriter:
    def __init__(self, output, width, height, fps, *, codec="auto", bitrate="20M", pixel_format="rgba"):
        if pixel_format not in ("rgba", "nv12"):
            raise ValueError("pixel_format must be rgba or nv12")
        if width % 2 or height % 2:
            raise ValueError("H.264/HEVC export requires even width and height")
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kKmM]?)", str(bitrate))
        if match is None:
            raise ValueError("bitrate must be bits/second or a value such as '20M'")
        bits = int(float(match[1]) * {"": 1, "k": 1000, "m": 1000000}[match[2].lower()])
        if bits <= 0:
            raise ValueError("bitrate must be positive")
        encoders = available_encoders()
        if codec == "auto":
            codec = (
                "h264_videotoolbox"
                if sys.platform == "darwin" and "h264_videotoolbox" in encoders
                else "libx264"
            )
        if codec not in encoders:
            raise RuntimeError(f"PyAV does not provide encoder {codec}; available: {encoders}")
        self.codec, self.width, self.height, self.pixel_format = codec, width, height, pixel_format
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.rate = Fraction(str(fps)).limit_denominator(100000)
        self.time_base = 1 / self.rate
        self.count = 0
        with tempfile.NamedTemporaryFile(
            prefix=".proteinmotion-",
            suffix=self.output.suffix or ".mp4",
            dir=self.output.parent,
            delete=False,
        ) as tmp:
            self.temp_path = Path(tmp.name)
        self.container = None
        try:
            self.container = av.open(str(self.temp_path), "w", options={"movflags": "+faststart"})
            self.stream = self.container.add_stream(codec, rate=self.rate)
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = "nv12" if codec.endswith("videotoolbox") else "yuv420p"
            self.stream.bit_rate = bits
            self.stream.codec_context.time_base = self.time_base
            if codec.endswith("videotoolbox"):
                self.stream.options = {"allow_sw": "0"}
            else:
                self.stream.options = {"crf": "18", "preset": "fast"}
            if codec == "hevc_videotoolbox":
                self.stream.codec_context.codec_tag = "hvc1"
            if pixel_format == "nv12":
                ctx = self.stream.codec_context
                ctx.colorspace = 1  # AVCOL_SPC_BT709
                ctx.color_primaries = 1
                ctx.color_trc = 1
                ctx.color_range = 1  # AVCOL_RANGE_MPEG / limited range
            self.stream.codec_context.open()
        except BaseException:
            if self.container is not None:
                self.container.close()
            self.temp_path.unlink(missing_ok=True)
            raise

    def write(self, pixels):
        shape = (
            (self.height, self.width, 4)
            if self.pixel_format == "rgba"
            else (self.height * 3 // 2, self.width)
        )
        if pixels.shape != shape or str(pixels.dtype) != "uint8":
            raise ValueError(f"Video frame must have shape {shape} and use uint8 {self.pixel_format}")
        frame = av.VideoFrame.from_ndarray(pixels, format=self.pixel_format)
        frame.pts, frame.time_base = self.count, self.time_base
        if self.pixel_format == "nv12":
            frame.colorspace = 1
            frame.color_range = 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            self.container.close()
            if kind is None:
                self.temp_path.replace(self.output)
        finally:
            self.container.close()
            self.temp_path.unlink(missing_ok=True)
