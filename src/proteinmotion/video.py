"""Direct PyAV export with NVENC/VideoToolbox and a software fallback.

``ProteinScene.render`` writes its frames through ``VideoWriter``: RGBA arrays from EEVEE, or
NV12 arrays that the native and Studio renderers convert on the GPU (shaders/nv12.wgsl), which
hardware encoders take without another conversion. ``proteinmotion doctor`` lists and tests
the encoders with ``available_encoders`` and ``probe_encoders``.
"""

import math
import re
import sys
import tempfile
import warnings
from fractions import Fraction
from pathlib import Path

import av


def available_encoders():
    """Encoders compiled into PyAV; hardware usability is checked when opened."""
    result = []
    for name in (
        "h264_nvenc",
        "hevc_nvenc",
        "av1_nvenc",
        "h264_videotoolbox",
        "hevc_videotoolbox",
        "libx264",
        "libx265",
    ):
        try:
            av.Codec(name, "w")
        except (ValueError, av.FFmpegError):
            continue
        result.append(name)
    return result


def _encoder_candidates(codec, encoders):
    """Encoders to try, in order.

    An explicit ``codec`` must be in ``encoders``. "auto" tries the platform's H.264 hardware
    encoder (VideoToolbox on macOS, NVENC elsewhere), then libx264.
    """
    if codec != "auto":
        if codec not in encoders:
            raise RuntimeError(f"PyAV does not provide encoder {codec}; available: {encoders}")
        return [codec]
    preferred = "h264_videotoolbox" if sys.platform == "darwin" else "h264_nvenc"
    candidates = [name for name in (preferred, "libx264") if name in encoders]
    if not candidates:
        raise RuntimeError(f"PyAV does not provide an H.264 export encoder; available: {encoders}")
    return candidates


def _configure_encoder(ctx, codec, width, height, rate, bits, pixel_format):
    """Set size, pixel format, frame rate (a Fraction), bit rate (bits/s), quality options and
    color tags on the PyAV codec context ``ctx``.

    ``pixel_format`` is the format of the frames the caller will send, "rgba" or "nv12".
    """
    ctx.width, ctx.height = width, height
    # Hardware encoders take NV12; the software encoders take planar YUV 4:2:0. PyAV converts
    # frames of another format when they are encoded.
    ctx.pix_fmt = "nv12" if codec.endswith(("videotoolbox", "nvenc")) else "yuv420p"
    ctx.bit_rate = bits
    ctx.time_base = 1 / rate
    ctx.framerate = rate
    if codec.endswith("videotoolbox"):
        # Refuse VideoToolbox's software path. Without a free hardware encoder, opening fails,
        # and with codec="auto" VideoWriter falls back to libx264 with a warning.
        ctx.options = {"allow_sw": "0"}
    elif codec.endswith("nvenc"):
        # Keep the renderer's NV12 frames in NV12 through upload to NVENC.
        # P4 balances throughput/quality without costly lookahead or multipass.
        ctx.options = {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": "18"}
    else:
        # Constant-quality mode; CRF 18 is near visually lossless for x264.
        ctx.options = {"crf": "18", "preset": "fast"}
    if codec.startswith("hevc_"):
        # Apple players need the hvc1 sample entry to play HEVC from an MP4.
        ctx.codec_tag = "hvc1"
    if pixel_format == "nv12":
        # Tag the stream with the conversion that shaders/nv12.wgsl applied.
        ctx.colorspace = 1  # AVCOL_SPC_BT709
        ctx.color_primaries = 1
        ctx.color_trc = 1
        ctx.color_range = 1  # AVCOL_RANGE_MPEG / limited range


def probe_encoders():
    """Try encoding a small frame; listing a codec alone does not verify a GPU."""
    result = {}
    for codec in available_encoders():
        ctx = None
        try:
            ctx = av.CodecContext.create(codec, "w")
            _configure_encoder(ctx, codec, 320, 240, Fraction(30), 1_000_000, "nv12")
            ctx.open()
            frame = av.VideoFrame(320, 240, ctx.pix_fmt)
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.pts, frame.time_base = 0, Fraction(1, 30)
            ctx.encode(frame)
            ctx.encode(None)  # Flush, so the probe runs a complete encode.
        except (av.FFmpegError, ValueError) as exc:
            result[codec] = {"usable": False, "error": str(exc)}
        else:
            result[codec] = {"usable": True}
        finally:
            # Releasing the context also releases its hardware encoder session.
            del ctx
    return result


def _parse_bitrate(bitrate):
    """Bits per second from an int, a float, or a string such as "20M" or "800k"."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kKmM]?)", str(bitrate))
    if match is None:
        raise ValueError("bitrate must be bits/second or a value such as '20M'")
    bits = int(float(match[1]) * {"": 1, "k": 1000, "m": 1000000}[match[2].lower()])
    if bits <= 0:
        raise ValueError("bitrate must be positive")
    return bits


class VideoWriter:
    """Encode uint8 frames into a movie file, on a hardware encoder when one works.

    Frames go to a hidden temporary file next to ``output``, which replaces ``output`` only
    when the ``with`` block finishes without an error. A failed export therefore leaves any
    existing file untouched.
    """

    def __init__(self, output, width, height, fps, *, codec="auto", bitrate="20M", pixel_format="rgba"):
        if pixel_format not in ("rgba", "nv12"):
            raise ValueError("pixel_format must be rgba or nv12")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
            raise ValueError("Video dimensions must be positive integers")
        # 4:2:0 chroma has one sample per 2 × 2 pixel block.
        if width % 2 or height % 2:
            raise ValueError("H.264/HEVC export requires even width and height")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        bits = _parse_bitrate(bitrate)
        candidates = _encoder_candidates(codec, available_encoders())
        self.width, self.height, self.pixel_format = width, height, pixel_format
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Exact rational frame rate: str() keeps 29.97 as 2997/100, not its binary float value.
        self.rate = Fraction(str(fps)).limit_denominator(100000)
        self.time_base = 1 / self.rate
        self.count = 0
        self.fallback_reason = None
        self.container = None
        # Try each candidate encoder. With codec="auto", a hardware encoder that fails to open
        # falls back to the next one with a warning; an explicitly chosen encoder must work.
        for index, candidate in enumerate(candidates):
            try:
                self._open(candidate, bits)
            except (av.FFmpegError, ValueError) as exc:
                if codec != "auto" or index == len(candidates) - 1:
                    raise RuntimeError(
                        f"Could not initialize encoder {candidate}: {exc}. "
                        "Run proteinmotion doctor --check-encoders; use --codec libx264 for software export."
                    ) from exc
                self.fallback_reason = f"{candidate}: {exc}"
            else:
                self.codec = candidate
                if self.fallback_reason:
                    warnings.warn(
                        f"Hardware encoding unavailable ({self.fallback_reason}); using {candidate}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                break

    def _open(self, codec, bits):
        """Open a temporary output file and an encoder stream on it; clean up if that fails."""
        with tempfile.NamedTemporaryFile(
            prefix=".proteinmotion-",
            suffix=self.output.suffix or ".mp4",
            dir=self.output.parent,
            delete=False,
        ) as tmp:
            self.temp_path = Path(tmp.name)
        self.container = None
        try:
            # faststart moves the MP4 index to the front, so players can start before the
            # whole file has loaded.
            self.container = av.open(str(self.temp_path), "w", options={"movflags": "+faststart"})
            self.stream = self.container.add_stream(codec, rate=self.rate)
            _configure_encoder(
                self.stream.codec_context,
                codec,
                self.width,
                self.height,
                self.rate,
                bits,
                self.pixel_format,
            )
            self.stream.codec_context.open()
        except BaseException:
            if self.container is not None:
                self.container.close()
            self.temp_path.unlink(missing_ok=True)
            raise

    def write(self, pixels):
        """Encode one frame: (height, width, 4) RGBA, or (height × 3 / 2, width) NV12, uint8."""
        # NV12 is a full-height luma plane followed by a half-height plane of interleaved U, V.
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
            frame.colorspace = 1  # BT.709, limited range, as converted on the GPU.
            frame.color_range = 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            # On success, flush the encoder's buffered frames, finish the file and move it into
            # place. On an error, keep the existing output.
            if kind is None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            self.container.close()
            if kind is None:
                self.temp_path.replace(self.output)
        finally:
            # Closing again is harmless; the temporary file is gone after a successful replace.
            self.container.close()
            self.temp_path.unlink(missing_ok=True)
