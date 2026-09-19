"""Optional Blender EEVEE backend, using one persistent Blender process per export."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ._eevee_geometry import MeshExporter
from .annotations import Annotation


@dataclass(frozen=True)
class EEVEEOptions:
    """EEVEE quality and executable settings. Blender 4.5+ is installed separately.

    More distinct opacity values require more render layers. The layer limit raises
    an error rather than silently changing a scene's transparency.
    """

    blender: str | None = None
    samples: int = 64
    supersampling: float = 1.5
    max_blur: float = 24.0
    max_opacity_layers: int = 64
    timeout: float = 180.0

    def __post_init__(self):
        if self.blender is not None:
            object.__setattr__(self, "blender", str(self.blender))
        for name in ("samples", "max_opacity_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("supersampling", "max_blur", "timeout"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.supersampling < 1:
            raise ValueError("supersampling must be at least 1")


def find_blender(executable=None):
    """Find Blender via an explicit path, environment, PATH or standard installation."""
    explicit = executable or os.environ.get("PROTEINMOTION_BLENDER")
    candidates = (
        [explicit]
        if explicit
        else [shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"]
    )
    if not explicit and sys.platform == "win32":
        installed = []
        for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            if root := os.environ.get(variable):
                installed.extend((Path(root) / "Blender Foundation").glob("Blender */blender.exe"))
        candidates.extend(
            sorted(
                set(installed),
                key=lambda p: tuple(map(int, re.findall(r"\d+", p.parent.name))),
                reverse=True,
            )
        )
    for candidate in candidates:
        if candidate:
            path = Path(shutil.which(str(candidate)) or str(candidate)).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    raise RuntimeError(
        "EEVEE requires Blender 4.5 or later. Install Blender from https://www.blender.org/download/ "
        "and set PROTEINMOTION_BLENDER or pass --blender /path/to/blender. "
        "The native renderer works without Blender."
    )


class EEVEE:
    """Persistent frame renderer implementing ProteinScene's renderer interface."""

    def __init__(self, width=1920, height=1080, *, options=None):
        self.options = options or EEVEEOptions()
        if not isinstance(self.options, EEVEEOptions):
            raise TypeError("eevee must be an EEVEEOptions instance")
        if any(not isinstance(v, int) or v < 1 for v in (width, height)):
            raise ValueError("Image dimensions must be positive integers")
        executable = find_blender(self.options.blender)
        self.width, self.height = width, height
        self.exporter, self.overlay = MeshExporter(), None
        self.process = self.log = self.temporary = None
        self.closed = False
        try:
            self.temporary = tempfile.TemporaryDirectory(prefix="proteinmotion-eevee-")
            self.folder = Path(self.temporary.name)
            self.log = (self.folder / "blender.log").open("w")
            command = [executable, "--background", "--factory-startup"]
            if sys.platform == "darwin":
                command += ["--gpu-backend", "metal"]
            command += [
                "--python-exit-code",
                "1",
                "--python",
                str(Path(__file__).with_name("_blender_worker.py")),
                "--",
                str(self.folder),
            ]
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self.adapter_info = self._wait(self.folder / "ready.json")
        except BaseException:
            self.close()
            raise

    def _wait(self, status):
        start = time.monotonic()
        while not status.exists():
            if self.process.poll() is not None or time.monotonic() - start > self.options.timeout:
                detail = (self.folder / "blender.log").read_text(errors="replace")[-6000:]
                raise RuntimeError(f"Blender EEVEE failed or timed out.\n{detail}")
            time.sleep(0.01)
        result = json.loads(status.read_text())
        status.unlink()
        if "error" in result:
            raise RuntimeError(f"Blender EEVEE: {result['error']}")
        return result

    def render(self, objects, camera, background):
        """Render current objects to an RGBA uint8 image; annotations stay sharp."""
        if self.closed:
            raise RuntimeError("EEVEE renderer is closed")
        arrays, index = {}, 0
        annotations = []
        for obj in objects:
            if isinstance(obj, Annotation):
                annotations.append(obj)
                continue
            for mesh in self.exporter.meshes(obj):
                for key, data in mesh.items():
                    arrays[f"{index}_{key}"] = data
                index += 1
        np.savez(self.folder / "frame.npz", **arrays)
        request = dict(
            width=self.width,
            height=self.height,
            meshes=index,
            options=asdict(self.options),
            eye=camera.eye.tolist(),
            target=camera.target.tolist(),
            fov=float(camera.fov),
            dof=bool(camera.dof),
            focus=camera.focus_point.tolist(),
            fstop=camera.fstop,
            background=np.asarray(background).tolist(),
        )
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        self._wait(self.folder / "done.json")
        with Image.open(self.folder / "frame.png") as im:
            if im.size != (self.width, self.height):
                im = im.resize((self.width, self.height), Image.Resampling.LANCZOS)
            pixels = np.array(im.convert("RGBA"))
        if annotations:
            # Reuse the existing vector text/Write/leader renderer. Black/white
            # passes recover exact per-pixel transmittance, including MSAA edges.
            if self.overlay is None:
                from .renderer import Renderer

                self.overlay = Renderer(self.width, self.height, msaa=4)
            black = self.overlay.render(annotations, camera, np.zeros(3)).astype(np.float32)
            white = self.overlay.render(annotations, camera, np.ones(3)).astype(np.float32)
            pixels[:, :, :3] = (
                np.clip(
                    black[:, :, :3]
                    + pixels[:, :, :3].astype(np.float32) * (white[:, :, :3] - black[:, :, :3]) / 255,
                    0,
                    255,
                )
                .round()
                .astype(np.uint8)
            )
        return pixels

    def close(self):
        """Release Blender, overlay GPU buffers and temporary frame files."""
        if self.closed:
            return
        self.closed = True
        if self.overlay is not None:
            self.overlay.close()
        if self.process is not None:
            if self.process.stdin:
                with suppress(BrokenPipeError, OSError):
                    self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.log is not None:
            self.log.close()
        if self.temporary is not None:
            self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
