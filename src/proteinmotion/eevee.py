"""Optional Blender EEVEE backend, using one persistent Blender process per export.

``EEVEE`` has the same ``render(objects, camera, background)`` interface as the native
``Renderer``; ``ProteinScene.render`` and ``render_frame`` create one for
``renderer="eevee"``. Blender runs as a separate process with its own Python, which runs
``_blender_worker.py``. The two sides exchange files in a temporary folder:

1. On start, Blender runs the worker with the folder path after ``--``. The worker builds its
   scene and writes ``ready.json`` (Blender version and GPU backend, or an error).
2. For each frame, ``_eevee_geometry`` turns every 3D object into world-space triangle
   meshes in Å. They are saved to ``frame.npz``, and one JSON line on the worker's stdin
   carries the image size, camera, background, fog and ``EEVEEOptions``.
3. The worker renders ``frame.png`` and writes ``done.json`` (``{"ok": true}`` or an error).
   This module reads the PNG and draws 2D annotations over it with the native renderer.

Blender's console output goes to ``blender.log`` in the folder and is quoted in errors.
Closing the worker's stdin ends its loop and Blender exits.
"""

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

from ._eevee_geometry import MeshExporter, tunnel_meshes
from .annotations import Annotation


@dataclass(frozen=True)
class EEVEEOptions:
    """EEVEE quality and executable settings. Blender 4.5+ is installed separately.

    More distinct opacity values require more render layers. The layer limit raises
    an error rather than silently changing a scene's transparency.
    """

    blender: str | None = None  # Executable path; None lets find_blender() search.
    samples: int = 64  # EEVEE render samples per pixel.
    supersampling: float = 1.5  # Render at this multiple of the output size, then downsample.
    max_blur: float = 24.0  # Largest depth-of-field blur (bokeh size), in output pixels.
    max_opacity_layers: int = 64  # Each distinct face opacity in a frame needs one view layer.
    timeout: float = 180.0  # Seconds to wait for Blender to start or to finish one frame.

    def __post_init__(self):
        # A frozen dataclass cannot assign normally; store a path-like blender as a string.
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
    # A given path (argument or PROTEINMOTION_BLENDER) is the only candidate: a wrong path
    # raises rather than falling back to another Blender.
    explicit = executable or os.environ.get("PROTEINMOTION_BLENDER")
    candidates = [explicit] if explicit else [shutil.which("blender")]
    if not explicit and sys.platform == "darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    if not explicit and sys.platform == "win32":
        # Windows installs to "Blender Foundation/Blender 4.5/blender.exe" and so on.
        # Try the newest version first, comparing the numbers in the folder name.
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
            # A bare command name resolves through PATH; anything else is taken as a path.
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
        # The 2D overlay renderer is created on the first frame that has annotations.
        self.exporter, self.overlay = MeshExporter(), None
        self.process = self.log = self.temporary = None
        self.closed = False
        try:
            self.temporary = tempfile.TemporaryDirectory(prefix="proteinmotion-eevee-")
            self.folder = Path(self.temporary.name)
            self.log = (self.folder / "blender.log").open("w")
            # Run without a window, from Blender's factory settings rather than the user's
            # startup file. --python-exit-code makes an uncaught worker exception exit with 1.
            command = [executable, "--background", "--factory-startup"]
            if sys.platform == "darwin":
                command += ["--gpu-backend", "metal"]
            # Blender passes the arguments after "--" to the worker: the exchange folder.
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
            # Also on KeyboardInterrupt: stop Blender and remove the folder before re-raising.
            self.close()
            raise

    def _wait(self, status):
        """Poll for a status file from the worker; return its JSON and delete the file.

        Raises RuntimeError with the end of Blender's log if Blender exits or the timeout
        passes first, or with the worker's traceback if the file reports an error.
        """
        start = time.monotonic()
        while not status.exists():
            if self.process.poll() is not None or time.monotonic() - start > self.options.timeout:
                detail = (self.folder / "blender.log").read_text(errors="replace")[-6000:]
                raise RuntimeError(f"Blender EEVEE failed or timed out.\n{detail}")
            time.sleep(0.01)
        result = json.loads(status.read_text())
        # Delete it so the next wait cannot see this frame's status.
        status.unlink()
        if "error" in result:
            raise RuntimeError(f"Blender EEVEE: {result['error']}")
        return result

    def render(self, objects, camera, background):
        """Render current objects to an RGBA uint8 image; annotations stay sharp."""
        if self.closed:
            raise RuntimeError("EEVEE renderer is closed")
        camera.update_tracking()
        # Molecules and other 3D objects go to Blender; annotations are drawn afterward in 2D.
        annotations, geometry = [], []
        for obj in objects:
            if isinstance(obj, Annotation):
                annotations.append(obj)
                # 3D parts of annotations (distance rulers, interaction lines) render in the
                # scene, where molecules can hide them; the overlay keeps only the 2D parts.
                if hasattr(obj, "_geometry_objects"):
                    geometry.extend(
                        g
                        for g in obj._geometry_objects(camera, self.width, self.height)
                        if not isinstance(g, Annotation)
                    )
                continue
            geometry.append(obj)
        count = self._export_meshes(geometry, camera)
        # Positions are world Å and fov is the vertical field of view in radians. The
        # background is display RGB in [0, 1].
        request = dict(
            width=self.width,
            height=self.height,
            meshes=count,
            options=asdict(self.options),
            eye=camera.eye.tolist(),
            target=camera.target.tolist(),
            fov=float(camera.fov),
            dof=bool(camera.dof),
            focus=camera.focus_point.tolist(),
            fstop=camera.fstop,
            background=np.asarray(background).tolist(),
            # Native distance fog: smoothstep over this range of camera distance, times depth_cue.
            fog=[
                float(camera.distance - camera.radius * 1.05),
                float(camera.distance + camera.radius * 1.5),
                float(camera.depth_cue),
            ],
        )
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        self._wait(self.folder / "done.json")
        # Blender rendered at the supersampled size; downsample to the output size.
        with Image.open(self.folder / "frame.png") as im:
            if im.size != (self.width, self.height):
                im = im.resize((self.width, self.height), Image.Resampling.LANCZOS)
            pixels = np.array(im.convert("RGBA"))
        # Composite any transparent pixels over the exact background color, as native does.
        # The worker's compositor starts from an opaque background color, so in practice
        # alpha is already 255 everywhere and this changes nothing.
        alpha = pixels[:, :, 3:4].astype(np.float32) / 255
        backdrop = np.round(np.clip(np.asarray(background, np.float32), 0, 1) * 255)
        pixels[:, :, :3] = np.round(pixels[:, :, :3] * alpha + backdrop * (1 - alpha)).astype(np.uint8)
        pixels[:, :, 3] = 255
        if annotations:
            self._draw_annotations(pixels, annotations, camera)
        return pixels

    def _export_meshes(self, geometry, camera):
        """Save the meshes of ``geometry`` and the cutaway tunnel to frame.npz; return their count."""
        meshes = [mesh for obj in geometry for mesh in self.exporter.meshes(obj, camera)]
        meshes += tunnel_meshes(camera, self.height)
        # Flat npz keys "<mesh index>_<array name>", such as "0_vertices", as the worker reads them.
        arrays, index = {}, 0
        for mesh in meshes:
            for key, data in mesh.items():
                arrays[f"{index}_{key}"] = data
            index += 1
        np.savez(self.folder / "frame.npz", **arrays)
        return index

    def _draw_annotations(self, pixels, annotations, camera):
        """Draw the 2D parts of annotations over ``pixels`` (RGBA uint8, changed in place)."""
        # Reuse the existing vector text/Write/leader renderer. Black/white
        # passes recover exact per-pixel transmittance, including MSAA edges.
        if self.overlay is None:
            from .renderer import Renderer

            self.overlay = Renderer(self.width, self.height, msaa=4)
            self.overlay.draw_geometry_objects = False
        black = self.overlay.render(annotations, camera, np.zeros(3)).astype(np.float32)
        white = self.overlay.render(annotations, camera, np.ones(3)).astype(np.float32)
        # Over black the overlay gives its premultiplied color A; over white it gives
        # A + 255 T, where T is the transmittance. So the result over the film F is
        # A + F T = black + F (white - black) / 255.
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

    def close(self):
        """Release Blender, overlay GPU buffers and temporary frame files."""
        if self.closed:
            return
        self.closed = True
        if self.overlay is not None:
            self.overlay.close()
        if self.process is not None:
            # End of stdin ends the worker loop. Give Blender 5 s to exit, then terminate,
            # then kill.
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
