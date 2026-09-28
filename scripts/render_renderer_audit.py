"""Render the native/studio visual audit without modifying either renderer.

    .venv/bin/python scripts/render_renderer_audit.py
    .venv/bin/python scripts/render_renderer_audit.py --only R088,R089
    .venv/bin/python scripts/render_renderer_audit.py --gallery-only

Stills: 1920×1080 per renderer, lossless PNG, plus labeled comparison JPEGs.
Clips: 1280×720 per renderer, side-by-side 2560×784, 60 fps at original timeline speed.
Each case runs in a fresh subprocess to release GPU resources between cases.
"""

import argparse
import ast
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from renderer_audit import cases  # noqa: E402


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def header(width, height, case, when):
    im = Image.new("RGBA", (width, height), "#111a26")
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(
        str(ROOT / "src/proteinmotion/fonts/SourceSans3-Regular.otf"), int(height * 0.31)
    )
    d.text((18, 3), f"{case.id}  /  {case.title}  /  t = {when:.3f} s", fill="#eaf0f8", font=font)
    d.text((18, height * 0.50), "NATIVE  ·  current renderer", fill="#8cdad0", font=font)
    d.text((width // 2 + 18, height * 0.50), "STUDIO  ·  new renderer", fill="#f4c58d", font=font)
    return np.array(im)


def pair_image(left, right, case, when, header_height):
    w, h = left.shape[1], left.shape[0]
    pair = np.empty((h + header_height, w * 2, 4), np.uint8)
    pair[:header_height] = header(w * 2, header_height, case, when)
    pair[header_height:, :w] = left
    pair[header_height:, w:] = right
    return pair


def scene_digest(scene, when):
    """Check that seek/render does not give the second backend changed input geometry."""
    from proteinmotion import Protein

    h = hashlib.sha256()
    objs = scene.seek(when)
    h.update(np.asarray(scene.camera.matrix(scene.width / scene.height), np.float64).tobytes())
    for obj in objs:
        if isinstance(obj, Protein):
            for x in (obj.positions, obj.model_matrix, obj.atom_opacities, obj.representation):
                h.update(np.asarray(x).tobytes())
            h.update(str((obj.opacity, obj.surface_opacity, obj.color_scheme)).encode())
    return h.hexdigest()


def worker(args):
    from proteinmotion import StudioLook
    from proteinmotion.renderer import Renderer
    from proteinmotion.studio import StudioRenderer
    from proteinmotion.video import VideoWriter

    case = next(c for c in cases() if c.id == args.worker)
    folder = args.out / case.id
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(
        id=case.id,
        group=case.group,
        title=case.title,
        api=case.api,
        notes=case.notes,
        look=asdict(StudioLook(**case.look)),
        explicit_look=case.look,
        scene_options=case.scene_options,
        samples=[],
        errors=[],
        status="rendering",
        clip=None,
    )
    started = time.perf_counter()
    observed = set()
    source_prefix = str(ROOT / "src/proteinmotion") + "/"

    def observe(frame, event, arg):
        if event == "call" and frame.f_code.co_filename.startswith(source_prefix):
            observed.add(frame.f_code.co_qualname)

    sys.setprofile(observe)
    try:
        scene = case.factory(width=args.width, height=args.height, fps=args.fps, **case.scene_options).build()
        result["duration"] = scene.duration
        result["scene_class"] = type(scene).__name__
        target = case.factory.func if hasattr(case.factory, "func") else case.factory
        if hasattr(case.factory, "args") and case.factory.args:
            target = case.factory.args[0]
            target = target.func if hasattr(target, "func") else target
        result["source"] = str(Path(inspect.getsourcefile(target)).relative_to(ROOT))
        result["source_line"] = inspect.getsourcelines(target)[1]
        times = list(case.times)
        if not times:
            times = [scene.duration * x for x in (0.2, 0.5, 0.8)] if case.video else [scene.duration / 2]
        elif case.video and len(times) == 1:
            times = [times[0], scene.duration * 0.2, scene.duration * 0.8]
        times = list(dict.fromkeys(round(float(np.clip(t, 0, scene.duration)), 6) for t in times))
        native = Renderer(args.width, args.height, msaa=scene.msaa)
        studio = StudioRenderer(args.width, args.height, msaa=scene.msaa, look=StudioLook(**case.look))
        try:
            result["adapter"] = dict(native.adapter.info)
            for i, when in enumerate(times):
                before = scene_digest(scene, when)
                left = scene.render_frame(when, renderer=native)
                after_native = scene_digest(scene, when)
                right = scene.render_frame(when, renderer=studio)
                after_studio = scene_digest(scene, when)
                if len({before, after_native, after_studio}) != 1:
                    raise AssertionError(f"Input geometry/camera changed between backends at {when}")
                sample = dict(time=when, state_hash=before, width=args.width, height=args.height)
                for name, pixels in (("native", left), ("studio", right)):
                    file = f"{case.id}/{i:02d}-{name}.png"
                    Image.fromarray(pixels).save(args.out / file)
                    sample[name] = file
                pair = Image.fromarray(pair_image(left, right, case, when, 96)).convert("RGB")
                sample["pair"] = f"{case.id}/{i:02d}-pair.jpg"
                pair.save(args.out / sample["pair"], quality=94, subsampling=0)
                thumb = pair.copy()
                thumb.thumbnail((1280, 400))
                sample["thumb"] = f"{case.id}/{i:02d}-thumb.webp"
                thumb.save(args.out / sample["thumb"], quality=87)
                result["samples"].append(sample)
                dump(folder / "result.json", result)
        finally:
            native.close()
            studio.close()
        sys.setprofile(None)
        if case.video and not args.stills_only:
            scene.width, scene.height = args.video_width, args.video_width * 9 // 16
            w, h = scene.width, scene.height
            use_exports = case.id == "R153"
            exports = {}
            if use_exports:
                import av

                sys.setprofile(observe)
                for name in ("native", "studio"):
                    path = folder / f"{name}-export.mp4"
                    options = {"look": StudioLook(**case.look)} if name == "studio" else {}
                    exports[name] = scene.render(path, renderer=name, progress=False, **options)
                sys.setprofile(None)
                native = av.open(str(folder / "native-export.mp4"))
                studio = av.open(str(folder / "studio-export.mp4"))
                frames_native, frames_studio = native.decode(video=0), studio.decode(video=0)
            else:
                native = Renderer(w, h, msaa=scene.msaa)
                studio = StudioRenderer(w, h, msaa=scene.msaa, look=StudioLook(**case.look))
            count = math.ceil(scene.duration * args.fps)
            try:
                output = folder / "comparison.mp4"
                with VideoWriter(output, w * 2, h + 64, args.fps, bitrate="24M") as writer:
                    for index in range(count):
                        when = index / args.fps
                        if use_exports:
                            left = next(frames_native).to_ndarray(format="rgba")
                            right = next(frames_studio).to_ndarray(format="rgba")
                        else:
                            left = scene.render_frame(when, renderer=native)
                            right = scene.render_frame(when, renderer=studio)
                        writer.write(pair_image(left, right, case, when, 64))
                    codec = writer.codec
                result["clip"] = dict(
                    file=f"{case.id}/comparison.mp4",
                    frames=count,
                    fps=args.fps,
                    width=w * 2,
                    height=h + 64,
                    codec=codec,
                    speed=1,
                    source_exports=exports,
                )
            finally:
                native.close()
                studio.close()
        result["status"] = "rendered"
    except Exception as exc:
        result["status"] = "error"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        (folder / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        sys.setprofile(None)
    result["observed_calls"] = sorted(observed)
    result["elapsed_seconds"] = time.perf_counter() - started
    dump(folder / "result.json", result)
    print(f"{case.id} {result['status']} {result['elapsed_seconds']:.1f}s {case.title}", flush=True)
    return 0 if result["status"] == "rendered" else 1


def coverage(entries):
    import proteinmotion

    mapping = {}
    for c in entries:
        for name in c["api"]:
            mapping.setdefault(name, []).append(c["id"])
    observed = {}
    for c in entries:
        for name in c.get("observed_calls", []):
            observed.setdefault(name, []).append(c["id"])
    public = []
    for name in proteinmotion.__all__:
        ids = sorted(
            set(
                mapping.get(name, [])
                + [i for key, v in mapping.items() if key.startswith(name + ".") for i in v]
            )
        )
        note = ""
        status = "visual comparison" if ids else "not separately rendered"
        if name == "charges_from_pqr":
            status, note = (
                "input + visual comparison",
                "Nonvisual PQR parser exercised through the R150 electrostatic network.",
            )
        elif name == "EEVEEOptions":
            status, note = (
                "other backend",
                "Blender EEVEE settings are outside the requested native/studio pair. Lens-control boundary shown in R097.",
            )
        elif name == "FocusPull":
            status, note = (
                "capability boundary",
                "R097 exercises the control; both requested backends lack EEVEE lens depth of field.",
            )
        public.append(dict(symbol=name, status=status, cases=ids, note=note))
    # Inventory all documented modules, with honest distinction between a feature
    # exercised directly and a method used internally by the rendering pipeline.
    catalog = json.loads((ROOT / "docs/reference/catalog.json").read_text())
    methods = []
    for module in catalog["modules"]:
        tree = ast.parse((ROOT / f"src/proteinmotion/{module}.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and f"{module}.{node.name}" in catalog["entries"]:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and not method.name.startswith("_"):
                        name = f"{node.name}.{method.name}"
                        methods.append(
                            dict(
                                symbol=name,
                                module=module,
                                line=method.lineno,
                                direct_cases=mapping.get(name, []),
                                observed_cases=observed.get(name, []),
                                note={
                                    "ProteinScene.construct": "Subclass hook; implementations are compared throughout the audit.",
                                    "Animation.apply": "Abstract animation hook; concrete animations are compared.",
                                    "Region.position": "Nonvisual getter of the parent object's translation; no independent rendering behavior.",
                                    "InteractionHighlight.text_progress": "Nonvisual read-only write-progress accessor; animated interaction lines are compared.",
                                    "ProteinScene.preview": "Interactive preview currently selects native only; no studio preview-window comparison.",
                                    "Renderer.draw": "GPU presentation path; this audit captures readback/export, not a swapchain window.",
                                    "EEVEE.render": "Blender EEVEE is outside the native/studio pair.",
                                    "EEVEE.close": "Blender EEVEE is outside the native/studio pair.",
                                }.get(name, ""),
                            )
                        )
    documented = []
    for key in catalog["entries"]:
        name = key.split(".", 1)[1]
        ids = set(mapping.get(name, []) + observed.get(name, []) + observed.get(name + ".__init__", []))
        ids.update(i for symbol, values in mapping.items() if symbol.startswith(name + ".") for i in values)
        documented.append(
            dict(symbol=key, cases=sorted(ids), note="EEVEE-only backend" if key.startswith("eevee.") else "")
        )
    return dict(
        exports=public,
        methods=methods,
        feature_cases=mapping,
        observed_calls=observed,
        documented_entries=documented,
        scope="Visual feature audit, not an exhaustive argument-combination test or a unit-test substitute. Public exports are accounted for; methods without direct case tags are listed explicitly. Private GPU implementation helpers are excluded.",
    )


def gallery(args):
    entries = []
    for case in cases():
        file = args.out / case.id / "result.json"
        if file.exists():
            entry = json.loads(file.read_text())
            # Resolve source links against the current, formatted audit source.
            target = case.factory.func if hasattr(case.factory, "func") else case.factory
            if hasattr(case.factory, "args") and case.factory.args:
                target = case.factory.args[0]
                target = target.func if hasattr(target, "func") else target
            entry["source"] = str(Path(inspect.getsourcefile(target)).relative_to(ROOT))
            entry["source_line"] = inspect.getsourcelines(target)[1]
            entries.append(entry)
        else:
            entries.append(
                dict(
                    id=case.id,
                    group=case.group,
                    title=case.title,
                    api=case.api,
                    notes=case.notes,
                    status="pending",
                    errors=[],
                    samples=[],
                    clip=None,
                )
            )
    sources = (
        sorted((ROOT / "src/proteinmotion").glob("**/*.py"))
        + sorted((ROOT / "src/proteinmotion/shaders").glob("*.wgsl"))
        + sorted((ROOT / "examples").glob("*.py"))
        + [Path(__file__), ROOT / "scripts/renderer_audit_gallery.html"]
    )
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        defaults=dict(
            still=[args.width, args.height],
            video_per_side=[args.video_width, args.video_width * 9 // 16],
            fps=args.fps,
        ),
        methodology="Native on left; studio on right. Identical scene, geometry, camera, time, background and MSAA. Default StudioLook unless specified. Movies contain both views in one frame, at 1× timeline speed. PNGs are direct lossless readbacks; JPEG/WebP previews and H.264 movies are compressed. Renderer output differences are retained.",
        cases=entries,
        coverage=coverage(entries),
    )
    dump(args.out / "manifest.json", manifest)
    dump(args.out / "coverage.json", manifest["coverage"])
    template = (ROOT / "scripts/renderer_audit_gallery.html").read_text()
    (args.out / "index.html").write_text(
        template.replace("/*AUDIT_DATA*/null", json.dumps(manifest).replace("</", "<\\/"))
    )
    (args.out / "README.md").write_text(
        "# Native / Studio renderer audit\n\nOpen index.html. Native is always left; studio is always right.\n\n"
        + manifest["methodology"]
        + "\n\n"
        + f"{len(entries)} cases; {sum(bool(c.get('clip')) for c in entries)} synchronized clips; "
        + f"{sum(len(c['samples']) for c in entries)} pairs of lossless stills.\n\n"
        + "Choose a case, inspect samples or play the clip, then flag it and add a note. 'Stamp time' appends the current clip time to your note. Export review downloads a JSON file. Notes are stored only in this browser's local storage until exported.\n\n"
        + "Run from the repository: `.venv/bin/python scripts/render_renderer_audit.py`. Re-render a case: add `--only R088`. The full render takes time; `--resume` skips cases already marked rendered. `--gallery-only` rebuilds the gallery without rendering.\n\n"
        + "Coverage is a visual audit of public features, not all possible parameters. `coverage.json` enumerates public exports and public methods, including methods without direct case tags. PQR parsing has no independent visual output. EEVEE is outside this two-backend audit; FocusPull's lack of blur in both native and studio is documented in R097. Preview-window interaction is not tested.\n\n"
        + "NMR model interpolation and threading are illustrative animations, not physical dynamics. R101/R104 use a synthetic ideal helix with explicit hydrogen atoms.\n"
    )
    rows = ["# Public API coverage", "", "| Export | Status | Cases |", "|---|---|---|"]
    for item in manifest["coverage"]["exports"]:
        rows.append(f"| {item['symbol']} | {item['status']} | {', '.join(item['cases']) or item['note']} |")
    (args.out / "coverage.md").write_text("\n".join(rows) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "output/renderer-audit")
    p.add_argument("--only", default="")
    p.add_argument("--worker")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gallery-only", action="store_true")
    p.add_argument("--stills-only", action="store_true")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    if not args.gallery_only:
        selected = set(args.only.split(",")) if args.only else None
        for case in cases():
            if selected and case.id not in selected:
                continue
            result = args.out / case.id / "result.json"
            if args.resume and result.exists() and json.loads(result.read_text())["status"] == "rendered":
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                case.id,
                "--out",
                str(args.out),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--video-width",
                str(args.video_width),
                "--fps",
                str(args.fps),
            ]
            if args.stills_only:
                command.append("--stills-only")
            subprocess.run(command, cwd=ROOT, check=False)
            gallery(args)
    manifest = gallery(args)
    counts = {
        status: sum(c["status"] == status for c in manifest["cases"])
        for status in ("rendered", "error", "pending")
    }
    print(json.dumps(counts), flush=True)
    return bool(counts["error"])


if __name__ == "__main__":
    raise SystemExit(main())
