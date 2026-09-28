"""Decode every audit movie and verify every image; create printable overview sheets."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "output/renderer-audit")
    args = p.parse_args()
    out = args.out.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    report = dict(created=datetime.now(timezone.utc).isoformat(), errors=[], cases=[], frames=0, images=0)
    started = time.perf_counter()
    for case in manifest["cases"]:
        entry = dict(id=case["id"], images=0, decoded_frames=0, status="checked")
        if case["status"] != "rendered":
            report["errors"].append(f"{case['id']}: render status {case['status']}")
        for sample in case["samples"]:
            for field in ("native", "studio", "pair", "thumb"):
                file = out / sample[field]
                try:
                    with Image.open(file) as im:
                        size = im.size
                        im.verify()
                    if field in ("native", "studio") and size != (sample["width"], sample["height"]):
                        raise ValueError(f"dimensions {size}")
                    entry["images"] += 1
                except Exception as exc:
                    report["errors"].append(f"{sample[field]}: {exc}")
        clip = case.get("clip")
        if clip:
            try:
                with av.open(str(out / clip["file"])) as video:
                    stream = video.streams.video[0]
                    if float(stream.average_rate) != clip["fps"]:
                        raise ValueError(f"frame rate {stream.average_rate}")
                    pts = None
                    for frame in video.decode(stream):
                        if (frame.width, frame.height) != (clip["width"], clip["height"]):
                            raise ValueError("frame dimensions differ from manifest")
                        if pts is not None and frame.pts <= pts:
                            raise ValueError("non-monotonic frame timestamps")
                        pts = frame.pts
                        entry["decoded_frames"] += 1
                    if entry["decoded_frames"] != clip["frames"]:
                        raise ValueError(f"{entry['decoded_frames']} frames; expected {clip['frames']}")
                # The export-integration case also retains the uncomposited files.
                for name, export in clip.get("source_exports", {}).items():
                    count = 0
                    with av.open(str(out / case["id"] / f"{name}-export.mp4")) as source:
                        for _ in source.decode(video=0):
                            count += 1
                    if count != export["frames"]:
                        raise ValueError(f"{name} original export has {count} frames")
            except Exception as exc:
                report["errors"].append(f"{clip['file']}: {exc}")
        report["cases"].append(entry)
        report["frames"] += entry["decoded_frames"]
        report["images"] += entry["images"]
        print(f"{case['id']}: {entry['images']} images, {entry['decoded_frames']} decoded frames", flush=True)

    # Twelve side-by-side scenes per 4K-width board. These are previews; PNGs are the
    # source of truth. Captions remain legible without opening the interactive gallery.
    font_path = ROOT / "src/proteinmotion/fonts/SourceSans3-Regular.otf"
    font = ImageFont.truetype(str(font_path), 29)
    title_font = ImageFont.truetype(str(font_path), 42)
    complete = [c for c in manifest["cases"] if c["samples"]]
    sheets = []
    for start in range(0, len(complete), 12):
        board = Image.new("RGB", (3840, 2020), "#10151c")
        d = ImageDraw.Draw(board)
        d.text((32, 22), "PROTEINMOTION / NATIVE ← → STUDIO / VISUAL AUDIT", fill="#eaf0f8", font=title_font)
        for i, case in enumerate(complete[start : start + 12]):
            sample_index = (
                min(2, len(case["samples"]) - 1)
                if case["group"] in ("Cutaways", "Atom detail", "Threading")
                else 0
            )
            with Image.open(out / case["samples"][sample_index]["pair"]) as image:
                image = image.convert("RGB")
                image.thumbnail((1260, 402), Image.Resampling.LANCZOS)
                x, y = 10 + (i % 3) * 1280, 100 + (i // 3) * 476
                board.paste(image, (x, y))
                d.text((x + 12, y + 405), f"{case['id']} · {case['title']}", fill="#eaf0f8", font=font)
        path = out / f"overview-{start // 12 + 1:02d}.jpg"
        board.save(path, quality=93, subsampling=0)
        sheets.append(path.name)
    report["overview_sheets"] = sheets
    report["seconds"] = time.perf_counter() - started
    report["passed"] = not report["errors"]
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    return not report["passed"]


if __name__ == "__main__":
    raise SystemExit(main())
