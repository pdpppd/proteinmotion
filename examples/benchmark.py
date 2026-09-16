"""Reproducible end-to-end and render-only timings; no claim of universal peak performance."""

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from showcase import DATA, Showcase

from proteinmotion import Protein, ProteinScene, Rotate, linear
from proteinmotion.renderer import Renderer
from proteinmotion.structure import Topology


def replicate(protein, copies):
    """One GPU object containing translated copies; a synthetic scale test, not a biological assembly."""
    t = protein.topology
    atoms, residues, bonds, chains, xyz = [], [], [], [], []
    edge = int(np.ceil(copies ** (1 / 3)))
    for i in range(copies):
        atom_offset, residue_offset = len(atoms), len(residues)
        chain_name = f"copy{i}"
        atoms.extend(
            replace(a, chain=chain_name, residue_index=a.residue_index + residue_offset) for a in t.atoms
        )
        residues.extend(
            replace(
                r,
                chain=chain_name,
                ca=r.ca + atom_offset if r.ca >= 0 else -1,
                oxygen=r.oxygen + atom_offset if r.oxygen >= 0 else -1,
            )
            for r in t.residues
        )
        bonds.append(t.bonds + atom_offset)
        chains.extend(c + residue_offset for c in t.chains)
        offset = np.array([i % edge, (i // edge) % edge, i // (edge * edge)]) * 38.0
        xyz.append(protein.positions + offset)
    topology = Topology(tuple(atoms), tuple(residues), np.concatenate(bonds), tuple(chains))
    return Protein(topology, np.concatenate(xyz)).center()


def benchmark(width, height, frames, copies, representation):
    p = replicate(Protein.from_file(DATA), copies)
    getattr(p, representation)()
    s = ProteinScene(width=width, height=height, fps=60)
    s.add(p)
    s.camera.frame(p, margin=1.1, aspect=width / height)
    s.play(Rotate(p, 2 * np.pi), run_time=frames / 60, rate_func=linear)
    with Renderer(width, height) as renderer:
        renderer.draw(s.seek(0), s.camera, s.background)
        renderer.device.queue.read_texture({"texture": renderer.texture}, {"bytes_per_row": 256}, (1, 1, 1))
        start = time.perf_counter()
        for i in range(frames):
            renderer.draw(s.seek(i / 60), s.camera, s.background)
        renderer.device.queue.read_texture({"texture": renderer.texture}, {"bytes_per_row": 256}, (1, 1, 1))
        seconds = time.perf_counter() - start
        info = renderer.adapter_info
    return {
        "kind": "render-only",
        "representation": representation,
        "atoms": len(p.topology.atoms),
        "residues": len(p.topology.residues),
        "frames": frames,
        "width": width,
        "height": height,
        "msaa": 4,
        "seconds": seconds,
        "fps": frames / seconds,
        "adapter": info,
        "notes": "Warm shader/geometry caches; includes CPU scene submission and waits for GPU completion. "
        "Excludes readback/encoding and initial load/upload. Replicated ubiquitin scale test.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    records = []
    for copies, representation in [
        (1, "cartoon"),
        (1, "ball_and_stick"),
        (100, "cartoon"),
        (100, "ball_and_stick"),
    ]:
        result = benchmark(1920, 1080, args.frames, copies, representation)
        print(f"{copies * 602:6d} atoms · {representation:15s} · {result['fps']:.1f} fps", flush=True)
        records.append(result)
    movie = args.output.with_suffix(".mp4")
    s = Showcase(width=1920, height=1080, fps=60)
    result = s.render(movie)
    result.update(
        kind="end-to-end export",
        width=1920,
        height=1080,
        msaa=4,
        notes="All representations, camera, deformation, synthetic trajectory; includes renderer startup, "
        "GPU upload, readback and VideoToolbox encode. Scene compilation is excluded.",
    )
    records.append(result)
    data = {"platform": platform.platform(), "python": platform.python_version(), "results": records}
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
