"""Check a noneditable installation, including resources, with no source checkout imports.

Run with the installed environment's Python, from outside the repository.
Pass --render on a native GPU to verify a short actual movie and all decoded frames.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

import proteinmotion
from proteinmotion.cli import load_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("proteinmotion")
    package = Path(proteinmotion.__file__).resolve()
    root = Path(distribution.locate_file("proteinmotion")).resolve()
    assert package.parent == root, f"Imported a different checkout: {package}"
    direct = distribution.read_text("direct_url.json")
    assert not direct or not json.loads(direct).get("dir_info", {}).get("editable", False)
    assert proteinmotion.__version__ == distribution.version
    for relative in [
        "shaders/molecule.wgsl",
        "shaders/text.wgsl",
        "shaders/nv12.wgsl",
        "shaders/transparency.wgsl",
        "fonts/SourceSans3-Regular.otf",
        "fonts/SourceSans3-Semibold.otf",
        "fonts/OFL.txt",
        "licenses/Manim-LICENSE.txt",
        "_skill/SKILL.md",
        "_skill/agents/openai.yaml",
        "_skill/references/animation.md",
        "_skill/references/annotations-and-interactions.md",
        "_skill/assets/film.py",
        "_skill/assets/1ubq.cif",
    ]:
        assert files("proteinmotion").joinpath(relative).is_file(), f"Missing resource: {relative}"
    with tempfile.TemporaryDirectory(prefix="proteinmotion-install-check-") as temporary:
        work = Path(temporary)
        movie = work / "movie"
        skill = work / "skill"
        for command in [["--version"], ["init", str(movie)], ["install-skill", "--path", str(skill)]]:
            subprocess.run([sys.executable, "-I", "-m", "proteinmotion", *command], cwd=work, check=True)
        scene = load_scene(movie / "film.py", "ProteinMovie", width=640, height=360, fps=30).build()
        assert 15 < scene.duration < 30
        assert (skill / "SKILL.md").is_file()
        if args.render:
            import av

            subprocess.run([sys.executable, "-I", "-m", "proteinmotion", "doctor"], cwd=work, check=True)
            # The first seconds exercise molecule shaders, vector text, fonts and encoding.
            scene.duration = 2
            output = work / "installed.mp4"
            report = scene.render(output, progress=False)
            with av.open(output) as video:
                frames = list(video.decode(video=0))
            assert len(frames) == 60 and frames[-1].width == 640
            assert frames[-1].to_ndarray(format="rgb24").std() > 5
            print(json.dumps(report, indent=2))
    print(f"Verified installed ProteinMotion {distribution.version}: {package}")


if __name__ == "__main__":
    main()
