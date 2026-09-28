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
        "_gpu.py",
        "_blender_worker.py",
        "_eevee_geometry.py",
        "_thread_geometry.py",
        "studio.py",
        "looks.py",
        "styles.py",
        "timeline.py",
        "nucleic.py",
        "shaders/molecule.wgsl",
        "shaders/text.wgsl",
        "shaders/panel.wgsl",
        "shaders/nv12.wgsl",
        "shaders/transparency.wgsl",
        "shaders/studio_cartoon.wgsl",
        "shaders/studio_post.wgsl",
        "shaders/studio_splat.wgsl",
        "shaders/studio_surface.wgsl",
        "fonts/SourceSans3-Regular.otf",
        "fonts/SourceSans3-Semibold.otf",
        "fonts/OFL.txt",
        "licenses/Manim-LICENSE.txt",
        "_skill/SKILL.md",
        "_skill/agents/openai.yaml",
        "_skill/references/animation.md",
        "_skill/references/data-visualization.md",
        "_skill/references/nucleic-acids.md",
        "_skill/references/annotations-and-interactions.md",
        "_skill/references/ligands-and-side-chains.md",
        "_skill/references/cutaways-and-threading.md",
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
            check_studio(movie / "1ubq.cif", work)
    print(f"Verified installed ProteinMotion {distribution.version}: {package}")


def check_studio(structure, work):
    """Exercise the installed authoring API, geometry, film effects and video encoder."""
    import av
    import numpy as np

    from proteinmotion import (
        FadeIn,
        FadeOut,
        Protein,
        ProteinScene,
        Representation,
        Reveal,
        StudioLook,
        Text,
        Thread,
        Write,
    )
    from proteinmotion.studio import StudioRenderer

    class InstalledStudio(ProteinScene):
        renderer = "studio"
        look = StudioLook(lighting="soft", material="medium", effects="film")

        def construct(self):
            protein = Protein.from_file(structure, chains="A").cartoon().center()
            helix = protein.select(residue_range=(23, 34))
            self.camera.frame(protein)
            self.play(Thread(protein, glow=0).during(1), Write(Text("Installed Studio")).during(0.5))
            self.play(Representation(protein, "surface", kind="vdw", grid_spacing=0.8), run_time=0.5)
            self.play(Reveal(self.camera, helix, padding=2), run_time=0.5)
            self.play(Representation(protein, "ball_and_stick"), run_time=0.5)
            self.play(FadeOut(protein), run_time=0.25)
            self.play(FadeIn(protein), run_time=0.25)

    scene = InstalledStudio(width=640, height=360, fps=30).build()
    with StudioRenderer(scene.width, scene.height, msaa=scene.msaa, look=scene.look) as renderer:
        times = (0.7, 1.5, 2, 2.5, 3)
        images = [scene.render_frame(time, renderer=renderer) for time in times]
        for time, expected in reversed(list(zip(times, images))):
            np.testing.assert_array_equal(scene.render_frame(time, renderer=renderer), expected)
        assert all(image[:, :, :3].std() > 5 for image in images)
        assert all(np.any(a != b) for a, b in zip(images, images[1:]))
    output = work / "installed-studio.mp4"
    report = scene.render(output, progress=False)
    with av.open(output) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == 90 and (frames[-1].width, frames[-1].height) == (640, 360)
    assert frames[-1].to_ndarray(format="rgb24").std() > 5
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
