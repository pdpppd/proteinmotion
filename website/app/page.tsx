import Link from "next/link";
import { ArrowRightIcon, GitHubLogoIcon } from "@radix-ui/react-icons";
import HeroPlayer from "@/components/HeroPlayer";
import { codeHTML } from "@/lib/content";
import { repo } from "@/lib/config";
const snippet = `class MyFilm(ProteinScene):
    def construct(self):
        p = Protein.from_file("protein.cif").cartoon()
        self.add(p)
        self.camera.frame(p)

        helix = p.select(residues=(23, 34))
        self.play(Colorize(helix, "#50e0d0"), run_time=1.5)
        self.play(Write(helix.callout("α helix")))
        self.focus(helix, run_time=1.5)
        self.play(PlayTrajectory(p), run_time=8)`;
export default function Home() {
  return (
    <main id="main" className="mx-auto max-w-[1400px] px-5 md:px-10">
      <section className="grid items-center gap-12 pb-14 pt-12 lg:grid-cols-[.85fr_1.15fr] lg:gap-14 lg:pb-20 lg:pt-20">
        <div>
          <div className="mb-7 flex items-center gap-2.5 text-xs font-medium text-accent">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            PYTHON API · NATIVE METAL
          </div>
          <h1 className="max-w-lg text-[2.7rem] font-medium leading-[1.05] tracking-[-.055em] sm:text-[3.7rem]">
            Protein animation <br />
            <span className="text-muted">in Python</span>
          </h1>
          <p className="mt-6 max-w-md text-[17px] leading-relaxed text-muted">
            Load a protein structure or trajectory, add animations and labels,
            and export a video. Render cartoons, ball-and-stick models, ribbons,
            and molecular surfaces.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/docs/getting-started/" className="button-primary">
              Get started <ArrowRightIcon />
            </Link>
            <a href={repo} className="button-secondary">
              <GitHubLogoIcon /> View source
            </a>
          </div>
          <p className="mt-5 font-mono text-[11px] text-muted">
            Python 3.11+ · Apple silicon · MIT licensed
          </p>
        </div>
        <HeroPlayer />
      </section>
      <section
        aria-label="Capabilities"
        className="grid grid-cols-2 gap-x-6 gap-y-6 border-y border-line py-7 md:grid-cols-4"
      >
        {[
          [
            "01",
            "Molecular representations",
            "Cartoon, ribbon, atoms & surfaces",
          ],
          ["02", "Animation", "Movement, color & transparency"],
          ["03", "Labels & measurements", "Residues, distances & interactions"],
          ["04", "States & trajectories", "NMR ensembles & MD playback"],
        ].map(([n, title, detail]) => (
          <div key={n}>
            <span className="font-mono text-[10px] text-accent">{n}</span>
            <h2 className="mt-2 text-sm font-medium">{title}</h2>
            <p className="mt-1.5 text-xs leading-relaxed text-muted">
              {detail}
            </p>
          </div>
        ))}
      </section>
      <section className="grid gap-10 py-16 lg:grid-cols-[.8fr_1.2fr] lg:gap-24 lg:py-24">
        <div>
          <p className="eyebrow">PYTHON SCENES</p>
          <h2 className="mt-4 max-w-sm text-3xl font-medium leading-tight tracking-[-.035em]">
            Define each animation <br />
            in a Python script.
          </h2>
          <p className="mt-5 max-w-sm text-sm leading-7 text-muted">
            Create a protein object, add it to a scene, and call play() to
            animate it. The API follows Manim’s scene syntax. Set the duration
            of each animation and combine animations in one call to run them
            together.
          </p>
          <Link
            href="/docs/molecular-example/"
            className="mt-6 inline-flex items-center gap-2 text-sm font-medium text-accent"
          >
            Read a complete script <ArrowRightIcon />
          </Link>
          <p className="mt-6 max-w-sm text-xs leading-6 text-muted">
            ProteinMotion runs as a standalone package. Use the exported videos
            in Manim, presentations, or a video editor.
          </p>
        </div>
        <div className="min-w-0">
          <div className="mb-3 flex justify-between font-mono text-[11px] text-muted">
            <span>scene.py</span>
            <span>INSIDE A PROTEINSCENE</span>
          </div>
          <div dangerouslySetInnerHTML={{ __html: codeHTML(snippet) }} />
        </div>
      </section>
      <section className="grid gap-10 border-t border-line py-12 md:grid-cols-[1.1fr_.9fr] md:gap-24">
        <div>
          <p className="eyebrow">MEASURED ON APPLE M3 MAX</p>
          <h2 className="mt-4 text-3xl font-medium tracking-[-.035em]">
            Rendering performance
          </h2>
          <p className="mt-5 max-w-lg text-sm leading-7 text-muted">
            A 24-second NMR cartoon video at 1080p/60 fps exported in 3.69
            seconds on an Apple M3 Max. This single run includes rendering, GPU
            readback, and hardware encoding. Loading and scene construction were
            timed separately.
          </p>
          <Link
            href="/docs/validation/"
            className="mt-5 inline-flex items-center gap-2 text-sm font-medium text-accent"
          >
            View benchmarks <ArrowRightIcon />
          </Link>
        </div>
        <div className="grid grid-cols-2 gap-8 self-center">
          <div>
            <p className="font-mono text-4xl tracking-tight">116</p>
            <p className="mt-3 text-xs leading-6 text-muted">
              Ubiquitin NMR models
              <br />
              in the example ensemble
            </p>
          </div>
          <div>
            <p className="font-mono text-4xl tracking-tight">107</p>
            <p className="mt-3 text-xs leading-6 text-muted">
              Tests passed for v0.7.0
              <br />
              including native GPU checks
            </p>
          </div>
        </div>
      </section>
      <section className="border-t border-line py-10">
        <h2 className="text-xl font-medium tracking-tight">
          Use with an AI agent
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-7 text-muted">
          The ProteinMotion Movies skill provides instructions and examples for
          writing scenes, rendering videos, and checking the results. Use it
          with an agent that can read local files and run Python commands.
        </p>
        <Link
          href="/docs/agent-skill/"
          className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-accent"
        >
          Set up the skill <ArrowRightIcon />
        </Link>
      </section>
      <section className="mt-6 flex flex-wrap items-center justify-between gap-6 rounded-xl bg-accent-soft px-7 py-8">
        <div>
          <h2 className="text-lg font-medium tracking-tight">
            Example structures and scripts
          </h2>
          <p className="mt-2 text-sm text-muted">
            Ubiquitin, calmodulin, troponin C, and GroEL/GroES examples are
            included.
          </p>
        </div>
        <Link href="/gallery/" className="button-secondary bg-white">
          View examples <ArrowRightIcon />
        </Link>
      </section>
    </main>
  );
}
