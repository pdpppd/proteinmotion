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
            PYTHON API · METAL · EEVEE
          </div>
          <h1 className="max-w-lg text-[2.7rem] font-medium leading-[1.05] tracking-[-.055em] sm:text-[3.7rem]">
            Molecular animation <br />
            <span className="text-muted">in Python</span>
          </h1>
          <p className="mt-6 max-w-md text-[17px] leading-relaxed text-muted">
            Animate proteins, DNA, and RNA from a structure or trajectory. Add
            motion and labels, then export a video. Render cartoons, nucleotide
            bases, ball-and-stick models, and molecular surfaces.
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
      <section className="grid gap-8 border-t border-line py-10 md:grid-cols-2 lg:grid-cols-4">
        {[
          [
            "DNA and RNA",
            "Draw bases as slabs, rings, sticks, or ladder rods. Highlight selected nucleotides and show molecular surfaces.",
            "dna-rna",
          ],
          [
            "Numerical properties",
            "Color residues by B factors, RMSF, or custom values. Use cartoon thickness and a color scale to show the range.",
            "numerical-properties",
          ],
          [
            "Plots that follow the movie",
            "Show distance traces, contact maps, and a sequence strip alongside trajectory playback.",
            "synchronized-plots",
          ],
          [
            "Density maps and slices",
            "Load an MRC or CCP4 map, change its contour level, and move a slice through the volume.",
            "density-maps",
          ],
        ].map(([title, description, slug]) => (
          <div key={slug}>
            <h2 className="text-xl font-medium tracking-tight">{title}</h2>
            <p className="mt-3 text-sm leading-7 text-muted">{description}</p>
            <Link
              href={`/docs/${slug}/`}
              className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-accent"
            >
              View code and output <ArrowRightIcon />
            </Link>
          </div>
        ))}
      </section>
      <section className="border-t border-line py-10">
        <h2 className="text-xl font-medium tracking-tight">
          Depth of field with EEVEE
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-7 text-muted">
          Set lens focus on selected residues and animate focus changes as the
          camera moves. The homepage film uses EEVEE for helix close-ups,
          transparent surroundings, and molecular surfaces. Install Blender
          separately to render the scene.
        </p>
        <Link
          href="/docs/calmodulin-in-focus/"
          className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-accent"
        >
          View the film and full script <ArrowRightIcon />
        </Link>
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
            Protein, DNA double-helix, and transfer RNA examples include their
            input structures and full scripts.
          </p>
        </div>
        <Link href="/gallery/" className="button-secondary bg-white">
          View examples <ArrowRightIcon />
        </Link>
      </section>
    </main>
  );
}
