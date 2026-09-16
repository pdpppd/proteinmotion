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
            Protein animation,
            <br />
            <span className="text-muted">written in Python.</span>
          </h1>
          <p className="mt-6 max-w-md text-[17px] leading-relaxed text-muted">
            Turn molecular structures into films. Compose cartoons, morphs,
            surfaces, and measured interactions with a small, expressive API.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/docs/getting-started/" className="button-primary">
              Make your first film <ArrowRightIcon />
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
          ["02", "Motion with intent", "Eased motion, color & opacity"],
          ["03", "A closer view", "Labels, rulers & interactions"],
          ["04", "Real structural ensembles", "NMR models & lazy MD playback"],
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
          <p className="eyebrow">AUTHOR THE SCENE</p>
          <h2 className="mt-4 max-w-sm text-3xl font-medium leading-tight tracking-[-.035em]">
            Small scripts.
            <br />
            Expressive molecular films.
          </h2>
          <p className="mt-5 max-w-sm text-sm leading-7 text-muted">
            If you know Manim, the rhythm is familiar: create an object, add it
            to a scene, then play animations. ProteinMotion adds molecular
            representations and a renderer built for native GPUs.
          </p>
          <Link
            href="/docs/molecular-example/"
            className="mt-6 inline-flex items-center gap-2 text-sm font-medium text-accent"
          >
            Read a complete script <ArrowRightIcon />
          </Link>
          <p className="mt-6 max-w-sm text-xs leading-6 text-muted">
            A standalone package with Manim-inspired syntax. Export clips for
            compositing into Manim or another editor.
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
            Less waiting between edits.
          </h2>
          <p className="mt-5 max-w-lg text-sm leading-7 text-muted">
            The 24-second, 1080p/60 fps cartoon NMR film exported in 3.69
            seconds, including rendering and hardware encoding. A single local
            run; loading and scene preparation are measured separately.
          </p>
          <Link
            href="/docs/validation/"
            className="mt-5 inline-flex items-center gap-2 text-sm font-medium text-accent"
          >
            See the methods and raw measurements <ArrowRightIcon />
          </Link>
        </div>
        <div className="grid grid-cols-2 gap-8 self-center">
          <div>
            <p className="font-mono text-4xl tracking-tight">116</p>
            <p className="mt-3 text-xs leading-6 text-muted">
              Deposited NMR conformers
              <br />
              in the example ensemble
            </p>
          </div>
          <div>
            <p className="font-mono text-4xl tracking-tight">91</p>
            <p className="mt-3 text-xs leading-6 text-muted">
              Passing local tests
              <br />
              including native GPU checks
            </p>
          </div>
        </div>
      </section>
      <section className="mt-6 flex flex-wrap items-center justify-between gap-6 rounded-xl bg-accent-soft px-7 py-8">
        <div>
          <h2 className="text-lg font-medium tracking-tight">
            Start with a real structure.
          </h2>
          <p className="mt-2 text-sm text-muted">
            Ubiquitin, calmodulin, troponin C, and GroEL/GroES examples are
            included.
          </p>
        </div>
        <Link href="/gallery/" className="button-secondary bg-white">
          Explore the gallery <ArrowRightIcon />
        </Link>
      </section>
    </main>
  );
}
