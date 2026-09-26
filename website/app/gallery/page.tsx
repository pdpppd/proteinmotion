import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRightIcon, DownloadIcon } from "@radix-ui/react-icons";
import Video from "@/components/Video";
import { asset, demoCommand, demos, repo } from "@/lib/config";
import { codeHTML } from "@/lib/content";
export const metadata: Metadata = {
  title: "Example gallery",
  description: "ProteinMotion videos with source code and rendering commands.",
};
export default function Gallery() {
  return (
    <main
      id="main"
      className="mx-auto max-w-[1400px] px-5 py-12 md:px-10 md:py-16"
    >
      <h1 className="text-4xl font-medium tracking-[-.045em] md:text-5xl">
        Video examples
      </h1>
      <p className="mt-5 max-w-2xl text-base leading-7 text-muted">
        Each video includes its Python script and render command. Calmodulin in
        focus plays at 1080p/60 fps. The tunnel, threading, ligand, side-chain,
        DNA, and RNA examples use 720p/60 fps. Each render command lists the
        settings used for its preview.
      </p>
      <div className="mt-14 grid gap-x-10 gap-y-16 md:grid-cols-2">
        {demos.map((demo) => (
          <section id={demo.id} key={demo.id} className="min-w-0 scroll-mt-24">
            <Video file={demo.file} title={demo.title} />
            <h2 className="mt-5 text-xl font-medium tracking-tight">
              {demo.title}
            </h2>
            <p className="mt-2 text-sm leading-7 text-muted">{demo.detail}</p>
            <div className="mt-4 flex flex-wrap items-center gap-5 text-xs">
              <span className="text-muted">{demo.duration}</span>
              <a
                href={`${repo}/blob/main/${demo.source}`}
                className="inline-flex items-center gap-1.5 font-medium text-accent"
              >
                View source <ArrowRightIcon />
              </a>
              <a
                href={asset(`media/${demo.file}.mp4`)}
                download
                className="inline-flex items-center gap-1.5 text-muted"
              >
                <DownloadIcon /> Download preview
              </a>
              {demo.pdb && (
                <span className="font-mono text-[10px] text-muted">
                  PDB {demo.pdb}
                </span>
              )}
            </div>
            <div
              className="mt-4"
              dangerouslySetInnerHTML={{
                __html: codeHTML(demoCommand(demo), "bash"),
              }}
            />
          </section>
        ))}
      </div>
      <div className="mt-14 border-t border-line pt-8">
        <p className="max-w-3xl text-sm leading-7 text-muted">
          The NMR examples interpolate between deposited models to show
          structural variation. The morph examples interpolate between two
          structures. These animations illustrate coordinate changes; physical
          dynamics require simulation data.
        </p>
        <Link
          href="/docs/rendering/"
          className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-accent"
        >
          Rendering methods <ArrowRightIcon />
        </Link>
      </div>
    </main>
  );
}
