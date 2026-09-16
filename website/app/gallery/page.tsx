import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRightIcon, DownloadIcon } from "@radix-ui/react-icons";
import Video from "@/components/Video";
import { asset, demos, repo } from "@/lib/config";
import { codeHTML } from "@/lib/content";
export const metadata: Metadata = {
  title: "Example gallery",
  description:
    "Real films rendered with ProteinMotion, with source code and reproducible commands.",
};
export default function Gallery() {
  return (
    <main
      id="main"
      className="mx-auto max-w-[1400px] px-5 py-12 md:px-10 md:py-16"
    >
      <p className="eyebrow">RENDERED WITH PROTEINMOTION</p>
      <h1 className="mt-4 text-4xl font-medium tracking-[-.045em] md:text-5xl">
        From a structure to a story.
      </h1>
      <p className="mt-5 max-w-2xl text-base leading-7 text-muted">
        Actual package output, with the source and command for every film. These
        web previews are compressed to 720p/30 fps; original renders were
        1080p/60 fps.
      </p>
      <div className="mt-14 grid gap-x-10 gap-y-16 md:grid-cols-2">
        {demos.map((demo, i) => (
          <section id={demo.id} key={demo.id} className="min-w-0 scroll-mt-24">
            <div className="mb-3 flex justify-between font-mono text-[10px] tracking-wide text-muted">
              <span>
                {String(i + 1).padStart(2, "0")} / {demo.label}
              </span>
              <span>{demo.duration}</span>
            </div>
            <Video file={demo.file} title={demo.title} />
            <h2 className="mt-5 text-xl font-medium tracking-tight">
              {demo.title}
            </h2>
            <p className="mt-2 text-sm leading-7 text-muted">{demo.detail}</p>
            <div className="mt-4 flex flex-wrap items-center gap-5 text-xs">
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
              <span className="font-mono text-[10px] text-muted">
                PDB {demo.pdb}
              </span>
            </div>
            <div
              className="mt-4"
              dangerouslySetInnerHTML={{
                __html: codeHTML(
                  `proteinmotion render ${demo.source} ${demo.scene} \
  -o ${demo.file}.mp4 --fps 60`,
                  "bash",
                ),
              }}
            />
          </section>
        ))}
      </div>
      <div className="mt-14 border-t border-line pt-8">
        <p className="max-w-3xl text-sm leading-7 text-muted">
          NMR model order is not a physical time sequence. Morphs and state
          interpolation are visual transitions, without energy minimization or a
          claim of physical pathways. The contact matcher is bounded and does
          not promise global optimality on large inputs.
        </p>
        <Link
          href="/docs/rendering/"
          className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-accent"
        >
          Methods and limits <ArrowRightIcon />
        </Link>
      </div>
    </main>
  );
}
