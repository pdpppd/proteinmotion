import Link from "next/link";
import { ArrowRightIcon } from "@radix-ui/react-icons";
import Video from "./Video";

export default function HeroPlayer() {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3 font-mono text-[10px] tracking-wide text-muted">
        <span>THE COMPLETE FEATURE TOUR</span>
        <span>01:06 · 10 CHAPTERS</span>
      </div>
      <Video
        file="showcase"
        title="ProteinMotion feature tour: representations, motion, labels, interactions and large structures"
        autoplay
      />
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
        <span>One film. From a single residue to 58,870 atoms.</span>
        <Link
          href="/docs/showcase/"
          className="inline-flex items-center gap-1.5 font-medium text-accent"
        >
          Chapters & source <ArrowRightIcon />
        </Link>
      </div>
    </div>
  );
}
