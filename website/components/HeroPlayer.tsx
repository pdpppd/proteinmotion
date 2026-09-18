import Link from "next/link";
import { ArrowRightIcon } from "@radix-ui/react-icons";
import Video from "./Video";

export default function HeroPlayer() {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3 font-mono text-[10px] tracking-wide text-muted">
        <span>CALMODULIN IN FOCUS</span>
        <span>01:08 · EEVEE · 60 FPS</span>
      </div>
      <Video
        file="calmodulin-in-focus"
        title="Calmodulin with depth of field, helix close-ups, and molecular surfaces"
        autoplay
      />
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
        <span>Depth of field · atomic detail · surfaces</span>
        <Link
          href="/docs/calmodulin-in-focus/"
          className="inline-flex items-center gap-1.5 font-medium text-accent"
        >
          Chapters and code <ArrowRightIcon />
        </Link>
      </div>
    </div>
  );
}
