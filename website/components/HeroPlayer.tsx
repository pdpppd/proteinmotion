import Link from "next/link";
import { ArrowRightIcon } from "@radix-ui/react-icons";
import Video from "./Video";

export default function HeroPlayer() {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3 font-mono text-[10px] tracking-wide text-muted">
        <span>FEATURE DEMO</span>
        <span>01:41 · 60 FPS</span>
      </div>
      <Video
        file="showcase"
        title="ProteinMotion demo with calmodulin and troponin C"
        autoplay
      />
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
        <span>Calmodulin → troponin C</span>
        <Link
          href="/docs/showcase/"
          className="inline-flex items-center gap-1.5 font-medium text-accent"
        >
          Chapters and code <ArrowRightIcon />
        </Link>
      </div>
    </div>
  );
}
