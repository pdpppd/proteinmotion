import Link from "next/link";
import { ArrowRightIcon } from "@radix-ui/react-icons";
import Video from "./Video";

export default function HeroPlayer() {
  return (
    <div>
      <Video
        file="calmodulin-in-focus"
        title="Calmodulin with depth of field, helix close-ups, and molecular surfaces"
        autoplay
      />
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
        <span>1:08 · EEVEE · 60 fps</span>
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
