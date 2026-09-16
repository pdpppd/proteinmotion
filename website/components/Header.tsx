import Link from "next/link";
import { GitHubLogoIcon } from "@radix-ui/react-icons";
import Search from "./Search";
import { searchData } from "@/lib/content";
import { repo } from "@/lib/config";
export default function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-paper/95 backdrop-blur-sm">
      <div className="mx-auto flex h-[76px] max-w-[1400px] items-center justify-between gap-3 px-5 md:px-10">
        <Link
          href="/"
          className="flex items-center gap-2.5 font-semibold tracking-[-.035em] text-xl"
        >
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          ProteinMotion
          <span className="ml-1 hidden rounded border border-line px-1.5 py-1 font-mono text-[10px] font-normal tracking-normal text-muted lg:inline">
            v0.4
          </span>
        </Link>
        <nav
          aria-label="Main navigation"
          className="flex items-center gap-4 md:gap-7"
        >
          <Link
            href="/docs/getting-started/"
            className="hidden text-sm text-muted transition hover:text-ink sm:block"
          >
            Docs
          </Link>
          <Link
            href="/gallery/"
            className="text-sm text-muted transition hover:text-ink"
          >
            Gallery
          </Link>
          <Search items={searchData()} />
          <a
            href={repo}
            aria-label="ProteinMotion on GitHub"
            className="hidden text-ink/80 transition hover:text-accent sm:block"
          >
            <GitHubLogoIcon className="h-5 w-5" />
          </a>
        </nav>
      </div>
    </header>
  );
}
