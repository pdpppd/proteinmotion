"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  Cross2Icon,
  MagnifyingGlassIcon,
  ArrowRightIcon,
} from "@radix-ui/react-icons";
type Item = { title: string; slug: string; description: string; text: string };
export default function Search({ items }: { items: Item[] }) {
  const modal = useRef<HTMLDialogElement>(null);
  const [query, setQuery] = useState("");
  const open = () => {
    setQuery("");
    modal.current?.showModal();
  };
  const close = () => modal.current?.close();
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        open();
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, []);
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const results = items
    .filter((item) =>
      terms.every((term) =>
        (item.title + " " + item.text).toLowerCase().includes(term),
      ),
    )
    .slice(0, 8);
  return (
    <>
      <button
        className="flex items-center gap-2 rounded-lg border border-line bg-white px-3 py-2 text-sm text-muted transition hover:border-ink/30 active:scale-[.98]"
        onClick={open}
        aria-label="Search documentation"
      >
        <MagnifyingGlassIcon className="h-4 w-4" />
        <span className="hidden sm:inline">Search docs</span>
        <kbd className="ml-5 hidden font-mono text-xs text-muted/70 md:inline">
          ⌘ K
        </kbd>
      </button>
      <dialog
        ref={modal}
        aria-labelledby="search-title"
        className="search-dialog"
        onClick={(e) => {
          if (e.target === e.currentTarget) close();
        }}
      >
        <div className="border-b border-line p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 id="search-title" className="font-semibold">
              Search documentation
            </h2>
            <button
              onClick={close}
              className="rounded p-1.5 hover:bg-paper"
              aria-label="Close search"
            >
              <Cross2Icon />
            </button>
          </div>
          <label htmlFor="docs-search" className="sr-only">
            Search by concept or API name
          </label>
          <input
            id="docs-search"
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Try focus, contact map, or trajectory…"
            className="w-full rounded-lg border border-line bg-paper px-4 py-3 text-base outline-none focus:border-accent"
          />
        </div>
        <div className="max-h-[60dvh] overflow-y-auto p-2">
          <p className="px-3 py-2 text-xs text-muted" role="status">
            {terms.length
              ? `${results.length} matching pages`
              : "Browse the documentation"}
          </p>
          {results.length ? (
            results.map((item) => (
              <Link
                onClick={close}
                key={item.slug}
                href={`/docs/${item.slug}/`}
                className="group flex items-center justify-between rounded-lg px-3 py-3 hover:bg-paper focus:bg-paper"
              >
                <div>
                  <div className="text-sm font-semibold">{item.title}</div>
                  <p className="mt-1 text-sm leading-relaxed text-muted">
                    {item.description}
                  </p>
                </div>
                <ArrowRightIcon className="ml-4 shrink-0 text-muted group-hover:text-accent" />
              </Link>
            ))
          ) : (
            <div className="px-3 py-8">
              <p className="font-medium">No matching pages</p>
              <p className="mt-2 text-sm text-muted">
                Try an API name such as “Focus”, or a broader term such as
                “animation”.
              </p>
            </div>
          )}
        </div>
      </dialog>
    </>
  );
}
