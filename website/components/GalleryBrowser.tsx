"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  Cross2Icon,
  DownloadIcon,
  ExternalLinkIcon,
  MagnifyingGlassIcon,
} from "@radix-ui/react-icons";
import type { GalleryCategory, GalleryItem } from "@/lib/gallery";

const ALL = "all";
const seconds = (value: number) =>
  value >= 60
    ? `${Math.floor(value / 60)}:${String(Math.round(value % 60)).padStart(2, "0")}`
    : `${value.toFixed(value < 10 ? 1 : 0)} s`;

export default function GalleryBrowser({
  items,
  categories,
}: {
  items: GalleryItem[];
  categories: GalleryCategory[];
}) {
  const [category, setCategory] = useState(ALL);
  const [query, setQuery] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);
  const search = useRef<HTMLInputElement>(null);
  const tabs = useRef<HTMLDivElement>(null);
  const titles = useMemo(
    () => Object.fromEntries(categories.map((c) => [c.id, c.title])),
    [categories],
  );
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matching = useMemo(
    () =>
      items.filter((item) => terms.every((term) => item.search.includes(term))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [items, query],
  );
  const counts = useMemo(() => {
    const result: Record<string, number> = { [ALL]: matching.length };
    for (const item of matching)
      result[item.category] = (result[item.category] ?? 0) + 1;
    return result;
  }, [matching]);
  const visible =
    category === ALL
      ? matching
      : matching.filter((item) => item.category === category);
  const active = categories.find((c) => c.id === category);

  // Restore ?category= and #example from the URL, and follow back/forward.
  useEffect(() => {
    const sync = () => {
      const params = new URLSearchParams(window.location.search);
      const wanted = params.get("category");
      setCategory(
        wanted && categories.some((c) => c.id === wanted) ? wanted : ALL,
      );
      const hash = decodeURIComponent(window.location.hash.slice(1));
      const id = items.some((i) => i.id === hash)
        ? hash
        : items.some((i) => i.id === `film-${hash}`)
          ? `film-${hash}`
          : null;
      setOpenId(id);
    };
    sync();
    window.addEventListener("popstate", sync);
    window.addEventListener("hashchange", sync);
    return () => {
      window.removeEventListener("popstate", sync);
      window.removeEventListener("hashchange", sync);
    };
  }, [items, categories]);

  const chooseCategory = (id: string) => {
    setCategory(id);
    const url = new URL(window.location.href);
    if (id === ALL) url.searchParams.delete("category");
    else url.searchParams.set("category", id);
    url.hash = "";
    window.history.replaceState(null, "", url);
  };
  const open = useCallback((id: string | null) => {
    setOpenId(id);
    const url = new URL(window.location.href);
    url.hash = id ?? "";
    window.history.replaceState(
      null,
      "",
      id ? url : url.href.replace(/#$/, ""),
    );
  }, []);

  // Keep the selected category chip in view when the row scrolls on small screens.
  useEffect(() => {
    const row = tabs.current;
    const tab = row?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (row && tab) {
      const offset =
        tab.getBoundingClientRect().left - row.getBoundingClientRect().left;
      row.scrollTo({
        left: row.scrollLeft + offset - (row.clientWidth - tab.clientWidth) / 2,
        behavior: "smooth",
      });
    }
  }, [category]);

  // "/" focuses the search box unless the visitor is typing elsewhere.
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (
        event.key === "/" &&
        !openId &&
        !/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) &&
        !target.isContentEditable
      ) {
        event.preventDefault();
        search.current?.focus();
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [openId]);

  const sections =
    category === ALL && !terms.length
      ? categories
          .map((c) => ({
            category: c,
            items: visible.filter((item) => item.category === c.id),
          }))
          .filter((section) => section.items.length)
      : [{ category: active, items: visible }];
  const openIndex = visible.findIndex((item) => item.id === openId);
  const current = items.find((item) => item.id === openId) ?? null;
  const step = (offset: number) => {
    if (openIndex < 0) return;
    open(visible[(openIndex + offset + visible.length) % visible.length].id);
  };

  return (
    <>
      <div className="sticky top-[76px] z-20 -mx-5 border-b border-line bg-paper/95 px-5 pb-3 pt-4 backdrop-blur-sm md:-mx-10 md:px-10">
        <div className="flex flex-wrap items-center gap-3">
          <label className="relative flex min-w-0 flex-1 items-center sm:max-w-sm">
            <span className="sr-only">Search examples</span>
            <MagnifyingGlassIcon className="pointer-events-none absolute left-3 h-4 w-4 text-muted" />
            <input
              ref={search}
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search by feature, API name, or PDB ID"
              className="w-full rounded-lg border border-line bg-white py-2.5 pl-9 pr-10 text-sm outline-none transition focus:border-accent"
            />
            {!query && (
              <kbd className="pointer-events-none absolute right-3 hidden font-mono text-xs text-muted/70 sm:inline">
                /
              </kbd>
            )}
          </label>
          <p className="text-xs text-muted" role="status" aria-live="polite">
            {visible.length} {visible.length === 1 ? "example" : "examples"}
            {terms.length ? ` matching “${query.trim()}”` : ""}
          </p>
        </div>
        <div
          ref={tabs}
          role="tablist"
          aria-label="Gallery categories"
          className="-mx-1 mt-3 flex gap-1 overflow-x-auto px-1 pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        >
          {[{ id: ALL, title: "All" }, ...categories].map((tab) => {
            const selected = tab.id === category;
            const count = counts[tab.id] ?? 0;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={selected}
                aria-controls="gallery-results"
                onClick={() => chooseCategory(tab.id)}
                className={`flex shrink-0 items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-[13px] transition active:scale-[.98] ${
                  selected
                    ? "border-accent bg-accent text-white"
                    : count
                      ? "border-line bg-white text-ink hover:border-ink/30"
                      : "border-line bg-white text-muted/60 hover:border-ink/20"
                }`}
              >
                {tab.title}
                <span
                  className={`font-mono text-[10px] ${selected ? "text-white/75" : "text-muted"}`}
                >
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div id="gallery-results" role="tabpanel" className="mt-8">
        {active && (
          <p className="mb-8 max-w-2xl text-sm leading-7 text-muted">
            {active.description}
          </p>
        )}
        {!visible.length && (
          <div className="rounded-xl border border-dashed border-line px-6 py-16 text-center">
            <p className="text-sm text-muted">
              No examples match “{query.trim()}”
              {category !== ALL ? ` in ${active?.title}` : ""}.
            </p>
            <div className="mt-4 flex justify-center gap-3">
              {category !== ALL && counts[ALL] > 0 && (
                <button
                  className="button-secondary py-2"
                  onClick={() => chooseCategory(ALL)}
                >
                  Search all categories
                </button>
              )}
              <button
                className="button-secondary py-2"
                onClick={() => setQuery("")}
              >
                Clear search
              </button>
            </div>
          </div>
        )}
        {sections.map(({ category: section, items: list }) => (
          <section
            key={section?.id ?? "results"}
            aria-label={section?.title ?? "Results"}
            className="mb-14"
          >
            {category === ALL && !terms.length && section && (
              <div className="mb-5 flex items-baseline justify-between gap-4 border-b border-line pb-3">
                <h2 className="text-xl font-medium tracking-tight">
                  {section.title}
                </h2>
                <button
                  onClick={() => chooseCategory(section.id)}
                  className="inline-flex shrink-0 items-center gap-1.5 text-xs font-medium text-accent"
                >
                  {section.count} examples <ArrowRightIcon />
                </button>
              </div>
            )}
            <ul className="grid gap-x-6 gap-y-9 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
              {list.map((item) => (
                <li key={item.id} id={item.id} className="min-w-0 scroll-mt-48">
                  <Card
                    item={item}
                    label={
                      category === ALL && terms.length
                        ? titles[item.category]
                        : undefined
                    }
                    onOpen={() => open(item.id)}
                  />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>

      {current && (
        <Detail
          item={current}
          category={titles[current.category]}
          position={
            openIndex >= 0 ? `${openIndex + 1} of ${visible.length}` : ""
          }
          onClose={() => open(null)}
          onStep={openIndex >= 0 && visible.length > 1 ? step : undefined}
        />
      )}
    </>
  );
}

function Card({
  item,
  label,
  onOpen,
}: {
  item: GalleryItem;
  label?: string;
  onOpen: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [preview, setPreview] = useState(false);
  const start = () => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (!window.matchMedia("(hover: hover)").matches) return;
    setPreview(true);
    requestAnimationFrame(() => video.current?.play().catch(() => {}));
  };
  const stop = () => {
    video.current?.pause();
    setPreview(false);
  };
  return (
    <button
      type="button"
      onClick={onOpen}
      onMouseEnter={start}
      onMouseLeave={stop}
      onFocus={start}
      onBlur={stop}
      className="group block w-full text-left"
      aria-label={`${item.title}: open video and code`}
    >
      <div className="relative aspect-video overflow-hidden rounded-xl bg-[#0b1220] ring-1 ring-black/5 transition group-hover:ring-accent/40">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={item.poster}
          alt=""
          loading="lazy"
          decoding="async"
          className="absolute inset-0 h-full w-full object-cover transition duration-500 group-hover:scale-[1.02]"
        />
        {preview && (
          <video
            ref={video}
            src={item.video}
            muted
            loop
            playsInline
            preload="auto"
            className="absolute inset-0 h-full w-full object-cover"
          />
        )}
        <span className="absolute bottom-2.5 right-2.5 rounded bg-black/55 px-1.5 py-0.5 font-mono text-[10px] text-white/90 backdrop-blur-sm">
          {seconds(item.duration)}
        </span>
      </div>
      <div className="mt-3.5 flex items-start justify-between gap-3">
        <h3 className="text-[15px] font-medium leading-snug tracking-tight group-hover:text-accent">
          {item.title}
        </h3>
        {item.pdb && (
          <span className="mt-0.5 shrink-0 font-mono text-[10px] text-muted">
            {item.pdb}
          </span>
        )}
      </div>
      {label && (
        <p className="mt-1 text-[11px] font-medium uppercase tracking-wide text-accent/80">
          {label}
        </p>
      )}
      <p className="mt-1.5 line-clamp-2 text-[13px] leading-6 text-muted">
        {item.summary}
      </p>
      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {item.tags.slice(0, 4).map((tag) => (
          <span
            key={tag}
            className="rounded border border-line bg-white px-1.5 py-0.5 font-mono text-[10px] text-muted"
          >
            {tag}
          </span>
        ))}
      </div>
    </button>
  );
}

function Detail({
  item,
  category,
  position,
  onClose,
  onStep,
}: {
  item: GalleryItem;
  category: string;
  position: string;
  onClose: () => void;
  onStep?: (offset: number) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (element && !element.open) {
      element.showModal();
      element.focus(); // Arrow keys step through examples; no control starts focused.
    }
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, []);
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === "VIDEO" || !onStep) return;
      if (event.key === "ArrowRight") onStep(1);
      if (event.key === "ArrowLeft") onStep(-1);
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [onStep]);
  return (
    <dialog
      ref={dialog}
      aria-labelledby="gallery-detail-title"
      onClose={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) dialog.current?.close();
      }}
      tabIndex={-1}
      className="gallery-dialog outline-none"
    >
      <div className="flex items-center justify-between gap-3 border-b border-line px-5 py-3">
        <p className="min-w-0 truncate text-xs text-muted">
          <span className="font-medium text-accent">{category}</span>
          {position && <span> · {position}</span>}
        </p>
        <div className="flex items-center gap-1">
          {onStep && (
            <>
              <button
                onClick={() => onStep(-1)}
                className="rounded p-2 hover:bg-paper"
                aria-label="Previous example"
              >
                <ArrowLeftIcon />
              </button>
              <button
                onClick={() => onStep(1)}
                className="rounded p-2 hover:bg-paper"
                aria-label="Next example"
              >
                <ArrowRightIcon />
              </button>
            </>
          )}
          <button
            onClick={() => dialog.current?.close()}
            className="rounded p-2 hover:bg-paper"
            aria-label="Close"
          >
            <Cross2Icon />
          </button>
        </div>
      </div>
      <div className="grid min-h-0 flex-1 overflow-y-auto lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:overflow-hidden">
        <div className="min-w-0 p-5 lg:overflow-y-auto">
          <video
            key={item.id}
            src={item.video}
            poster={item.poster}
            controls
            autoPlay
            muted
            loop
            playsInline
            aria-label={item.title}
            className="aspect-video w-full rounded-lg bg-[#0b1220]"
          />
          <h2
            id="gallery-detail-title"
            className="mt-5 text-2xl font-medium tracking-[-.025em]"
          >
            {item.title}
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-7 text-muted">
            {item.summary}
          </p>
          <div className="mt-4 flex flex-wrap gap-1.5">
            {item.tags.map((tag) => (
              <span
                key={tag}
                className="rounded border border-line bg-paper px-1.5 py-0.5 font-mono text-[10px] text-muted"
              >
                {tag}
              </span>
            ))}
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
            <a
              href={item.sourceUrl}
              className="inline-flex items-center gap-1.5 font-medium text-accent"
            >
              View source <ExternalLinkIcon />
            </a>
            <a
              href={item.video}
              download
              className="inline-flex items-center gap-1.5 text-muted hover:text-ink"
            >
              <DownloadIcon /> Download MP4
            </a>
            <span className="font-mono text-[10px] text-muted">
              {item.pdb && `${item.pdb} · `}
              {seconds(item.duration)} · 60 fps
            </span>
          </div>
        </div>
        <div className="min-w-0 border-t border-line bg-paper/60 p-5 lg:overflow-y-auto lg:border-l lg:border-t-0">
          {item.codeHTML && (
            <>
              <p className="example-label">
                Scene <span>{item.sourceLabel}</span>
              </p>
              <div
                className="gallery-code"
                dangerouslySetInnerHTML={{ __html: item.codeHTML }}
              />
            </>
          )}
          <p className={`example-label ${item.codeHTML ? "mt-5" : ""}`}>
            Render <span>From a source checkout</span>
          </p>
          <div dangerouslySetInnerHTML={{ __html: item.commandHTML }} />
        </div>
      </div>
    </dialog>
  );
}
