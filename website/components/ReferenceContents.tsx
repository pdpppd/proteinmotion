"use client";
import { useRef } from "react";

export default function ReferenceContents({
  items,
}: {
  items: { id: string; title: string }[];
}) {
  const panel = useRef<HTMLDetailsElement>(null);
  return (
    <details
      ref={panel}
      className="sticky top-[84px] z-20 mb-7 rounded-lg border border-line bg-paper px-4 py-3 shadow-sm"
    >
      <summary className="text-sm text-muted">On this page</summary>
      <nav
        aria-label="On this page"
        className="mt-4 grid max-h-[50dvh] gap-3 overflow-y-auto text-xs sm:grid-cols-3"
      >
        {items.map((item) => (
          <a
            key={item.id}
            className="break-words text-accent"
            href={`#${item.id}`}
            onClick={() => {
              if (panel.current) panel.current.open = false;
            }}
          >
            {item.title}
          </a>
        ))}
      </nav>
    </details>
  );
}
