"use client";
import { useState } from "react";
import Video from "./Video";
const views = [
  {
    name: "Surfaces & color",
    file: "surfaces",
    caption: "Ubiquitin · 2K39 · residue styling + rebuilt SES",
  },
  {
    name: "Interactions",
    file: "interactions",
    caption: "Ubiquitin · live rulers + screened Coulomb contacts",
  },
  {
    name: "Labels & callouts",
    file: "labels",
    caption: "Ubiquitin · 2K39 · vector writing + live labels",
  },
  {
    name: "Ball & stick",
    file: "nmr-atoms",
    caption: "Ubiquitin · 602 heavy atoms · 116 conformers",
  },
];
export default function HeroPlayer() {
  const [selected, setSelected] = useState(0);
  const view = views[selected];
  return (
    <div>
      <div
        className="mb-3 flex flex-wrap items-center gap-1"
        aria-label="Choose a rendered example"
      >
        {views.map((v, i) => (
          <button
            key={v.file}
            onClick={() => setSelected(i)}
            aria-pressed={i === selected}
            className={`rounded-md px-3 py-2 text-xs font-medium transition active:scale-[.98] ${i === selected ? "bg-ink text-white" : "text-muted hover:bg-ink/5"}`}
          >
            {v.name}
          </button>
        ))}
      </div>
      <Video key={view.file} file={view.file} title={view.caption} autoplay />
      <div className="mt-3 flex flex-wrap justify-between gap-2 font-mono text-[10px] leading-relaxed tracking-wide text-muted">
        <span>{view.caption}</span>
        <span>RENDERED WITH PROTEINMOTION</span>
      </div>
    </div>
  );
}
