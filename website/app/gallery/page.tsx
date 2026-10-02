import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRightIcon } from "@radix-ui/react-icons";
import GalleryBrowser from "@/components/GalleryBrowser";
import { FILMS, galleryData } from "@/lib/gallery";

export const metadata: Metadata = {
  title: "Example gallery",
  description:
    "Short Studio-rendered examples of every ProteinMotion feature, each with its complete Python scene.",
};

export default function Gallery() {
  const { items, categories } = galleryData();
  const focused = items.filter((item) => item.category !== FILMS.id).length;
  return (
    <main
      id="main"
      className="mx-auto max-w-[1400px] px-5 pb-12 pt-12 md:px-10 md:pt-16"
    >
      <div className="max-w-3xl">
        <h1 className="text-4xl font-medium tracking-[-.045em] md:text-5xl">
          Example gallery
        </h1>
        <p className="mt-5 text-base leading-7 text-muted">
          {focused} short examples, one feature each, rendered with the Studio
          renderer at 720p and 60 fps. Open any example for its video and the
          complete scene; every script runs as shown from a source checkout.
        </p>
        <p className="mt-3 text-sm leading-7 text-muted">
          The examples need ProteinMotion 0.14.0 or later. Morphs, NMR
          interpolation, and threading illustrate coordinate changes, not
          physical pathways.{" "}
          <Link
            href="/docs/getting-started/"
            className="inline-flex items-center gap-1 text-accent"
          >
            Get started <ArrowRightIcon />
          </Link>
        </p>
      </div>
      <div className="mt-8">
        <GalleryBrowser items={items} categories={categories} />
      </div>
    </main>
  );
}
