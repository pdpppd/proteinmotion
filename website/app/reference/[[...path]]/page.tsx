import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { reference, referencePaths } from "@/lib/reference";
import {
  ReferenceIndex,
  ReferenceNav,
  ModulePage,
  SymbolPage,
} from "@/components/ReferenceManual";
import ReferenceContents from "@/components/ReferenceContents";
import { site } from "@/lib/config";
export const dynamicParams = false;
export function generateStaticParams() {
  return referencePaths.map((path) => ({ path }));
}
type Props = { params: Promise<{ path?: string[] }> };
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { path = [] } = await params;
  const item =
    path.length === 2
      ? reference.symbols.find(
          (s) => s.module === path[0] && s.name === path[1],
        )
      : reference.modules.find((m) => m.id === path[0]);
  return {
    title:
      path.length === 2
        ? `${path[1]} · Reference`
        : path.length === 1
          ? `${reference.modules.find((m) => m.id === path[0])?.title} · Reference`
          : "Reference manual",
    description:
      item?.description ??
      "Python API reference for ProteinMotion: classes, functions, parameters, and methods.",
    alternates: {
      canonical: `${site}/reference/${path.length ? path.join("/") + "/" : ""}`,
    },
  };
}
export default async function ReferencePage({ params }: Props) {
  const { path = [] } = await params;
  const [module, name] = path;
  const m = reference.modules.find((m) => m.id === module);
  const symbol = reference.symbols.find(
    (s) => s.module === module && s.name === name,
  );
  if (path.length > 2 || (module && !m) || (name && !symbol)) notFound();
  const toc =
    symbol && !symbol.alias
      ? [
          {
            id: "signature",
            title: symbol.kind === "class" ? "Constructor" : "Signature",
          },
          ...(symbol.notes ? [{ id: "notes", title: "Notes" }] : []),
          ...(Object.keys(symbol.attributes).length
            ? [{ id: "attributes", title: "Attributes" }]
            : []),
          ...(symbol.example
            ? [{ id: "example", title: "Example and output" }]
            : []),
          ...(symbol.members.length
            ? [{ id: "methods", title: "Methods and properties" }]
            : []),
          ...symbol.members.map((m) => ({ id: m.name, title: m.name })),
        ]
      : [];
  return (
    <main
      id="main"
      className="mx-auto grid max-w-[1600px] gap-10 px-5 py-8 md:px-10 lg:grid-cols-[190px_minmax(0,1fr)] lg:py-12"
    >
      <aside className="hidden lg:block">
        <div className="sticky top-28 max-h-[calc(100dvh-140px)] overflow-y-auto pb-8">
          <ReferenceNav module={module} symbol={name} />
        </div>
      </aside>
      <article className="min-w-0">
        <details
          key={path.join("/")}
          className="mb-7 rounded-lg border border-line p-4 lg:hidden"
        >
          <summary className="text-sm font-medium">
            Reference navigation
          </summary>
          <div className="pt-5">
            <ReferenceNav module={module} symbol={name} />
          </div>
        </details>
        <nav
          aria-label="Breadcrumb"
          className="mb-7 flex flex-wrap gap-2 text-xs text-muted"
        >
          <Link href="/reference/">Reference</Link>
          {m && (
            <>
              <span>/</span>
              <Link href={`/reference/${module}/`}>{m.title}</Link>
            </>
          )}
          {name && (
            <>
              <span>/</span>
              <span className="font-mono">{name}</span>
            </>
          )}
        </nav>
        {toc.length > 0 && (
          <ReferenceContents key={path.join("/")} items={toc} />
        )}
        {symbol ? (
          <SymbolPage symbol={symbol} />
        ) : m ? (
          <ModulePage id={module} />
        ) : (
          <ReferenceIndex />
        )}
      </article>
    </main>
  );
}
