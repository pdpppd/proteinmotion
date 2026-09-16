import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ExternalLinkIcon,
} from "@radix-ui/react-icons";
import { guides } from "@/lib/config";
import { compile, readGuide, editLink } from "@/lib/content";
export const dynamicParams = false;
export function generateStaticParams() {
  return guides.map((g) => ({ slug: g.slug }));
}
export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const guide = guides.find((g) => g.slug === slug);
  return {
    title: guide?.title,
    description: guide?.description,
    alternates: {
      canonical: `https://pdpppd.github.io/proteinmotion/docs/${slug}/`,
    },
  };
}
export default async function Guide({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const index = guides.findIndex((g) => g.slug === slug);
  if (index < 0) notFound();
  const guide = guides[index];
  const { html, toc } = compile(readGuide(guide.file));
  const menu = (
    <nav aria-label="Documentation">
      {guides.map((g, i) => (
        <div key={g.slug}>
          {(i === 0 || g.group !== guides[i - 1].group) && (
            <p className="mb-2 mt-7 px-3 font-mono text-[10px] tracking-widest text-muted first:mt-0">
              {g.group}
            </p>
          )}
          <Link
            href={`/docs/${g.slug}/`}
            aria-current={slug === g.slug ? "page" : undefined}
            className={`my-0.5 block rounded-md px-3 py-2 text-[13px] transition ${slug === g.slug ? "bg-accent-soft font-medium text-accent" : "text-muted hover:bg-ink/5 hover:text-ink"}`}
          >
            {g.title}
          </Link>
        </div>
      ))}
    </nav>
  );
  return (
    <main
      id="main"
      className="mx-auto grid max-w-[1400px] gap-10 px-5 py-8 md:px-10 lg:grid-cols-[200px_minmax(0,1fr)] lg:gap-12 lg:py-12 xl:grid-cols-[200px_minmax(0,1fr)_180px]"
    >
      <aside className="hidden lg:block">
        <div className="sticky top-28 max-h-[calc(100dvh-140px)] overflow-y-auto pb-8">
          {menu}
        </div>
      </aside>
      <article className="min-w-0">
        <details
          key={slug}
          className="mb-7 rounded-lg border border-line p-4 lg:hidden"
        >
          <summary className="cursor-pointer text-sm font-medium">
            Documentation · {guide.title}
          </summary>
          <div className="pt-4">{menu}</div>
        </details>
        <p className="eyebrow">DOCUMENTATION / {guide.group}</p>
        <h1 className="mt-4 text-4xl font-medium leading-tight tracking-[-.045em]">
          {guide.title}
        </h1>
        <p className="mt-4 border-b border-line pb-8 text-base leading-relaxed text-muted">
          {guide.description}
        </p>
        <div
          className="prose-doc mt-8"
          dangerouslySetInnerHTML={{ __html: html }}
        />
        <a
          href={editLink(guide.file)}
          className="mt-12 inline-flex items-center gap-2 text-xs text-muted hover:text-accent"
        >
          View this page on GitHub <ExternalLinkIcon />
        </a>
        <div className="mt-8 grid grid-cols-2 gap-4 border-t border-line pt-6">
          {index > 0 ? (
            <Link href={`/docs/${guides[index - 1].slug}/`} className="text-sm">
              <span className="mb-2 flex items-center gap-2 text-xs text-muted">
                <ArrowLeftIcon /> Previous
              </span>
              {guides[index - 1].title}
            </Link>
          ) : (
            <span />
          )}
          {index < guides.length - 1 && (
            <Link
              href={`/docs/${guides[index + 1].slug}/`}
              className="text-right text-sm"
            >
              <span className="mb-2 flex items-center justify-end gap-2 text-xs text-muted">
                Next <ArrowRightIcon />
              </span>
              {guides[index + 1].title}
            </Link>
          )}
        </div>
      </article>
      <aside className="hidden xl:block">
        <nav
          aria-label="On this page"
          className="sticky top-28 max-h-[calc(100dvh-140px)] overflow-y-auto"
        >
          <p className="mb-5 font-mono text-[10px] tracking-widest text-muted">
            ON THIS PAGE
          </p>
          {toc
            .filter((t) => t.depth <= 3)
            .map((t) => (
              <a
                key={t.id}
                href={`#${t.id}`}
                className={`mb-3 block text-xs leading-relaxed text-muted hover:text-accent ${t.depth === 3 ? "pl-3" : ""}`}
              >
                {t.title}
              </a>
            ))}
        </nav>
      </aside>
    </main>
  );
}
