import type { Metadata } from "next";
import Link from "next/link";
import "@fontsource/geist/400.css";
import "@fontsource/geist/500.css";
import "@fontsource/geist/600.css";
import "@fontsource/geist/700.css";
import "@fontsource/geist-mono/400.css";
import "./globals.css";
import Header from "@/components/Header";
import CodeActions from "@/components/CodeActions";
import { asset, repo, site } from "@/lib/config";
export const metadata: Metadata = {
  metadataBase: new URL(site + "/"),
  title: {
    default: "ProteinMotion — Molecular animation in Python",
    template: "%s · ProteinMotion",
  },
  description:
    "Animate proteins, DNA, and RNA in Python. Render cartoons, nucleotide bases, atoms, surfaces, labels, NMR ensembles, and MD trajectories.",
  icons: { icon: asset("icon.svg") },
  openGraph: {
    title: "ProteinMotion",
    description: "Protein, DNA, and RNA animation in Python.",
    type: "website",
    images: [
      {
        url: `${site}/media/calmodulin-in-focus.jpg`,
        width: 1920,
        height: 1080,
      },
    ],
  },
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <Header />
        {children}
        <footer className="mt-16 border-t border-line">
          <div className="mx-auto flex max-w-[1400px] flex-wrap justify-between gap-5 px-5 py-8 text-xs leading-relaxed text-muted md:px-10">
            <span>ProteinMotion · Open source under the MIT license.</span>
            <div className="flex gap-5">
              <Link href="/docs/rendering/">Rendering</Link>
              <a href={`${repo}/blob/main/LICENSE`}>License</a>
              <a href={repo}>GitHub</a>
            </div>
          </div>
        </footer>
        <CodeActions />
      </body>
    </html>
  );
}
