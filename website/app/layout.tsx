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
    default: "ProteinMotion — Protein animation, written in Python",
    template: "%s · ProteinMotion",
  },
  description:
    "Programmatic protein films with a Manim-inspired Python API. Native Metal rendering, cartoons, ball-and-stick, contact-guided morphs, NMR and MD trajectories.",
  icons: { icon: asset("icon.svg") },
  openGraph: {
    title: "ProteinMotion",
    description: "Protein animation, written in Python.",
    type: "website",
    images: [{ url: `${site}/media/regions.jpg`, width: 1280, height: 720 }],
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
              <Link href="/docs/rendering/">Methods & limits</Link>
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
