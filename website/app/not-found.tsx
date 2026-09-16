import Link from "next/link";
export default function NotFound() {
  return (
    <main id="main" className="mx-auto max-w-3xl px-5 py-24">
      <p className="eyebrow">404</p>
      <h1 className="mt-4 text-4xl font-medium tracking-tight">
        This page is not here.
      </h1>
      <p className="mt-5 text-muted">
        Browse the documentation or use search to find the API or example you
        need.
      </p>
      <Link className="button-primary mt-8" href="/docs/getting-started/">
        Open the documentation
      </Link>
    </main>
  );
}
