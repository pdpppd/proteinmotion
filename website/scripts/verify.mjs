/** Validate a static export without a server or browser-only fallbacks. */
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
const root = path.resolve("out");
const base = process.env.PAGES_BASE_PATH ?? "/proteinmotion";
const origin = "https://pdpppd.github.io";
const files = [];
async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) await walk(filename);
    else if (entry.name.endsWith(".html")) files.push(filename);
  }
}
await walk(root);
const documents = new Map();
const failures = [];
let checked = 0;
for (const file of files) {
  const html = await readFile(file, "utf8");
  documents.set(file, {
    html,
    ids: new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1])),
  });
  if (!html.includes('id="main"'))
    failures.push(`Missing main landmark: ${file}`);
  if (!html.includes("<h1")) failures.push(`Missing page title: ${file}`);
  if (/\{\{(?:NMR|LABEL)_SOURCE\}\}/.test(html))
    failures.push("Example source placeholder was not expanded");
}
for (const [file, { html }] of documents) {
  const relative = path
    .relative(root, file)
    .replaceAll(path.sep, "/")
    .replace(/index\.html$/, "");
  const current = new URL(`${base}/${relative}`, origin);
  const linkHTML = html.replace(
    /<link\b[^>]*rel="(?:preconnect|dns-prefetch)"[^>]*>/g,
    "",
  );
  for (const match of linkHTML.matchAll(/\b(?:href|src|poster)="([^"]+)"/g)) {
    const href = match[1].replaceAll("&amp;", "&");
    if (!href || /^(mailto:|data:|tel:)/.test(href)) continue;
    const url = new URL(href, current);
    if (url.origin !== origin) continue;
    if (base && !url.pathname.startsWith(`${base}/`)) {
      failures.push(`Link escapes Pages base path in ${relative}: ${href}`);
      continue;
    }
    const target = path.resolve(
      root,
      "." + decodeURIComponent(url.pathname.slice(base.length)),
    );
    if (target !== root && !target.startsWith(root + path.sep)) {
      failures.push(`Path escapes export: ${href}`);
      continue;
    }
    let dest = target;
    try {
      if ((await stat(dest)).isDirectory())
        dest = path.join(dest, "index.html");
      await stat(dest);
      checked++;
      if (url.hash && documents.has(dest)) {
        const id = decodeURIComponent(url.hash.slice(1));
        if (!documents.get(dest).ids.has(id))
          failures.push(`Missing fragment from ${relative}: ${href}`);
      }
    } catch {
      failures.push(`Missing local target from ${relative}: ${href}`);
    }
  }
}
await stat(path.join(root, ".nojekyll"));
if (files.length < 13)
  failures.push(
    `Expected home, gallery, ten guides and 404; found ${files.length} HTML files`,
  );
if (failures.length) {
  console.error([...new Set(failures)].join("\n"));
  process.exit(1);
}
console.log(
  `Verified ${files.length} static HTML pages and ${checked} local links/media references, including fragment targets.`,
);
