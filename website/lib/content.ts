import fs from "node:fs";
import path from "node:path";
import { Marked } from "marked";
import hljs from "highlight.js/lib/common";
import { asset, guides, repo } from "./config";

export const readGuide = (file: string) =>
  fs
    .readFileSync(path.join(process.cwd(), "..", "docs", file), "utf8")
    .replace(
      "{{NMR_SOURCE}}",
      "```python\n" +
        fs.readFileSync(
          path.join(process.cwd(), "..", "examples/nmr_regions.py"),
          "utf8",
        ) +
        "```",
    )
    .replace(
      "{{LABEL_SOURCE}}",
      "```python\n" +
        fs.readFileSync(
          path.join(process.cwd(), "..", "examples/labels_and_callouts.py"),
          "utf8",
        ) +
        "```",
    );
export const searchData = () =>
  guides.map((g) => ({
    title: g.title,
    slug: g.slug,
    description: g.description,
    text: readGuide(g.file)
      .replace(/[`#*|]/g, "")
      .replace(/\s+/g, " "),
  }));
const escape = (s: string) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
export function codeHTML(text: string, language = "python") {
  const highlighted = hljs.getLanguage(language)
    ? hljs.highlight(text, { language }).value
    : escape(text);
  return `<div class="code-block"><div class="code-toolbar"><span>${escape(language)}</span><button type="button" class="copy-code" aria-label="Copy code">Copy</button></div><pre tabindex="0"><code class="hljs language-${escape(language)}">${highlighted}</code></pre></div>`;
}
export function compile(markdown: string) {
  const toc: { id: string; title: string; depth: number }[] = [];
  const used = new Map<string, number>();
  const parser = new Marked({ gfm: true });
  parser.use({
    renderer: {
      heading({ tokens, depth, text }) {
        const plain = text.replace(/[`*]/g, "");
        const slug = plain
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-|-$/g, "");
        const n = used.get(slug) ?? 0;
        used.set(slug, n + 1);
        const id = n ? `${slug}-${n}` : slug;
        toc.push({ id, title: plain, depth });
        return `<h${depth} id="${id}">${this.parser.parseInline(tokens)}<a class="heading-anchor" href="#${id}" aria-label="Link to ${escape(plain)}">#</a></h${depth}>`;
      },
      code({ text, lang }) {
        return codeHTML(text, lang?.split(" ")[0] || "text");
      },
      link({ href, tokens }) {
        if (!/^(https?:|mailto:|#)/.test(href)) {
          const filename = href.replace(/^docs\//, "");
          const guide = guides.find((g) => g.file === filename);
          href = guide
            ? asset(`docs/${guide.slug}/`)
            : filename === "README.md"
              ? asset("docs/getting-started/")
              : asset(`downloads/${filename}`);
        }
        return `<a href="${escape(href)}">${this.parser.parseInline(tokens)}</a>`;
      },
      image({ href, text }) {
        return `<img src="${escape(href.startsWith("http") ? href : asset("downloads/" + href))}" alt="${escape(text)}" loading="lazy" />`;
      },
    },
  });
  const html = parser.parse(markdown.replace(/^# .+\n/, "")) as string;
  return { html, toc };
}
export const editLink = (file: string) => `${repo}/blob/main/docs/${file}`;
