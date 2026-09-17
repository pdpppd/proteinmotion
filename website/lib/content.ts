import fs from "node:fs";
import path from "node:path";
import { Marked } from "marked";
import hljs from "highlight.js/lib/common";
import { asset, guides, repo } from "./config";
import { docExample } from "./doc-examples";

export const readGuide = (file: string) =>
  fs
    .readFileSync(path.join(process.cwd(), "..", "docs", file), "utf8")
    .replace(
      "{{NMR_SOURCE}}",
      "```python output=gallery-regions\n" +
        fs.readFileSync(
          path.join(process.cwd(), "..", "examples/nmr_regions.py"),
          "utf8",
        ) +
        "```",
    )
    .replace(
      "{{LABEL_SOURCE}}",
      "```python output=gallery-labels\n" +
        fs.readFileSync(
          path.join(process.cwd(), "..", "examples/labels_and_callouts.py"),
          "utf8",
        ) +
        "```",
    )
    .replace(
      "{{MOLECULAR_SOURCE}}",
      "```python output=gallery-surfaces\n" +
        fs.readFileSync(
          path.join(process.cwd(), "..", "examples/molecular_tools.py"),
          "utf8",
        ) +
        "```",
    );
export const searchData = () =>
  guides.map((g) => ({
    title: g.title,
    href: `/docs/${g.slug}/`,
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
function exampleHTML(id: string, text: string, language: string) {
  const example = docExample(id, text);
  const revision = example.video_sha256?.slice(0, 12);
  const media = (file: string) =>
    asset(file) + (revision ? `?v=${revision}` : "");
  const source = `${repo}/blob/main/${example.source}${example.line ? `#L${example.line}` : ""}`;
  return `<div class="doc-example" data-example="${escape(id)}">
    <div class="doc-example-code">
      <div class="example-label">Code <span>${language === "bash" ? "Render command" : example.code ? "Scene excerpt" : "Full script"}</span></div>
      ${codeHTML(text, language)}
    </div>
    <figure class="doc-example-output">
      <div class="example-label">Output <span>Preview · ${example.duration.toFixed(1)} s · ${example.fps} fps</span></div>
      <video controls muted playsinline preload="none" poster="${escape(media(example.poster))}" aria-label="${escape(example.title)}">
        <source src="${escape(media(example.video))}" type="video/mp4" />
        <a href="${escape(media(example.video))}">Download the video</a>
      </video>
      <figcaption>
        <strong>${escape(example.title)}</strong>
        <p>${escape(example.caption)}</p>
        <div class="example-links">
          <a href="${escape(source)}">Full source: ${escape(example.scene)}</a>
          <a href="${escape(media(example.video))}" download>Download MP4</a>
        </div>
      </figcaption>
    </figure>
  </div>`;
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
        const [language = "text", ...options] = lang?.split(/\s+/) ?? [];
        const output = options.find((option) => option.startsWith("output="));
        return output
          ? exampleHTML(output.slice(7), text, language)
          : codeHTML(text, language);
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
