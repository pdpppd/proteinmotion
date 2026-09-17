import fs from "node:fs";
import path from "node:path";
import { demos } from "./config";

export type DocExample = {
  title: string;
  caption: string;
  source: string;
  scene: string;
  duration: number;
  fps: number;
  video: string;
  poster: string;
  code?: string;
  line?: number;
  video_sha256?: string;
};
const rendered: Record<string, DocExample> = JSON.parse(
  fs.readFileSync(
    path.join(process.cwd(), "public/media/docs/manifest.json"),
    "utf8",
  ),
);

export function docExample(id: string, code: string): DocExample {
  if (id.startsWith("gallery-")) {
    const demo = demos.find((item) => item.id === id.slice(8));
    if (!demo) throw new Error(`Unknown gallery example: ${id}`);
    if (!code.includes(demo.scene))
      throw new Error(`Example ${id} must include its scene ${demo.scene}`);
    return {
      title: demo.title,
      caption: demo.detail,
      source: demo.source,
      scene: demo.scene,
      duration: Number.parseFloat(demo.duration),
      fps: demo.id === "showcase" ? 60 : 30,
      video: `media/${demo.file}.mp4`,
      poster: `media/${demo.file}.jpg`,
    };
  }
  const example = rendered[id];
  if (!example) throw new Error(`Unknown rendered example: ${id}`);
  if (example.code && example.code.trim() !== code.trim())
    throw new Error(
      `The ${id} snippet differs from its rendered source. Run scripts/render_docs_examples.py.`,
    );
  return example;
}
