import fs from "node:fs";
import path from "node:path";
import { asset, demoCommand, demos, repo } from "./config";
import { codeHTML } from "./content";

type Category = { id: string; title: string; description: string };
type Rendered = {
  id: string;
  category: string;
  file: string;
  scene: string | string[];
  title: string;
  summary: string;
  tags: string[];
  pdb: string;
  source: string;
  duration: number;
  video: string;
  poster: string;
  video_sha256: string;
  code: string;
};

/** One card in the gallery, ready for the client browser. */
export type GalleryItem = {
  id: string;
  category: string;
  title: string;
  summary: string;
  tags: string[];
  pdb: string;
  duration: number;
  video: string;
  poster: string;
  sourceUrl: string;
  sourceLabel: string;
  codeHTML: string;
  commandHTML: string;
  search: string;
};
export type GalleryCategory = Category & { count: number };

const root = path.join(process.cwd(), "..");
const read = <T>(file: string): T =>
  JSON.parse(fs.readFileSync(path.join(root, file), "utf8"));

export const FILMS: Category = {
  id: "films",
  title: "Films",
  description:
    "Longer narrated examples that combine several features in one story.",
};

/** Names a scene imports from proteinmotion, so API names find their examples. */
const imports = (code: string) =>
  (code.match(/from proteinmotion import \(?([^)]*?)\)?\n(?!\s)/)?.[1] ?? "")
    .split(/[\s,]+/)
    .filter(Boolean);

const command = (item: Rendered) => {
  const scenes = Array.isArray(item.scene) ? item.scene : [item.scene];
  return scenes
    .map(
      (scene) =>
        `proteinmotion render ${item.source} ${scene} \\\n  --width 1280 --height 720 --fps 60 -o ${item.id}${scenes.length > 1 ? "-" + scene.toLowerCase() : ""}.mp4`,
    )
    .join("\n");
};

export function galleryData() {
  const catalog = read<{ categories: Category[] }>(
    "examples/gallery/catalog.json",
  );
  const rendered = Object.values(
    read<Record<string, Rendered>>(
      "website/public/media/gallery/manifest.json",
    ),
  );
  const media = (file: string, sha: string) =>
    `${asset(file)}?v=${sha.slice(0, 12)}`;
  const focused: GalleryItem[] = rendered.map((item) => ({
    id: item.id,
    category: item.category,
    title: item.title,
    summary: item.summary,
    tags: item.tags,
    pdb: item.pdb,
    duration: item.duration,
    video: media(item.video, item.video_sha256),
    poster: media(item.poster, item.video_sha256),
    sourceUrl: `${repo}/blob/main/${item.source}`,
    sourceLabel: item.source,
    codeHTML: codeHTML(item.code.trimEnd()),
    commandHTML: codeHTML(command(item), "bash"),
    search: [
      item.title,
      item.summary,
      item.pdb,
      ...item.tags,
      ...imports(item.code),
    ]
      .join(" ")
      .toLowerCase(),
  }));
  const films: GalleryItem[] = demos.map((demo) => ({
    id: `film-${demo.id}`,
    category: FILMS.id,
    title: demo.title,
    summary: demo.detail,
    tags: demo.label
      .split("·")
      .map((tag) => tag.trim().toLowerCase())
      .filter(Boolean),
    pdb: demo.pdb,
    duration: Number.parseFloat(demo.duration),
    video: asset(`media/${demo.file}.mp4`),
    poster: asset(`media/${demo.file}.jpg`),
    sourceUrl: `${repo}/blob/main/${demo.source}`,
    sourceLabel: demo.source,
    codeHTML: "",
    commandHTML: codeHTML(demoCommand(demo), "bash"),
    search: [demo.title, demo.detail, demo.label, demo.pdb, demo.scene]
      .join(" ")
      .toLowerCase(),
  }));
  const items = [...focused, ...films];
  const categories: GalleryCategory[] = [...catalog.categories, FILMS].map(
    (category) => ({
      ...category,
      count: items.filter((item) => item.category === category.id).length,
    }),
  );
  return { items, categories };
}

/** Header search entries that open a gallery example. */
export const gallerySearch = () =>
  galleryData()
    .items.filter((item) => item.category !== FILMS.id)
    .map((item) => ({
      title: `${item.title} (gallery)`,
      href: `/gallery/#${item.id}`,
      description: item.summary,
      text: item.tags.join(" "),
    }));
