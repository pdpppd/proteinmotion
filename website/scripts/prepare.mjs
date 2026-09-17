import { createHash } from "node:crypto";
import {
  mkdir,
  readdir,
  copyFile,
  writeFile,
  readFile,
} from "node:fs/promises";
import { resolve } from "node:path";
const target = resolve("public/downloads");
await mkdir(target, { recursive: true });
for (const filename of await readdir("../docs")) {
  if (/\.(json|csv|pdf|png)$/.test(filename))
    await copyFile(`../docs/${filename}`, `${target}/${filename}`);
}
for (const filename of [
  "quickstart.py",
  "docs_examples.py",
  "nmr_regions.py",
  "labels_and_callouts.py",
  "molecular_tools.py",
  "alpha_helix_hbonds.py",
  "feature_showcase.py",
])
  await copyFile(`../examples/${filename}`, `${target}/${filename}`);
await writeFile("public/.nojekyll", "");

// A preview must correspond to the checked-in scene and data.
const rendered = JSON.parse(
  await readFile("public/media/docs/manifest.json", "utf8"),
);
const hashes = new Map();
async function hash(file) {
  if (!hashes.has(file))
    hashes.set(
      file,
      createHash("sha256")
        .update(await readFile(file))
        .digest("hex"),
    );
  return hashes.get(file);
}
for (const [id, example] of Object.entries(rendered)) {
  for (const [file, expected] of Object.entries(example.dependencies)) {
    if ((await hash(`../${file}`)) !== expected)
      throw new Error(
        `Preview ${id} is stale: ${file} changed. Run scripts/render_docs_examples.py.`,
      );
  }
  for (const [file, expected] of [
    [example.video, example.video_sha256],
    [example.poster, example.poster_sha256],
  ]) {
    if ((await hash(`public/${file}`)) !== expected)
      throw new Error(`Preview asset changed or is incomplete: ${file}`);
  }
}
console.log(
  `Checked sources and assets for ${Object.keys(rendered).length} rendered documentation examples.`,
);
