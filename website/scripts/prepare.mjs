import { mkdir, readdir, copyFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
const target = resolve("public/downloads");
await mkdir(target, { recursive: true });
for (const filename of await readdir("../docs")) {
  if (/\.(json|csv|pdf|png)$/.test(filename))
    await copyFile(`../docs/${filename}`, `${target}/${filename}`);
}
for (const filename of [
  "quickstart.py",
  "nmr_regions.py",
  "labels_and_callouts.py",
  "molecular_tools.py",
  "alpha_helix_hbonds.py",
])
  await copyFile(`../examples/${filename}`, `${target}/${filename}`);
await writeFile("public/.nojekyll", "");
