import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdir,
  readdir,
  copyFile,
  writeFile,
  readFile,
} from "node:fs/promises";
import { resolve } from "node:path";
execFileSync(
  process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3"),
  ["../scripts/build_reference.py", "--check"],
  { stdio: "inherit" },
);
const target = resolve("public/downloads");
await mkdir(target, { recursive: true });
for (const filename of await readdir("../docs")) {
  if (/\.(json|csv|pdf|png)$/.test(filename))
    await copyFile(`../docs/${filename}`, `${target}/${filename}`);
}
for (const filename of [
  "quickstart.py",
  "dna_styles.py",
  "dna_morph.py",
  "rna_styles.py",
  "eevee_focus.py",
  "calmodulin_in_focus.py",
  "numerical_properties.py",
  "ligands_and_side_chains.py",
  "depth_tunnels.py",
  "thread_hemoglobin.py",
  "binding_sites_film.py",
  "troponin_sites.py",
  "side_chain_ensemble.py",
  "nucleic_ions.py",
  "synchronized_plots.py",
  "density_maps.py",
  "docs_examples.py",
  "nmr_regions.py",
  "labels_and_callouts.py",
  "molecular_tools.py",
  "alpha_helix_hbonds.py",
  "feature_showcase.py",
])
  await copyFile(`../examples/${filename}`, `${target}/${filename}`);
await mkdir(`${target}/data`, { recursive: true });
for (const pdb of ["1cll", "1bna", "1ehz", "2dcg"])
  await copyFile(`../examples/data/${pdb}.cif`, `${target}/data/${pdb}.cif`);
await writeFile("public/.nojekyll", "");

// A preview must correspond to the checked-in scene and data.
const rendered = JSON.parse(
  await readFile("public/media/docs/manifest.json", "utf8"),
);
const hashes = new Map();
async function hash(file) {
  if (!hashes.has(file)) {
    let data = await readFile(file);
    // Git may check Python sources out with CRLF on Windows. Source identity
    // uses LF on every platform; binary structure and media hashes stay exact.
    if (file.endsWith(".py"))
      data = Buffer.from(data.toString("utf8").replace(/\r\n/g, "\n"));
    hashes.set(
      file,
      createHash("sha256").update(data).digest("hex"),
    );
  }
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
