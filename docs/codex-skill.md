# Make movies with Codex

The **ProteinMotion Movies** skill teaches Codex to author and render molecular films with the package. It covers scene timing, representations, residue colors and opacity, camera focus, vector labels, distance rulers, trajectories, deformation, protein morphs and measured interactions. It includes a runnable starter scene and deposited 1UBQ coordinates.

[Read the skill on GitHub](https://github.com/pdpppd/proteinmotion/tree/main/skills/proteinmotion-movies) · [Download the skill ZIP](https://github.com/pdpppd/proteinmotion/releases/download/v0.7.0/proteinmotion-movies-v0.7.0.zip)

## Install

After [installing ProteinMotion](getting-started.md):

```bash
proteinmotion install-skill
```

The command copies the bundled skill into `${CODEX_HOME:-~/.codex}/skills/proteinmotion-movies`. It is an explicit command; installing the Python package does not change Codex configuration by itself. Automatic skill selection is enabled. Start a new Codex conversation if the skill is not listed yet.

An identical installation is left unchanged. Modified bundled files are preserved unless you explicitly request replacement:

```bash
proteinmotion install-skill --force
```

Unrelated files in that skill folder are retained. To choose another destination, pass the complete skill folder path:

```bash
proteinmotion install-skill --path /path/to/skills/proteinmotion-movies
```

You can also extract the release ZIP into your skills directory. The ZIP and wheel contain the same skill source committed under `skills/proteinmotion-movies` in the repository.

## Example requests

> Use $proteinmotion-movies to make a 15-second film from my PDB file. Show a cartoon, highlight chain A residues 35–48, write a callout, then switch to ball-and-stick.

> Use $proteinmotion-movies to animate my XTC trajectory, keeping the camera centered on residues 50–90 and updating a distance label between two specified atoms.

> Use $proteinmotion-movies to morph calmodulin into troponin C using contact-map matching, fade unmatched residues, and continue the movie using the morphed target.

Give Codex your input paths and the scientific point you want to show. Specify chains/residues, length, aspect ratio or frame rate when they matter. The skill returns an editable scene and a rendered movie, and asks for missing information only when it is needed.

The skill distinguishes visual morphs and NMR interpolation from MD, explains inferred hydrogens and approximate electrostatics, and checks the actual rendered framing and labels. It uses the package's own API and renderer. It does not imply authorization to publish a movie or modify unrelated projects.

## Resources

- `SKILL.md`: task routing, environment setup, authoring and render checks.
- `references/animation.md`: timeline invariants, styling, states and morphs.
- `references/annotations-and-interactions.md`: region annotations and molecular analyses.
- `assets/film.py` and `assets/1ubq.cif`: the self-contained starter used by `proteinmotion init`.
- `agents/openai.yaml`: Codex display metadata and the example invocation.
