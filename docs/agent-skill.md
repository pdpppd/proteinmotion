# Use with an AI agent

The ProteinMotion Movies skill provides instructions for creating protein videos. It covers scene timing, representations, residue colors, labels, camera movement, trajectories, morphs, and interaction measurements. It includes a starter script and a ubiquitin structure.

Use the skill with an AI agent that can read local files and run Python commands. Agents with skill support can load it from their skills directory. Other agents can read `SKILL.md` directly.

[Skill source](https://github.com/pdpppd/proteinmotion/tree/main/skills/proteinmotion-movies) · [Download ZIP](https://github.com/pdpppd/proteinmotion/releases/download/v0.9.1/proteinmotion-movies-v0.9.1.zip)

## Install the skill

First, [install ProteinMotion](getting-started.md). Then choose the full destination folder for your agent:

```bash
proteinmotion install-skill --path /path/to/skills/proteinmotion-movies
```

Replace the path with a location your agent reads. You can also extract the ZIP there. Keep `SKILL.md`, `references/`, and `assets/` together so the relative links work. Follow your agent's instructions for loading a new skill.

For an agent that reads instructions directly, give it the path to `SKILL.md` and ask it to use those instructions for the task.

### Codex setup

The default destination is `${CODEX_HOME:-~/.codex}/skills/proteinmotion-movies`:

```bash
proteinmotion install-skill
```

Start a new Codex conversation, then use `$proteinmotion-movies` in your request. The included `agents/openai.yaml` supplies Codex display metadata; the instructions and examples are in the shared Markdown files.

### Update an existing installation

Repeating the install command preserves identical files. If bundled files have changed locally, review them before replacing them with `--force`:

```bash
proteinmotion install-skill --path /path/to/skills/proteinmotion-movies --force
```

The installer keeps additional files you have added to the folder.

## Describe the video

Give the agent the input file paths and the feature you want to show. Include chain IDs, residue numbers, duration, and output size when relevant. Ask for both the video and its Python script.

Example requests:

> Use the ProteinMotion Movies skill to make a 15-second video from my PDB file. Show a cartoon, highlight chain A residues 35–48, add a label, then switch to ball-and-stick.

> Animate my XTC trajectory. Keep the camera centered on residues 50–90 and show the distance between the two atoms I selected.

> Morph calmodulin into troponin C using contact-map matching. Fade unmatched residues, then rotate troponin C to show the final structure.

The skill instructs the agent to inspect the input structure, write a scene, render it, and check the framing and labels. It also explains how to describe interpolated states, inferred hydrogens, and approximate electrostatic calculations.

## Skill files

| File | Purpose |
|---|---|
| `SKILL.md` | Setup, scene creation, rendering, and review instructions |
| `references/animation.md` | Timing, styling, trajectories, and morphs |
| `references/annotations-and-interactions.md` | Labels, region tools, and measurements |
| `assets/film.py` | Starter scene used by `proteinmotion init` |
| `assets/1ubq.cif` | Ubiquitin structure for the starter scene |
| `agents/openai.yaml` | Codex display metadata |
