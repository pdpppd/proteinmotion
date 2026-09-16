export const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "/proteinmotion";
export const repo = "https://github.com/pdpppd/proteinmotion";
export const site = "https://pdpppd.github.io/proteinmotion";
export const asset = (path: string) => `${base}/${path.replace(/^\//, "")}`;
export const guides = [
  {
    slug: "getting-started",
    file: "getting-started.md",
    title: "Get started",
    group: "START HERE",
    description: "Install ProteinMotion and render your first film.",
  },
  {
    slug: "scenes",
    file: "scenes.md",
    title: "Scenes & motion",
    group: "AUTHORING",
    description:
      "Compose scenes, representations, transforms, and eased animation.",
  },
  {
    slug: "regions",
    file: "regions.md",
    title: "Regions & focus",
    group: "AUTHORING",
    description:
      "Select residues, add 3D highlights, and follow moving regions.",
  },
  {
    slug: "morphing",
    file: "morphing.md",
    title: "Morphs & matching",
    group: "AUTHORING",
    description:
      "Deform structures and morph different proteins using contact maps.",
  },
  {
    slug: "trajectories",
    file: "trajectories.md",
    title: "NMR & trajectories",
    group: "AUTHORING",
    description:
      "Animate multi-model structures and read MD trajectories lazily.",
  },
  {
    slug: "full-example",
    file: "full-example.md",
    title: "Complete NMR script",
    group: "EXAMPLES",
    description:
      "A runnable film with camera focus, highlights, and real NMR models.",
  },
  {
    slug: "api",
    file: "api.md",
    title: "API reference",
    group: "REFERENCE",
    description: "The public Python API, parameters, and command-line tools.",
  },
  {
    slug: "rendering",
    file: "rendering.md",
    title: "Rendering & limits",
    group: "REFERENCE",
    description:
      "Native Metal, transparency, hardware encoding, and practical limits.",
  },
  {
    slug: "validation",
    file: "VALIDATION.md",
    title: "Validation & speed",
    group: "REFERENCE",
    description:
      "Measured M3 Max benchmarks, tests, provenance, and raw results.",
  },
  {
    slug: "contributing",
    file: "contributing.md",
    title: "Contributing",
    group: "PROJECT",
    description:
      "Develop the package, build the docs, and run meaningful checks.",
  },
];
export const demos = [
  {
    id: "regions",
    title: "Follow a region",
    file: "regions",
    label: "FOCUS + HIGHLIGHTS",
    detail:
      "Live spheres, wire boxes, and atom halos follow the helix and tail of ubiquitin through NMR conformers.",
    source: "examples/nmr_regions.py",
    scene: "RegionTour",
    duration: "24.1 s",
    pdb: "2K39",
  },
  {
    id: "nmr-cartoon",
    title: "116 conformers, one scene",
    file: "nmr-cartoon",
    label: "NMR · CARTOON",
    detail:
      "Every deposited model of ubiquitin, aligned on core Cα atoms and eased between adjacent states.",
    source: "examples/nmr_regions.py",
    scene: "NMRStates",
    duration: "24 s",
    pdb: "2K39",
  },
  {
    id: "nmr-atoms",
    title: "See the atoms move",
    file: "nmr-atoms",
    label: "NMR · BALL & STICK",
    detail:
      "The same ensemble with 602 heavy atoms per model and element-colored bonds and spheres.",
    source: "examples/nmr_regions.py",
    scene: "NMRAtoms",
    duration: "24 s",
    pdb: "2K39",
  },
  {
    id: "morph",
    title: "Between different proteins",
    file: "morph",
    label: "CONTACT-GUIDED MORPH",
    detail:
      "Calmodulin to troponin C: 114 matched Cα pairs move N-to-C while unmatched residues fade.",
    source: "examples/backbone_morph.py",
    scene: "BackboneDemo",
    duration: "11.5 s",
    pdb: "1CLL / 1NCX",
  },
  {
    id: "morph-atoms",
    title: "A closer look at morphing",
    file: "morph-atoms",
    label: "MORPH · BALL & STICK",
    detail:
      "Matched residues translate with their Cα atoms while the endpoint atom sets blend with smooth transparency.",
    source: "examples/backbone_morph.py",
    scene: "BallAndStickDemo",
    duration: "11.5 s",
    pdb: "1CLL / 1NCX",
  },
  {
    id: "groel",
    title: "Go beyond a single chain",
    file: "groel",
    label: "LARGE ASSEMBLY",
    detail:
      "The complete GroEL/GroES coordinate set: 21 chains, 8,015 Cα residues, and 58,870 selected heavy atoms.",
    source: "examples/large_protein.py",
    scene: "GroELComplex",
    duration: "9 s",
    pdb: "1AON",
  },
];
