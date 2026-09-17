export const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "/proteinmotion";
export const repo = "https://github.com/pdpppd/proteinmotion";
export const site = "https://pdpppd.github.io/proteinmotion";
const assetRevisions: Record<string, string> = {
  "media/showcase.mp4": "1",
  "media/showcase.jpg": "1",
  "media/alpha-helix.mp4": "0.6.2",
  "media/alpha-helix.jpg": "0.6.2",
  "downloads/alpha-helix.png": "0.6.2",
};
export const asset = (path: string) => {
  const clean = path.replace(/^\//, "");
  const revision = assetRevisions[clean];
  return `${base}/${clean}${revision ? `?v=${revision}` : ""}`;
};
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
    slug: "text",
    file: "text.md",
    title: "Text & labels",
    group: "AUTHORING",
    description:
      "Write vector text, name amino acids, and anchor callouts to moving regions.",
  },
  {
    slug: "styling",
    file: "styling.md",
    title: "Colors & surfaces",
    group: "AUTHORING",
    description:
      "Color residues, ease their opacity, and render moving molecular surfaces.",
  },
  {
    slug: "interactions",
    file: "interactions.md",
    title: "Distances & interactions",
    group: "AUTHORING",
    description:
      "Measure atoms and residues, highlight hydrogen bonds, and import charges for screened electrostatics.",
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
    slug: "showcase",
    file: "showcase.md",
    title: "Complete feature tour",
    group: "EXAMPLES",
    description:
      "One 66-second film covering every major visual capability, with chapters and runnable source.",
  },
  {
    slug: "alpha-helix",
    file: "alpha-helix.md",
    title: "Alpha-helix H bonds",
    group: "EXAMPLES",
    description:
      "Verify the full i-to-i+4 network with visible hydrogens and a clear bond close-up.",
  },
  {
    slug: "molecular-example",
    file: "molecular-example.md",
    title: "Molecular tools script",
    group: "EXAMPLES",
    description:
      "Two complete films: styled surfaces and live molecular interaction rulers.",
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
    id: "showcase",
    title: "ProteinMotion in 66 seconds",
    file: "showcase",
    label: "THE COMPLETE FEATURE TOUR",
    detail:
      "Ten chapters: representations, residue styling, focus and labels, ensembles and MD readers, deformation, contact-guided morphs, hydrogen bonds, distance rulers, charge contacts, and a 58,870-atom complex.",
    source: "examples/feature_showcase.py",
    scene: "FeatureShowcase",
    duration: "66 s",
    pdb: "",
  },
  {
    id: "alpha-helix",
    title: "The bonds that hold a helix",
    file: "alpha-helix",
    label: "ALPHA-HELIX HYDROGEN-BOND TEST",
    detail:
      "A fixed idealized backbone with explicit H: all 12 expected i-to-i+4 bonds, then a close-up distinguishing gold H···O contacts from solid N–H bonds. Detection uses geometry, without a sequence-offset rule.",
    source: "examples/alpha_helix_hbonds.py",
    scene: "AlphaHelixHBonds",
    duration: "23 s",
    pdb: "",
  },
  {
    id: "surfaces",
    title: "Color is part of the story",
    file: "surfaces",
    label: "RESIDUE COLOR + MOVING SURFACES",
    detail:
      "Staggered color changes and local transparency carry from cartoon to atoms to a rebuilt solvent-excluded surface through ubiquitin NMR conformers.",
    source: "examples/molecular_tools.py",
    scene: "StylingAndSurface",
    duration: "24.8 s",
    pdb: "2K39",
  },
  {
    id: "interactions",
    title: "Measure the connections",
    file: "interactions",
    label: "HYDROGEN BONDS + SCREENED ELECTROSTATICS",
    detail:
      "Depth-tested 3D rulers and 2D overlay distances, geometric hydrogen bonds with virtual backbone H, and screened Coulomb contacts using illustrative formal charges.",
    source: "examples/molecular_tools.py",
    scene: "InteractionsAndDistances",
    duration: "20.2 s",
    pdb: "2K39",
  },
  {
    id: "labels",
    title: "Give each region a voice",
    file: "labels",
    label: "VECTOR TEXT + LIVE CALLOUTS",
    detail:
      "Contour-to-fill writing, region callouts, and automatic amino acid labels follow ubiquitin through NMR conformers in cartoon and ball-and-stick.",
    source: "examples/labels_and_callouts.py",
    scene: "ProteinLabels",
    duration: "18.8 s",
    pdb: "2K39",
  },
  {
    id: "writing",
    title: "Watch the letters take shape",
    file: "writing",
    label: "MANIM-STYLE WRITE",
    detail:
      "A close-up of cached vector glyphs drawing their contours, filling in, then erasing. Native Metal rendering, with Greek letters and configurable glyph timing.",
    source: "examples/labels_and_callouts.py",
    scene: "WritingStudy",
    duration: "10 s",
    pdb: "",
  },
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
