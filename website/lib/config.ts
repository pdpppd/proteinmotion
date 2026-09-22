export const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "/proteinmotion";
export const repo = "https://github.com/pdpppd/proteinmotion";
export const pypi = "https://pypi.org/project/proteinmotion/";
export const site = "https://pdpppd.github.io/proteinmotion";
const assetRevisions: Record<string, string> = {
  "media/showcase.mp4": "2",
  "media/showcase.jpg": "2",
  "downloads/showcase.png": "2",
  "media/alpha-helix.mp4": "0.6.2",
  "media/alpha-helix.jpg": "0.6.2",
  "downloads/alpha-helix.png": "0.6.2",
};
export const asset = (path: string) => {
  const clean = path.replace(/^\//, "");
  const revision = assetRevisions[clean];
  return `${base}/${clean}${revision ? `?v=${revision}` : ""}`;
};
export const guideAliases: Record<string, string> = {
  "codex-skill": "agent-skill",
};
export const guides = [
  {
    slug: "getting-started",
    file: "getting-started.md",
    title: "Get started",
    group: "START HERE",
    description: "Install on macOS or Windows with NVIDIA graphics and render a video.",
  },
  {
    slug: "agent-skill",
    file: "agent-skill.md",
    title: "AI agent skill",
    group: "START HERE",
    description: "Set up the movie-making skill for your AI agent.",
  },
  {
    slug: "scenes",
    file: "scenes.md",
    title: "Scenes & motion",
    group: "AUTHORING",
    description: "Load a protein, choose a representation, and animate it.",
  },
  {
    slug: "dna-rna",
    file: "dna-rna.md",
    title: "DNA & RNA",
    group: "AUTHORING",
    description:
      "Draw base slabs, rings, sticks, and surfaces. Animate nucleotide colors and opacity.",
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
    slug: "eevee",
    file: "eevee.md",
    title: "EEVEE & depth of field",
    group: "AUTHORING",
    description: "Render with Blender EEVEE and focus on selected residues.",
  },
  {
    slug: "text",
    file: "text.md",
    title: "Text & labels",
    group: "AUTHORING",
    description:
      "Add text, amino acid labels, and lines that point to selected regions.",
  },
  {
    slug: "styling",
    file: "styling.md",
    title: "Colors & surfaces",
    group: "AUTHORING",
    description:
      "Set residue colors and transparency, and render molecular surfaces.",
  },
  {
    slug: "numerical-properties",
    file: "numerical-properties.md",
    title: "Numerical properties",
    group: "AUTHORING",
    description:
      "Map B factors, RMSF, and custom residue values to color and thickness.",
  },
  {
    slug: "synchronized-plots",
    file: "synchronized-plots.md",
    title: "Plots & sequence tracks",
    group: "AUTHORING",
    description:
      "Add distance traces, live contact maps, and sequence strips to a movie.",
  },
  {
    slug: "density-maps",
    file: "density-maps.md",
    title: "Density maps & slices",
    group: "AUTHORING",
    description: "Load MRC/CCP4 maps and animate contours and slices.",
  },
  {
    slug: "interactions",
    file: "interactions.md",
    title: "Distances & interactions",
    group: "AUTHORING",
    description:
      "Add distance labels, detect hydrogen bonds, and estimate electrostatic interactions.",
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
      "Animate NMR ensembles and read MD trajectories one frame at a time.",
  },
  {
    slug: "full-example",
    file: "full-example.md",
    title: "Complete NMR script",
    group: "EXAMPLES",
    description:
      "Animate a ubiquitin NMR ensemble with camera focus and region highlights.",
  },
  {
    slug: "calmodulin-in-focus",
    file: "calmodulin-in-focus.md",
    title: "Calmodulin in focus",
    group: "EXAMPLES",
    description:
      "A 68-second EEVEE film with helix close-ups, focus pulls, and molecular surfaces.",
  },
  {
    slug: "showcase",
    file: "showcase.md",
    title: "Feature demo",
    group: "EXAMPLES",
    description:
      "A calmodulin and troponin C video with chapter times and source code.",
  },
  {
    slug: "alpha-helix",
    file: "alpha-helix.md",
    title: "Alpha-helix H bonds",
    group: "EXAMPLES",
    description:
      "Show the i-to-i+4 hydrogen bonds in an idealized alpha helix.",
  },
  {
    slug: "molecular-example",
    file: "molecular-example.md",
    title: "Molecular tools script",
    group: "EXAMPLES",
    description:
      "Scripts for residue colors, surfaces, distance labels, and interactions.",
  },
  {
    slug: "api",
    file: "api.md",
    title: "API quick reference",
    group: "REFERENCE",
    description: "The public Python API, parameters, and command-line tools.",
  },
  {
    slug: "rendering",
    file: "rendering.md",
    title: "Rendering",
    group: "REFERENCE",
    description: "How the renderer, transparency, and video export work.",
  },
  {
    slug: "validation",
    file: "VALIDATION.md",
    title: "Tests & benchmarks",
    group: "REFERENCE",
    description: "Test results, Apple M3 Max and NVIDIA RTX rendering times, and input data.",
  },
  {
    slug: "contributing",
    file: "contributing.md",
    title: "Contributing",
    group: "PROJECT",
    description: "Set up development tools, build the website, and run checks.",
  },
];
export type Demo = {
  id: string;
  title: string;
  file: string;
  label: string;
  detail: string;
  source: string;
  scene: string;
  duration: string;
  pdb: string;
  fps?: number;
  renderCommand?: string;
};
export const demoCommand = (demo: Demo) =>
  demo.renderCommand ??
  `proteinmotion render ${demo.source} ${demo.scene} \\\n  -o ${demo.file}.mp4 --fps 60`;
export const demos: Demo[] = [
  {
    id: "dna-morph",
    title: "DNA morph with C1′ anchors",
    file: "docs/dna-morph",
    label: "DNA · CONTACT-GUIDED MORPH",
    detail:
      "C1′ contact maps select five matching nucleotides between two DNA strands. Matched residues move in order with a 0.25-second delay; unmatched residues fade out and in.",
    source: "examples/dna_morph.py",
    scene: "DNAMorph",
    duration: "12.2 s",
    pdb: "1BNA → 2DCG",
    fps: 60,
    renderCommand:
      "proteinmotion render examples/dna_morph.py DNAMorph --fps 60 -o dna-morph.mp4",
  },
  {
    id: "dna-styles",
    title: "DNA base styles",
    file: "docs/dna-styles",
    label: "DNA · BASE REPRESENTATIONS",
    detail:
      "A DNA double helix with slabs, filled rings, sticks, and ladder rods. Residue colors and strand transparency carry through to atoms and a molecular surface.",
    source: "examples/dna_styles.py",
    scene: "DNAStyles",
    duration: "23.2 s",
    pdb: "1BNA",
    fps: 60,
    renderCommand:
      "proteinmotion render examples/dna_styles.py DNAStyles --fps 60 -o dna.mp4",
  },
  {
    id: "rna-styles",
    title: "Transfer RNA",
    file: "docs/rna-styles",
    label: "RNA · REGIONS AND SURFACES",
    detail:
      "Select the anticodon loop of yeast tRNA, fade its surroundings, then color a molecular surface by deposited B factors. Modified nucleotides retain their names and parent-base colors.",
    source: "examples/rna_styles.py",
    scene: "RNAStyles",
    duration: "14.5 s",
    pdb: "1EHZ",
    fps: 60,
    renderCommand:
      "proteinmotion render examples/rna_styles.py RNAStyles --fps 60 -o rna.mp4",
  },
  {
    id: "calmodulin-in-focus",
    title: "Calmodulin in focus",
    file: "calmodulin-in-focus",
    label: "EEVEE · DEPTH OF FIELD",
    detail:
      "A continuous camera tour with helix close-ups, focus pulls, transparent surroundings, backbone atoms, and a surface colored by B factor. Rendered with Blender EEVEE at 1080p/60 fps.",
    source: "examples/calmodulin_in_focus.py",
    scene: "CalmodulinInFocus",
    duration: "68 s",
    pdb: "1CLL",
    fps: 60,
    renderCommand:
      "python examples/calmodulin_in_focus.py \\\n  --output calmodulin-in-focus.mp4",
  },
  {
    id: "showcase",
    title: "Calmodulin and troponin C",
    file: "showcase",
    label: "FEATURE DEMO",
    detail:
      "Calmodulin changes representation, color, and conformation, then morphs into troponin C. The video also shows labels, camera focus, distances, hydrogen bonds, and electrostatic estimates.",
    source: "examples/feature_showcase.py",
    scene: "FeatureShowcase",
    duration: "100.5 s",
    pdb: "",
    fps: 60,
  },
  {
    id: "alpha-helix",
    title: "Alpha-helix hydrogen bonds",
    file: "alpha-helix",
    label: "ALPHA-HELIX HYDROGEN-BOND TEST",
    detail:
      "An idealized backbone with explicit hydrogens. The video shows all 12 expected i-to-i+4 hydrogen bonds, then zooms into one bond to compare H···O and N···O distances.",
    source: "examples/alpha_helix_hbonds.py",
    scene: "AlphaHelixHBonds",
    duration: "23 s",
    pdb: "",
  },
  {
    id: "surfaces",
    title: "Residue colors and surfaces",
    file: "surfaces",
    label: "RESIDUE COLOR + MOVING SURFACES",
    detail:
      "Residue colors change in sequence across a ubiquitin structure. The video switches between cartoon, ball-and-stick, and surface views, then updates the surface during NMR playback.",
    source: "examples/molecular_tools.py",
    scene: "StylingAndSurface",
    duration: "24.8 s",
    pdb: "2K39",
  },
  {
    id: "interactions",
    title: "Distances and interactions",
    file: "interactions",
    label: "HYDROGEN BONDS + SCREENED ELECTROSTATICS",
    detail:
      "Compare 3D and 2D distance lines. Highlight hydrogen bonds using inferred backbone hydrogens, then display screened Coulomb estimates using example formal charges.",
    source: "examples/molecular_tools.py",
    scene: "InteractionsAndDistances",
    duration: "20.2 s",
    pdb: "2K39",
  },
  {
    id: "labels",
    title: "Residue labels and callouts",
    file: "labels",
    label: "TEXT + CALLOUTS",
    detail:
      "Add amino acid names and region labels to ubiquitin. Lines connect the labels to the selected residues as the structure moves.",
    source: "examples/labels_and_callouts.py",
    scene: "ProteinLabels",
    duration: "18.8 s",
    pdb: "2K39",
  },
  {
    id: "writing",
    title: "Text writing animation",
    file: "writing",
    label: "MANIM-STYLE WRITE",
    detail:
      "Write draws each letter’s outline and fills it in. Unwrite erases the text. The example includes Greek letters and delays between letters.",
    source: "examples/labels_and_callouts.py",
    scene: "WritingStudy",
    duration: "10 s",
    pdb: "",
  },
  {
    id: "regions",
    title: "Region focus and highlights",
    file: "regions",
    label: "FOCUS + HIGHLIGHTS",
    detail:
      "The camera focuses on the helix and tail of ubiquitin. Spheres, boxes, and atom highlights mark the selected regions during NMR playback.",
    source: "examples/nmr_regions.py",
    scene: "RegionTour",
    duration: "24.1 s",
    pdb: "2K39",
  },
  {
    id: "nmr-cartoon",
    title: "NMR ensemble in cartoon view",
    file: "nmr-cartoon",
    label: "NMR · CARTOON",
    detail:
      "All 116 deposited ubiquitin models, aligned using core Cα atoms. The animation interpolates between consecutive models.",
    source: "examples/nmr_regions.py",
    scene: "NMRStates",
    duration: "24 s",
    pdb: "2K39",
  },
  {
    id: "nmr-atoms",
    title: "NMR ensemble in ball-and-stick",
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
    title: "Calmodulin to troponin C morph",
    file: "morph",
    label: "CONTACT-GUIDED MORPH",
    detail:
      "A contact-map match selects 114 Cα pairs. Matched residues move in sequence from N to C; unmatched residues fade out or in.",
    source: "examples/backbone_morph.py",
    scene: "BackboneDemo",
    duration: "11.5 s",
    pdb: "1CLL / 1NCX",
  },
  {
    id: "morph-atoms",
    title: "Ball-and-stick morph",
    file: "morph-atoms",
    label: "MORPH · BALL & STICK",
    detail:
      "Each matched residue moves with its Cα atom. Source atoms fade out as target atoms fade in.",
    source: "examples/backbone_morph.py",
    scene: "BallAndStickDemo",
    duration: "11.5 s",
    pdb: "1CLL / 1NCX",
  },
  {
    id: "groel",
    title: "GroEL/GroES assembly",
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
