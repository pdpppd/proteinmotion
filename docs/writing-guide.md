# Writing guide

Write for a reader who knows basic Python and protein structure. Explain what each feature does, how to use it, and what the result means.

These conventions adapt the [Google developer documentation style guide](https://developers.google.com/style/highlights) to ProteinMotion. They use simplified technical English: short sentences, common verbs, and consistent technical terms. Formal [ASD-STE100](https://www.asd-ste100.org/) compliance would require a separate vocabulary and rules review.

## Sentences and terms

- Name the object and its action: “The line connects the label to the selected residue.”
- Keep one main idea in each sentence. Split a sentence when its conditions or exceptions make it hard to follow.
- Use common verbs such as load, select, draw, move, and export.
- Use the same term for the same feature. Use *video* for the exported result and *scene* for the Python object that defines it.
- Keep established terms such as Cα, residue, trajectory, and solvent-accessible surface. Explain a term when readers need it to follow the instructions.
- State behavior directly. Put a relevant restriction beside the feature it affects.
- Use a negative statement when it prevents a specific mistake. Avoid repeated lists of what a feature is not.

## Page structure

Start with the purpose of the page. Give the shortest usable example, then explain its options. Put implementation details in the rendering or API reference.

Use headings that name a task or feature: “Install,” “Select residues,” or “Distance labels.” Use numbered steps for a sequence and bullets for independent options.

Keep introductory pages focused on installation and common tasks. Put benchmark conditions, method assumptions, and detailed test results in their reference pages. Preserve units, defaults, and scientific qualifications during edits.

## Tone

Describe features with facts. Remove slogans, praise, metaphors, and claims of simplicity or speed that add no useful information. A benchmark needs its hardware, input, output settings, and timing scope.

Address AI agents generally. Name a particular product only when explaining its setup or behavior. Distinguish instructions an agent can read from automatic skill discovery, which depends on the agent.

## Examples

| Avoid | Use |
|---|---|
| Apple silicon/macOS is the verified platform. | Rendering is tested on Apple silicon Macs. |
| Motion with intent | Animation |
| Give each region a voice | Region labels |
| The text stays screen-facing while the leaders follow live 3D anchors. | Labels appear over the video. A line connects each label to its selected region. |
| Make movies with Codex | Use with an AI agent |
| Real structural ensembles | NMR ensembles |

## Review

Read the prose separately from the code. Check that each paragraph answers a reader's question and that the examples match the documented API. Remove repeated qualifications. Check headings, captions, navigation, search descriptions, and metadata as well as the page body.

Build the website and check its links after changing page names or headings. Review the homepage and a guide at desktop and mobile widths.
