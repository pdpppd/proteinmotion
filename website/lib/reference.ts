import data from "./reference-data.json";
import rendered from "../public/media/docs/manifest.json";
import { demoCommand, demos } from "./config";
import { docExample } from "./doc-examples";

export type Parameter = {
  name: string;
  type: string;
  default: string | null;
  kind: string;
  description: string;
};
export type Member = {
  name: string;
  kind: string;
  owner: string;
  inherited: boolean;
  signature: string;
  parameters: Parameter[];
  description: string;
  returns: string;
  notes: string;
  source: string;
  line: number;
};
export type Symbol = {
  id: string;
  name: string;
  module: string;
  kind: string;
  href: string;
  description: string;
  notes: string;
  import_path: string;
  signature: string;
  parameters: Parameter[];
  extra_parameters: Parameter[];
  returns: string;
  source: string;
  line: number;
  bases: string[];
  attributes: Record<string, string>;
  guide: string;
  example: string | null;
  alias: string | null;
  members: Member[];
};
export const reference = data as unknown as {
  version: string;
  exports: string[];
  modules: { id: string; title: string; description: string; guide: string }[];
  symbols: Symbol[];
};
export const symbolById = (id: string) =>
  reference.symbols.find((s) => s.id === id);
export const referencePaths = [
  [],
  ...reference.modules.map((m) => [m.id]),
  ...reference.symbols.map((s) => [s.module, s.name]),
];
export const referenceSearch = () => [
  {
    title: "Reference manual",
    href: "/reference/",
    description: "Modules, classes, functions, and methods.",
    text: "Python API reference index",
  },
  ...reference.modules.map((m) => ({
    title: `proteinmotion.${m.id}`,
    href: `/reference/${m.id}/`,
    description: m.description,
    text: m.title,
  })),
  ...reference.symbols.flatMap((s) => [
    {
      title: s.name,
      href: s.href,
      description: s.description,
      text: [s.id, s.notes, s.parameters.map((p) => p.name).join(" ")].join(
        " ",
      ),
    },
    ...s.members.map((m) => ({
      title: `${s.name}.${m.name}`,
      href: `${s.href}#${m.name}`,
      description: m.description,
      text: [s.id, m.signature, m.notes].join(" "),
    })),
  ]),
];
export function referenceExample(id: string) {
  if (id.startsWith("gallery-")) {
    const demo = demos.find((d) => d.id === id.slice(8));
    if (!demo) throw new Error(`Unknown reference example ${id}`);
    return `\`\`\`bash output=${id}\n${demoCommand(demo)}\n\`\`\``;
  }
  // The existing renderer validates this exact source against its clip manifest.
  const example = docExampleForReference(id);
  return `\`\`\`python output=${id}\n${example.code}\n\`\`\``;
}
// Read through the same manifest as the guide code/output pairs.
function docExampleForReference(id: string) {
  const item = (rendered as unknown as Record<string, { code: string }>)[id];
  if (!item?.code)
    throw new Error(`Reference example ${id} needs a scene excerpt`);
  return docExample(id, item.code);
}
