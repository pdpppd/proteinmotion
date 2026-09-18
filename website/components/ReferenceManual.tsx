import Link from "next/link";
import { codeHTML, compile } from "@/lib/content";
import { repo } from "@/lib/config";
import {
  reference,
  symbolById,
  referenceExample,
  type Parameter,
  type Symbol,
} from "@/lib/reference";

function ApiText({ text }: { text: string }) {
  const names = reference.symbols
    .map((symbol) => symbol.name)
    .sort((a, b) => b.length - a.length);
  const pattern = new RegExp(`\\b(${names.join("|")})\\b`, "g");
  return (
    <>
      {text.split(pattern).map((part, index) => {
        const symbol = reference.symbols.find((symbol) => symbol.name === part);
        return symbol ? (
          <Link
            key={index}
            href={symbol.href}
            className="text-accent underline decoration-accent/30 underline-offset-4"
          >
            {part}
          </Link>
        ) : (
          part
        );
      })}
    </>
  );
}

export function ReferenceNav({
  module,
  symbol,
}: {
  module?: string;
  symbol?: string;
}) {
  return (
    <nav aria-label="Reference modules" className="reference-nav">
      <Link href="/reference/" className="mb-4 block font-medium">
        Reference manual
      </Link>
      <Link href="/docs/getting-started/" className="mb-6 block text-muted">
        User guides →
      </Link>
      {reference.modules.map((m) => (
        <details key={`${m.id}-${module}`} open={m.id === module}>
          <summary className="py-2 text-sm">{m.title}</summary>
          <div className="ml-2 border-l border-line pl-3">
            <Link
              href={`/reference/${m.id}/`}
              aria-current={m.id === module && !symbol ? "page" : undefined}
            >
              Module overview
            </Link>
            {reference.symbols
              .filter((s) => s.module === m.id)
              .map((s) => (
                <Link
                  key={s.id}
                  href={s.href}
                  aria-current={s.name === symbol ? "page" : undefined}
                  className="font-mono"
                >
                  {s.name}
                </Link>
              ))}
          </div>
        </details>
      ))}
    </nav>
  );
}
export function Signature({ text }: { text: string }) {
  return <div dangerouslySetInnerHTML={{ __html: codeHTML(text, "python") }} />;
}
export function Parameters({ parameters }: { parameters: Parameter[] }) {
  if (!parameters.length) return null;
  return (
    <div className="reference-table">
      <table>
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Default</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {parameters.map((p) => (
            <tr key={p.name}>
              <td>
                <code>
                  {p.kind === "variadic keyword"
                    ? "**"
                    : p.kind === "variadic positional"
                      ? "*"
                      : ""}
                  {p.name}
                </code>
                {p.type && (
                  <span className="mt-1 block font-mono text-[11px] text-muted">
                    {p.type}
                  </span>
                )}
                {p.kind === "keyword only" && (
                  <span className="mt-1 block text-[10px] text-muted">
                    keyword only
                  </span>
                )}
              </td>
              <td>
                {p.default === null ? (
                  <span className="text-muted">
                    {p.kind.startsWith("variadic") ? "—" : "Required"}
                  </span>
                ) : (
                  <code>{p.default}</code>
                )}
              </td>
              <td>
                <ApiText text={p.description} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function SymbolTable({ symbols }: { symbols: Symbol[] }) {
  return (
    <div className="reference-table">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {symbols.map((s) => (
            <tr key={s.id}>
              <td>
                <Link className="font-mono text-accent" href={s.href}>
                  {s.name}
                  {s.kind === "function" ? "()" : ""}
                </Link>
                <span className="mt-1 block text-[10px] text-muted">
                  {s.kind}
                </span>
              </td>
              <td>
                <ApiText text={s.description} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function ReferenceIndex() {
  return (
    <>
      <h1 className="reference-title">Reference manual</h1>
      <p className="reference-intro">
        Classes, functions, and methods for building protein animations. Browse
        a module below, or search for an API name such as{" "}
        <code>Protein.surface</code>.
      </p>
      <p className="reference-intro">
        For installation and step-by-step examples, start with the{" "}
        <Link href="/docs/getting-started/">user guides</Link>. API signatures
        and defaults are read from the package source.
      </p>
      <div className="my-8 flex flex-wrap gap-x-6 gap-y-2 text-sm">
        <a className="text-accent" href="#module-index">
          Module index
        </a>
        <a className="text-accent" href="#symbol-index">
          Alphabetical index
        </a>
        <Link className="text-accent" href="/docs/api/#command-line">
          Command line
        </Link>
      </div>
      <h2 id="module-index" className="reference-heading">
        Module index
      </h2>
      <div className="grid gap-x-10 xl:grid-cols-2">
        {reference.modules.map((m) => (
          <section className="min-w-0 border-t border-line py-6" key={m.id}>
            <Link
              href={`/reference/${m.id}/`}
              className="text-lg font-medium text-accent"
            >
              {m.title}
            </Link>
            <p className="mt-1 font-mono text-[11px] text-muted">
              proteinmotion.{m.id}
            </p>
            <p className="mt-3 text-sm leading-6 text-muted">{m.description}</p>
            <ul className="mt-4 flex flex-wrap gap-x-4 gap-y-2 text-xs">
              {reference.symbols
                .filter((s) => s.module === m.id)
                .map((s) => (
                  <li key={s.id}>
                    <Link
                      href={s.href}
                      className="font-mono hover:text-accent hover:underline"
                    >
                      {s.name}
                    </Link>
                  </li>
                ))}
            </ul>
          </section>
        ))}
      </div>
      <h2 id="symbol-index" className="reference-heading">
        Alphabetical index
      </h2>
      <SymbolTable
        symbols={[...reference.symbols].sort((a, b) =>
          a.name.localeCompare(b.name),
        )}
      />
    </>
  );
}
export function ModulePage({ id }: { id: string }) {
  const m = reference.modules.find((m) => m.id === id)!;
  const symbols = reference.symbols.filter((s) => s.module === id);
  return (
    <>
      <h1 className="reference-title">{m.title}</h1>
      <p className="font-mono text-sm text-muted">proteinmotion.{id}</p>
      <p className="reference-intro">
        <ApiText text={m.description} />
      </p>
      <Link
        href={`/docs/${m.guide}/`}
        className="text-sm text-accent underline underline-offset-4"
      >
        Guide and rendered examples →
      </Link>
      {id === "rates" && (
        <div className="mt-6">
          <Signature text="from proteinmotion import rates\n\nself.play(animation, run_time=2, rate_func=rates.smooth)" />
        </div>
      )}
      {["class", "function", "alias"].map((kind) => {
        const entries = symbols.filter((s) => s.kind === kind);
        return (
          entries.length > 0 && (
            <section key={kind}>
              <h2 className="reference-heading">
                {kind === "class"
                  ? "Classes"
                  : kind === "function"
                    ? "Functions"
                    : "Aliases"}
              </h2>
              <SymbolTable symbols={entries} />
            </section>
          )
        );
      })}
      <a
        href={`${repo}/blob/main/src/proteinmotion/${id}.py`}
        className="mt-8 inline-block text-xs text-muted underline underline-offset-4"
      >
        View module source
      </a>
    </>
  );
}
export function SymbolPage({ symbol: s }: { symbol: Symbol }) {
  const alias = s.alias ? symbolById(s.alias) : undefined;
  return (
    <>
      <h1 className="reference-title font-mono">{s.name}</h1>
      <p className="reference-intro">{s.description}</p>
      <div className="mb-6 flex flex-wrap gap-5 text-xs text-muted">
        <a
          href={`${repo}/blob/main/${s.source}#L${s.line}`}
          className="underline underline-offset-4"
        >
          View source
        </a>
        <Link
          href={`/docs/${s.guide}/`}
          className="text-accent underline underline-offset-4"
        >
          Guide and examples
        </Link>
        {s.bases.length > 0 && (
          <span>
            Inherits{" "}
            {s.bases.map((base, i) => (
              <span key={base}>
                {i > 0 && ", "}
                <Link
                  className="font-mono text-accent"
                  href={symbolById(base)!.href}
                >
                  {symbolById(base)!.name}
                </Link>
              </span>
            ))}
          </span>
        )}
      </div>
      <Signature text={`from ${s.import_path} import ${s.name}`} />
      {alias ? (
        <p className="reference-intro">
          Uses the same constructor and methods as{" "}
          <Link href={alias.href}>{alias.name}</Link>.
        </p>
      ) : (
        <>
          <h2 className="reference-heading" id="signature">
            {s.kind === "class" ? "Constructor" : "Signature"}
          </h2>
          <Signature text={s.signature} />
          {s.parameters.length > 0 && (
            <>
              <h3 className="reference-subheading" id="parameters">
                Parameters
              </h3>
              <Parameters parameters={s.parameters} />
            </>
          )}
          {s.extra_parameters.length > 0 && (
            <>
              <h3 className="reference-subheading">
                Additional keyword arguments
              </h3>
              <Parameters parameters={s.extra_parameters} />
            </>
          )}
          {s.returns && (
            <p className="reference-intro">
              <strong>Returns:</strong> <ApiText text={s.returns} />
            </p>
          )}
          {s.notes && (
            <section>
              <h2 className="reference-heading" id="notes">
                Notes
              </h2>
              <p className="reference-intro whitespace-pre-line">
                <ApiText text={s.notes} />
              </p>
            </section>
          )}
          {Object.keys(s.attributes).length > 0 && (
            <section>
              <h2 className="reference-heading" id="attributes">
                Attributes
              </h2>
              <div className="reference-table">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(s.attributes).map(([name, text]) => (
                      <tr key={name}>
                        <td>
                          <code>{name}</code>
                        </td>
                        <td>
                          <ApiText text={text} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
          {s.example && (
            <section>
              <h2 className="reference-heading" id="example">
                Example and output
              </h2>
              <p className="reference-intro">
                {s.example.startsWith("gallery-") ? (
                  "Run this command from a repository checkout."
                ) : (
                  <>
                    This excerpt runs inside a scene’s construct() method.{" "}
                    <a href={`${repo}/blob/main/examples/docs_examples.py`}>
                      The full example file
                    </a>{" "}
                    includes imports, structure loading, and camera setup. Run
                    it from a repository checkout.
                  </>
                )}
              </p>
              <div
                className="prose-doc"
                dangerouslySetInnerHTML={{
                  __html: compile(referenceExample(s.example)).html,
                }}
              />
            </section>
          )}
          {s.members.length > 0 && (
            <>
              <h2 className="reference-heading" id="methods">
                Methods and properties
              </h2>
              <div className="reference-table">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.members.map((m) => (
                      <tr key={m.name}>
                        <td>
                          <a
                            href={`#${m.name}`}
                            className="font-mono text-accent"
                          >
                            {m.name}
                            {m.kind === "property" ? "" : "()"}
                          </a>
                          <span className="mt-1 block text-[10px] text-muted">
                            {m.kind}
                            {m.inherited
                              ? ` · inherited from ${m.owner.split(".")[1]}`
                              : ""}
                          </span>
                        </td>
                        <td>
                          <ApiText text={m.description} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {s.members.map((m) => (
                <section className="reference-member" id={m.name} key={m.name}>
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <h3 className="font-mono text-base font-medium">
                      <a href={`#${m.name}`}>
                        {m.name}
                        {m.kind === "property" ? "" : "()"}
                      </a>
                    </h3>
                    <a
                      href={`${repo}/blob/main/${m.source}#L${m.line}`}
                      className="text-xs text-muted underline underline-offset-4"
                    >
                      View source
                    </a>
                  </div>
                  {m.inherited && (
                    <p className="mb-3 text-xs text-muted">
                      Inherited from{" "}
                      {symbolById(m.owner) ? (
                        <Link
                          href={`${symbolById(m.owner)!.href}#${m.name}`}
                          className="text-accent"
                        >
                          {m.owner}
                        </Link>
                      ) : (
                        m.owner
                      )}
                      .
                    </p>
                  )}
                  <Signature text={m.signature} />
                  <p className="reference-intro">
                    <ApiText text={m.description} />
                  </p>
                  <Parameters parameters={m.parameters} />
                  {m.returns && (
                    <p className="reference-intro">
                      <strong>Returns:</strong> <ApiText text={m.returns} />
                    </p>
                  )}
                  {m.notes && (
                    <p className="reference-intro">
                      <ApiText text={m.notes} />
                    </p>
                  )}
                </section>
              ))}
            </>
          )}
        </>
      )}
    </>
  );
}
