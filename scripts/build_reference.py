"""Generate the reference index from Python source and reviewed API descriptions.

Uses only the standard library. Run with --check in website builds to detect drift.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/proteinmotion"
CATALOG = ROOT / "docs/reference/catalog.json"
OUTPUT = ROOT / "website/lib/reference-data.json"
SPECIAL = {"__len__", "__or__", "__enter__", "__exit__"}


def generate():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    trees = {p.stem: ast.parse(p.read_text(encoding="utf-8")) for p in sorted(SOURCE.glob("*.py"))}
    definitions = {}
    imports = {}
    for module, tree in trees.items():
        imports[module] = {}
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                definitions[f"{module}.{node.name}"] = node
            elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                for name in node.names:
                    imports[module][name.asname or name.name] = f"{node.module}.{name.name}"
    exports = next(
        ast.literal_eval(n.value)
        for n in trees["__init__"].body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    )
    version = next(
        ast.literal_eval(n.value)
        for n in trees["__init__"].body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    )
    entries = catalog["entries"]
    documented = {key.split(".")[-1] for key in entries} | {"rates"}
    if missing := set(exports) - documented:
        raise ValueError(f"Undocumented package exports: {sorted(missing)}")

    def bases(key):
        node = definitions[key]
        module = key.split(".")[0]
        if not isinstance(node, ast.ClassDef):
            return []
        names = [ast.unparse(base) for base in node.bases]
        return [imports[module].get(name, f"{module}.{name}") for name in names]

    def lineage(key):
        yield key
        for base in bases(key):
            if base in definitions:
                yield from lineage(base)

    def members(key):
        found = {}
        for owner in lineage(key):
            for node in definitions[owner].body:
                if isinstance(node, ast.FunctionDef) and (
                    not node.name.startswith("_") or node.name in SPECIAL or node.name == "__init__"
                ):
                    found.setdefault(node.name, (owner, node))
        return found

    def parameters(node):
        if isinstance(node, ast.ClassDef):
            # The public record classes are dataclasses with ordinary fields.
            return [
                dict(
                    name=f.target.id,
                    type=ast.unparse(f.annotation),
                    annotation=True,
                    default=ast.unparse(f.value) if f.value is not None else None,
                    kind="positional or keyword",
                )
                for f in node.body
                if isinstance(f, ast.AnnAssign) and isinstance(f.target, ast.Name)
            ]
        args = node.args
        positional = args.posonlyargs + args.args
        defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
        result = []
        for index, (arg, default) in enumerate(zip(positional, defaults)):
            if arg.arg in ("self", "cls"):
                continue
            result.append(
                dict(
                    name=arg.arg,
                    type=ast.unparse(arg.annotation) if arg.annotation else "",
                    annotation=bool(arg.annotation),
                    default=ast.unparse(default) if default is not None else None,
                    kind="positional only" if index < len(args.posonlyargs) else "positional or keyword",
                )
            )
        if args.vararg:
            result.append(dict(name=args.vararg.arg, type="", default=None, kind="variadic positional"))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            result.append(
                dict(
                    name=arg.arg,
                    type=ast.unparse(arg.annotation) if arg.annotation else "",
                    annotation=bool(arg.annotation),
                    default=ast.unparse(default) if default is not None else None,
                    kind="keyword only",
                )
            )
        if args.kwarg:
            result.append(dict(name=args.kwarg.arg, type="", default=None, kind="variadic keyword"))
        return result

    def describe_params(params, metadata, context):
        for param in params:
            name = param["name"]
            description = metadata.get("parameters", {}).get(name, catalog["parameters"].get(name))
            if description is None:
                raise ValueError(f"Describe parameter {context}.{name}")
            if isinstance(description, str):
                param["description"] = description
            else:
                param.update(description)
        return params

    def signature(name, params, property_=False):
        if property_:
            return name
        parts = []
        keyword_marker = False
        for i, p in enumerate(params):
            kind = p["kind"]
            if kind == "keyword only" and not keyword_marker:
                parts.append("*")
                keyword_marker = True
            prefix = "*" if kind == "variadic positional" else "**" if kind == "variadic keyword" else ""
            if kind == "variadic positional":
                keyword_marker = True
            part = prefix + p["name"]
            if p["type"] and p.get("annotation"):
                part += ": " + p["type"]
            if p["default"] is not None:
                part += "=" + p["default"]
            parts.append(part)
            if kind == "positional only" and (i + 1 == len(params) or params[i + 1]["kind"] != kind):
                parts.append("/")
        compact = name + "(" + ", ".join(parts) + ")"
        return compact if len(compact) < 90 else name + "(\n    " + ",\n    ".join(parts) + ",\n)"

    def member_metadata(key, owner, name, node):
        specific = entries[key].get("members", {}).get(name, {})
        parent = entries.get(owner, {}).get("members", {}).get(name, {})
        shared = catalog["members"].get(name, {})
        if isinstance(shared, str):
            shared = {"description": shared}
        if isinstance(parent, str):
            parent = {"description": parent}
        if isinstance(specific, str):
            specific = {"description": specific}
        result = shared | parent | specific
        result.setdefault("description", ast.get_docstring(node))
        if not result["description"]:
            raise ValueError(f"Describe member {key}.{name}")
        return result

    symbols = []
    for key, metadata in entries.items():
        module, name = key.split(".")
        alias = metadata.get("alias")
        node = definitions[alias or key]
        is_class = isinstance(node, ast.ClassDef)
        all_members = members(alias or key) if is_class else {}
        constructor_owner, constructor = all_members.get("__init__", (alias or key, node))
        params = describe_params(parameters(constructor), metadata, key)
        symbol = dict(
            id=key,
            name=name,
            module=module,
            kind="alias" if alias else "class" if is_class else "function",
            href=f"/reference/{module}/{name}/",
            description=metadata["description"],
            notes=metadata.get("notes", ""),
            import_path="proteinmotion" if name in exports else f"proteinmotion.{module}",
            signature=signature(name, params),
            parameters=params,
            returns=metadata.get("returns", ""),
            source=f"src/proteinmotion/{(alias or key).split('.')[0]}.py",
            line=node.lineno,
            bases=[base for base in bases(alias or key) if base in entries],
            attributes=metadata.get(
                "attributes",
                {p["name"]: p["description"] for p in params}
                if isinstance(constructor, ast.ClassDef)
                else {},
            ),
            guide=metadata.get("guide", catalog["modules"][module]["guide"]),
            example=metadata.get("example"),
            alias=alias,
            members=[],
            extra_parameters=[],
        )
        if forwarded := metadata.get("forward_parameters"):
            target, names = forwarded
            target_node = definitions[target]
            if isinstance(target_node, ast.ClassDef):
                target_node = members(target)["__init__"][1]
            extra = [p for p in parameters(target_node) if p["name"] in names]
            if {p["name"] for p in extra} != set(names):
                raise ValueError(f"Forwarded parameters changed: {key}")
            symbol["extra_parameters"] = describe_params(extra, metadata, key)
        for member_name, (owner, m) in all_members.items():
            if member_name == "__init__" or alias:
                continue
            md = member_metadata(key, owner, member_name, m)
            decorators = [ast.unparse(d) for d in m.decorator_list]
            kind = (
                "property"
                if "property" in decorators
                else "class method"
                if "classmethod" in decorators
                else "method"
            )
            mp = describe_params(parameters(m), md, f"{key}.{member_name}")
            symbol["members"].append(
                dict(
                    name=member_name,
                    kind=kind,
                    owner=owner,
                    inherited=owner != key,
                    signature=signature(f"{name}.{member_name}", mp, kind == "property"),
                    parameters=mp,
                    description=md["description"],
                    returns=md.get("returns", ""),
                    notes=md.get("notes", ""),
                    source=f"src/proteinmotion/{owner.split('.')[0]}.py",
                    line=m.lineno,
                )
            )
        symbols.append(symbol)
    dependencies = [CATALOG, Path(__file__).resolve(), *sorted(SOURCE.glob("*.py"))]
    return {
        "version": version,
        "exports": exports,
        "modules": [dict(id=key, **value) for key, value in catalog["modules"].items()],
        "symbols": symbols,
        "dependencies": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in dependencies
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(generate(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("Reference data is stale. Run python3 scripts/build_reference.py.")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print("Reference data checked." if args.check else f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
