"""Cached downloads of PDB entries and AlphaFold DB models."""

import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path

PDB_URL = "https://files.rcsb.org/download/{identifier}.cif"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def cache_directory(cache=None):
    root = (
        Path(cache)
        if cache is not None
        else Path(os.environ.get("PROTEINMOTION_CACHE", "~/.cache/proteinmotion"))
    )
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download(url, destination):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def fetch_structure(identifier, *, cache=None):
    """Path to a cached mmCIF file for a PDB ID or an AlphaFold DB entry, downloading it once."""
    identifier = str(identifier).strip()
    root = cache_directory(cache)
    if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", identifier):
        path = root / f"{identifier.lower()}.cif"
        return path if path.is_file() else _download(PDB_URL.format(identifier=identifier.upper()), path)
    match = re.fullmatch(r"AF-([A-Za-z0-9]+)-F(\d+)", identifier, re.IGNORECASE)
    if match:
        found = sorted(root.glob(f"AF-{match[1].upper()}-F{match[2]}-model_v*.cif"))
        if found:
            return found[-1]
        with urllib.request.urlopen(ALPHAFOLD_API.format(accession=match[1].upper()), timeout=60) as response:
            entries = json.loads(response.read())
        entry = next((e for e in entries if e.get("entryId", "").upper() == identifier.upper()), None)
        if entry is None:
            raise ValueError(f"AlphaFold DB has no entry {identifier}")
        url = entry["cifUrl"]
        return _download(url, root / url.rsplit("/", 1)[-1])
    raise ValueError("Use a four-character PDB ID such as 4HHB or an AlphaFold DB ID such as AF-P0DP23-F1")
