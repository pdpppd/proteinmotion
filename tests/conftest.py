from pathlib import Path

import pytest

from proteinmotion import Protein


@pytest.fixture
def structure_path():
    return Path(__file__).resolve().parents[1] / "examples/data/1ubq.cif"


@pytest.fixture
def protein(structure_path):
    return Protein.from_file(structure_path).center()
