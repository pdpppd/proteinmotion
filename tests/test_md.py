import gemmi
import numpy as np
import pytest

from proteinmotion import Protein, Trajectory


@pytest.mark.parametrize("extension,tolerance", [("dcd", 1e-4), ("xtc", 0.011)])
def test_real_md_roundtrip(extension, tolerance, tmp_path, structure_path):
    mda = pytest.importorskip("MDAnalysis")
    st = gemmi.read_structure(str(structure_path))
    st.remove_waters()
    topology = tmp_path / "topology.pdb"
    st.write_pdb(str(topology))
    universe = mda.Universe(str(topology))
    group = universe.select_atoms("protein")
    original = group.positions.copy()
    path = tmp_path / f"traj.{extension}"
    with mda.Writer(str(path), n_atoms=len(group)) as writer:
        for i in range(5):
            group.positions = original + [i, i * 0.5, 0]
            writer.write(group)
    trajectory = Trajectory.from_mdanalysis(topology, path, stride=2)
    assert len(trajectory) == 3
    assert trajectory.n_atoms == 602
    np.testing.assert_allclose(trajectory.frame(2), original + [4, 2, 0], atol=tolerance)
    np.testing.assert_allclose(trajectory.aligned().frame(2), original, atol=tolerance * 2)
    protein = Protein.from_trajectory(trajectory)
    assert len(protein.topology.chains) == 1
