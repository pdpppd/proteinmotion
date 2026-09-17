"""Lens/timeline/export checks run without Blender; smoke render is opt-in."""

import os

import numpy as np
import pytest

from proteinmotion import Deform, EEVEEOptions, FocusPull, ProteinScene, Rotate
from proteinmotion._eevee_geometry import MeshExporter
from proteinmotion.camera import Camera
from proteinmotion.eevee import find_blender


def test_focus_selectors_and_follow(protein):
    camera = Camera().set_focus(protein, chain="A", residues=5, atoms="CA")
    before = camera.focus_point
    protein.shift([4, 2, -1])
    np.testing.assert_allclose(camera.focus_point, before + [4, 2, -1])
    camera.set_focus(protein, residues=(10, 20), atoms="CA", follow=False)
    frozen = camera.focus_point
    protein.shift([3, 0, 0])
    np.testing.assert_array_equal(camera.focus_point, frozen)
    camera.set_focus(None)
    assert not camera.dof
    with pytest.raises(ValueError, match="no atoms"):
        camera.set_focus(protein, residues=50000)
    with pytest.raises(ValueError, match="fstop"):
        camera.set_focus(protein, fstop=0)
    with pytest.raises(ValueError, match="finite world"):
        camera.set_focus([0, np.nan, 1])


@pytest.mark.parametrize("reverse", [False, True])
def test_focus_pull_seek_with_camera_and_deformation(protein, reverse):
    class Movie(ProteinScene):
        def construct(self):
            self.add(protein)
            self.camera.frame(protein)
            self.camera.set_focus(protein, residues=5)
            first = protein.select(residues=5)
            last = protein.select(residues=(10, 20))
            self.start = self.camera.focus_point.copy()
            animations = [
                self.camera.animate.orbit(0.3),
                FocusPull(self.camera, last, fstop=4),
                Deform(protein, lambda xyz: xyz + [0, 2, 0]),
            ]
            self.play(*(animations[::-1] if reverse else animations), run_time=2)
            self.play(self.camera.animate.orbit(0.2), run_time=1)
            self.play(FocusPull(self.camera, first), run_time=1)
            self.play(Rotate(protein, 0.3), run_time=1)

    scene = Movie().build()
    results = {}
    for t in (0, 1, 2, 2.5, 3.5, 4, 5):
        scene.seek(t)
        results[t] = scene.camera.focus_point
        if t == 1:
            expected = 0.5 * (scene.start + protein.select(residues=(10, 20)).world_positions.mean(0))
            np.testing.assert_allclose(results[t], expected, atol=1e-6)
    for t in (5, 2.5, 1, 0, 3.5, 2, 4):
        scene.seek(t)
        np.testing.assert_allclose(scene.camera.focus_point, results[t], atol=1e-6)
    scene.seek(5)
    np.testing.assert_allclose(scene.camera.focus_point, protein.select(residues=5).world_positions.mean(0))


@pytest.mark.parametrize("representation", ["cartoon", "ribbon", "ball_and_stick", "surface"])
def test_mesh_export_geometry_styles_and_transform(protein, representation):
    getattr(protein, representation)()
    protein.select(residues=(10, 20)).set_opacity(0.17)
    protein.select(residues=5).set_color("#ff0000")
    exporter = MeshExporter()
    meshes = exporter.meshes(protein)
    assert meshes
    for mesh in meshes:
        assert all(np.isfinite(v).all() for v in mesh.values())
        assert len(mesh["faces"]) == len(mesh["opacity"])
        assert mesh["vertices"].shape == mesh["colors"].shape == mesh["normals"].shape
        assert mesh["faces"].min() >= 0 and mesh["faces"].max() < len(mesh["vertices"])
        assert np.all((mesh["opacity"] >= 0) & (mesh["opacity"] <= 1))
    assert any(np.any(np.isclose(m["opacity"], 0.17)) for m in meshes)
    protein.shift([1, 2, 3])
    moved = exporter.meshes(protein)
    for before, after in zip(meshes, moved):
        np.testing.assert_allclose(after["vertices"], before["vertices"] + [1, 2, 3], atol=3e-6)


def test_options_and_missing_blender(monkeypatch):
    for opts in (
        dict(samples=0),
        dict(samples=2.5),
        dict(supersampling=0.5),
        dict(timeout=np.nan),
        dict(max_opacity_layers=0),
    ):
        with pytest.raises(ValueError):
            EEVEEOptions(**opts)
    monkeypatch.setenv("PROTEINMOTION_BLENDER", "/missing/blender")
    with pytest.raises(RuntimeError, match="Install Blender"):
        find_blender()
    with pytest.raises(RuntimeError):
        find_blender("/missing/explicit-blender")


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("PROTEINMOTION_TEST_EEVEE"), reason="Set PROTEINMOTION_TEST_EEVEE=1")
def test_eevee_real_frames(protein):
    from proteinmotion import Text
    from proteinmotion.eevee import EEVEE

    camera = Camera().frame(protein)
    camera.set_focus(protein, residues=(10, 20))
    protein.select(residues=(21, 76)).set_opacity(0.06)
    with EEVEE(160, 90, options=EEVEEOptions(samples=16, supersampling=1)) as renderer:
        first = renderer.render([protein], camera, [0.03, 0.04, 0.07])
        assert first[:, :, :3].max() > 100
        second = renderer.render([protein], camera, [0.03, 0.04, 0.07])
        assert first.shape == (90, 160, 4)
        np.testing.assert_array_equal(first, second)
        labelled = renderer.render([protein, Text("Focus")], camera, [0.03, 0.04, 0.07])
        assert np.any(first != labelled)
        camera.set_focus(protein, residues=(60, 70), fstop=1.4)
        changed = renderer.render([protein], camera, [0.03, 0.04, 0.07])
        assert np.any(changed != first)
        # Empty scenes and fully faded objects produce valid backgrounds too.
        empty = renderer.render([], camera, [0.03, 0.04, 0.07])
        assert empty.shape == first.shape
        # The Blender view and native overlay projection must agree, even when
        # orbiting. This catches up-axis, handedness and aspect-ratio mistakes.
        from proteinmotion.annotations import project_region

        region = protein.select(residues=5)
        marker = region.highlight(style="sphere", color="#ff0000", opacity=1)
        camera.orbit(0.35, 0.12).set_focus(region)
        frame = renderer.render([marker], camera, [0, 0, 0])
        mask = frame[:, :, 0].astype(float) > frame[:, :, 1] * 1.8 + 30
        y, x = np.nonzero(mask)
        assert len(x) > 3
        np.testing.assert_allclose(
            [x.mean() + 0.5, y.mean() + 0.5], project_region(region, camera, 160, 90), atol=2
        )
        # Representation and coordinate changes reuse the same Blender process.
        protein.ball_and_stick()
        atoms = renderer.render([protein], camera, [0.03, 0.04, 0.07])
        protein.set_positions(protein.positions + [1, 0, 0])
        moved = renderer.render([protein], camera, [0.03, 0.04, 0.07])
        assert np.any(atoms != moved)
        protein.surface(resolution=1)
        surface = renderer.render([protein], camera, [0.03, 0.04, 0.07])
        assert surface[:, :, :3].max() > 100
    with EEVEE(80, 60, options=EEVEEOptions(max_opacity_layers=1)) as renderer:
        with pytest.raises(RuntimeError, match="opacity layers"):
            renderer.render([protein], camera, [0, 0, 0])
