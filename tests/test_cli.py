"""Command-line options for render and still. No test here launches Blender."""

import sys

import pytest

from proteinmotion import EEVEEOptions, ProteinScene, cli

SCENES = {
    "native": """
from proteinmotion import ProteinScene


class Empty(ProteinScene):
    def construct(self):
        self.wait(0.1)
""",
    # A scene class that chooses EEVEE and its options itself.
    "eevee": """
from proteinmotion import EEVEEOptions, ProteinScene


class Focus(ProteinScene):
    renderer = "eevee"
    eevee = EEVEEOptions(samples=128, max_blur=12)

    def construct(self):
        self.wait(0.1)
""",
}


def run_cli(monkeypatch, tmp_path, scene, command, *flags):
    """Run ``proteinmotion <command>`` on the scene file ``SCENES[scene]``; return the output path."""
    path = tmp_path / "scene.py"
    path.write_text(SCENES[scene])
    output = tmp_path / ("frame.png" if command == "still" else "movie.mp4")
    monkeypatch.setattr(sys, "argv", ["proteinmotion", command, str(path), "-o", str(output), *flags])
    cli.main()
    return output


@pytest.fixture
def calls(monkeypatch):
    """Keyword arguments of each render() or render_frame() call, which render nothing."""
    recorded = []

    def record(self, *args, **kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(ProteinScene, "render", record)
    monkeypatch.setattr(ProteinScene, "render_frame", record)
    return recorded


@pytest.mark.parametrize("command", ["render", "still"])
@pytest.mark.parametrize(
    "scene,flags", [("native", ()), ("native", ("--renderer", "studio")), ("eevee", ("--renderer", "native"))]
)
def test_eevee_flags_are_not_checked_for_other_renderers(monkeypatch, tmp_path, calls, command, scene, flags):
    # --samples 0 used to raise a ValueError traceback before the native renderer started.
    run_cli(monkeypatch, tmp_path, scene, command, "--samples", "0", *flags)
    assert calls[-1]["eevee"] is None


@pytest.mark.parametrize("command", ["render", "still"])
def test_scene_that_chooses_eevee_gets_the_eevee_flags(monkeypatch, tmp_path, calls, command):
    # These flags used to be dropped when the scene class, not --renderer, chose EEVEE.
    flags = ("--samples", "16", "--blender", "/opt/blender", "--supersampling", "1.25")
    run_cli(monkeypatch, tmp_path, "eevee", command, *flags)
    assert calls[-1]["renderer"] is None
    # Flags replace the matching fields of the scene's options; max_blur is the scene's.
    expected = EEVEEOptions(blender="/opt/blender", samples=16, supersampling=1.25, max_blur=12)
    assert calls[-1]["eevee"] == expected


@pytest.mark.parametrize(
    "scene,expected", [("eevee", EEVEEOptions(samples=128, max_blur=12)), ("native", EEVEEOptions())]
)
def test_renderer_flag_keeps_the_scene_eevee_options(monkeypatch, tmp_path, calls, scene, expected):
    # --renderer eevee used to replace the scene's EEVEE options with the defaults.
    run_cli(monkeypatch, tmp_path, scene, "still", "--renderer", "eevee")
    assert calls[-1]["renderer"] == "eevee"
    assert calls[-1]["eevee"] == expected


@pytest.mark.parametrize(
    "scene,flags,message",
    [
        ("native", ("--renderer", "eevee", "--samples", "0"), "samples must be a positive integer"),
        ("eevee", ("--samples", "-3"), "samples must be a positive integer"),
        ("eevee", ("--supersampling", "0.5"), "supersampling must be at least 1"),
    ],
)
def test_bad_eevee_values_are_usage_errors(monkeypatch, tmp_path, calls, capsys, scene, flags, message):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, tmp_path, scene, "still", *flags)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not calls


@pytest.mark.gpu
def test_native_still_ignores_eevee_samples(monkeypatch, tmp_path):
    flags = ("--width", "64", "--height", "48", "--samples", "0")
    output = run_cli(monkeypatch, tmp_path, "native", "still", *flags)
    assert output.stat().st_size > 0
