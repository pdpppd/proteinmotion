"""Studio presets, deterministic film effects, and the actual GPU export path."""

import sys

import av
import numpy as np
import pytest

from proteinmotion import EEVEEOptions, FadeIn, ProteinScene, StudioLook, Text
from proteinmotion.cli import main
from proteinmotion.studio import StudioRenderer


def test_effect_overrides_and_default_compatibility():
    default = StudioLook()
    assert (default.lighting, default.material, default.bloom, default.grain, default.halation) == (
        "studio",
        "glossy",
        0.35,
        0,
        0,
    )
    film = StudioLook(
        effects="film", grain="off", bloom="strong", bloom_threshold=0.9, halation=0.0, grain_size=2
    )
    assert film.grain == film.halation == 0
    assert film.bloom == 0.65 and film.bloom_threshold == 0.9 and film.grain_size == 2
    assert StudioLook(lighting="soft", material="rough", effects="clean").bloom == 0


@pytest.mark.parametrize(
    "settings,field",
    [
        ({"lighting": "missing"}, "lighting"),
        ({"material": "missing"}, "material"),
        ({"surface_mode": "missing"}, "surface_mode"),
        ({"cartoon_style": "missing"}, "cartoon_style"),
        ({"effects": "missing"}, "effects"),
        ({"grain": "missing"}, "grain"),
        ({"bloom": "missing"}, "bloom"),
        ({"halation": "missing"}, "halation"),
        ({"grain": -0.1}, "grain"),
        ({"grain": float("nan")}, "grain"),
        ({"bloom": float("inf")}, "bloom"),
        ({"halation_radius": 0}, "halation_radius"),
        ({"grain_seed": 2**32}, "grain_seed"),
        ({"grain_seed": 1.5}, "grain_seed"),
        ({"roughness": 0}, "roughness"),
        ({"key_direction": (0, 0, 0)}, "key_direction"),
        ({"halation_color": (1, 0)}, "halation_color"),
        ({"knee": 1}, "knee"),
    ],
)
def test_invalid_looks_fail_before_gpu_allocation(settings, field):
    with pytest.raises(ValueError, match=field):
        StudioLook(**settings)


def test_scene_rejects_ignored_settings(tmp_path):
    scene = ProteinScene()
    for renderer in ("native", "eevee"):
        with pytest.raises(ValueError, match="StudioLook settings"):
            scene.render_frame(renderer=renderer, look=StudioLook())
        with pytest.raises(ValueError, match="StudioLook settings"):
            scene.render(tmp_path / "unused.mp4", renderer=renderer, look=StudioLook())
    with pytest.raises(ValueError, match="EEVEE settings"):
        scene.render_frame(renderer="studio", eevee=EEVEEOptions())
    assert not list(tmp_path.iterdir())


def test_cli_studio_options(monkeypatch, tmp_path, capsys):
    from proteinmotion import cli

    class Capture:
        def render_frame(self, time, **kwargs):
            assert time == 1.0 and kwargs["renderer"] == "studio"
            look = kwargs["look"]
            assert look.lighting == "soft" and look.material == "rough"
            assert look.grain == 0.025 and look.bloom == 0.25 and look.halation == 0

    monkeypatch.setattr(cli, "load_scene", lambda *a, **k: Capture())
    args = [
        "proteinmotion",
        "still",
        "unused.py",
        "Movie",
        "--time",
        "1",
        "-o",
        str(tmp_path / "x.png"),
        "--renderer",
        "studio",
        "--lighting",
        "soft",
        "--material",
        "rough",
        "--effects",
        "film",
        "--grain",
        "0.025",
        "--bloom",
        "soft",
        "--halation",
        "off",
    ]
    monkeypatch.setattr(sys, "argv", args)
    main()
    args[args.index("studio")] = "native"
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and "require --renderer studio" in capsys.readouterr().err


@pytest.fixture
def gpu():
    import wgpu

    if not wgpu.gpu.enumerate_adapters_sync():
        pytest.skip("No native GPU adapter available")


@pytest.mark.gpu
@pytest.mark.parametrize("msaa", [1, 4])
def test_grain_is_monochrome_luminance_based_and_seekable(gpu, msaa):
    scene = ProteinScene(width=256, height=128, msaa=msaa, background="#808080")
    scene.wait(2)
    look = StudioLook(effects="clean", grain=0.08, grain_seed=127)
    with StudioRenderer(256, 128, msaa=msaa, look=look) as renderer:
        mid = scene.render_frame(0.5, renderer=renderer)
        other_time = scene.render_frame(1.0, renderer=renderer)
        np.testing.assert_array_equal(scene.render_frame(0.5, renderer=renderer), mid)
        assert np.abs(mid.astype(int) - other_time).sum() > 20000
        np.testing.assert_array_equal(mid[:, :, 0], mid[:, :, 1])
        np.testing.assert_array_equal(mid[:, :, 1], mid[:, :, 2])
        scene.background[:] = 0.015
        dark = scene.render_frame(0.5, renderer=renderer)
        assert mid[:, :, 0].std() > dark[:, :, 0].std() * 2
        # Scene times outside the timeline clamp identically, including grain.
        np.testing.assert_array_equal(
            scene.render_frame(-1, renderer=renderer), scene.render_frame(0, renderer=renderer)
        )
        # A new seed gives a different texture, but disabling grain restores a flat image.
        renderer.look.grain_seed = 2**32 - 1
        changed_seed = scene.render_frame(0.5, renderer=renderer)
        assert np.any(changed_seed != dark)
        renderer.look.grain = 0
        flat = scene.render_frame(0.5, renderer=renderer)
        assert flat[:, :, 0].std() == 0


@pytest.mark.gpu
def test_effects_keep_overlay_text_sharp(gpu):
    scene = ProteinScene(width=512, height=256, background="#404040")
    scene.add(Text("Sharp text", position=(0.12, 0.3), font_size=100, color="#ffffff"))
    scene.wait(1)
    clean = scene.render_frame(renderer="studio", look=StudioLook(effects="clean"))
    film = scene.render_frame(renderer="studio", look=StudioLook(effects="film", grain=0.15))
    white = (clean[:, :, :3] == 255).all(2)
    assert white.sum() > 50
    np.testing.assert_array_equal(film[white], clean[white])
    assert np.abs(clean.astype(int) - film).sum() > 20000


@pytest.mark.gpu
def test_lighting_and_material_presets_are_distinct(gpu, protein):
    scene = ProteinScene(width=320, height=240, msaa=1)
    scene.add(protein.surface(resolution=0.9))
    scene.camera.frame(protein)
    images = {}
    for light, material in [
        ("studio", "glossy"),
        ("soft", "glossy"),
        ("dramatic", "glossy"),
        ("flat", "glossy"),
        ("studio", "rough"),
        ("studio", "medium"),
    ]:
        look = StudioLook(lighting=light, material=material, effects="clean")
        img = scene.render_frame(renderer="studio", look=look)
        assert img.shape == (240, 320, 4) and (img[:, :, 3] == 255).all()
        for prior in images.values():
            assert np.abs(img.astype(int) - prior).sum() > 10000
        images[light, material] = img


@pytest.mark.gpu
def test_bloom_and_halation_are_independent(gpu, protein):
    scene = ProteinScene(width=384, height=256, msaa=4)
    scene.add(protein.surface(resolution=0.8).set_color("#ffffff"))
    scene.camera.frame(protein)
    with StudioRenderer(384, 256, msaa=4, look=StudioLook(effects="clean")) as renderer:
        clean = scene.render_frame(renderer=renderer).astype(int)
        renderer.look.halation, renderer.look.halation_threshold, renderer.look.halation_radius = 1, 0.4, 12
        halo = scene.render_frame(renderer=renderer).astype(int)
        delta = (halo - clean)[:, :, :3]
        assert delta.sum() > 1000
        assert delta[:, :, 0].sum() > 2 * delta[:, :, 2].sum()  # Warm fringe, not neutral bloom.
        np.testing.assert_array_equal(halo[:12, :12], clean[:12, :12])
        renderer.look.halation, renderer.look.bloom, renderer.look.bloom_threshold = 0, 0.8, 0.5
        bloom = scene.render_frame(renderer=renderer).astype(int)
        assert np.abs(bloom - clean).sum() > 1000
        assert np.abs(bloom - halo).sum() > 1000
        renderer.look.bloom = 0
        np.testing.assert_array_equal(scene.render_frame(renderer=renderer), clean)


@pytest.mark.gpu
def test_studio_movie_uses_grain_time_through_fades(gpu, protein, tmp_path):
    scene = ProteinScene(width=192, height=128, fps=10, background="#505050")
    scene.camera.frame(protein)
    scene.play(FadeIn(protein), run_time=0.4)
    scene.wait(0.4)
    look = StudioLook(effects="film")
    result = scene.render(
        tmp_path / "film.mp4", renderer="studio", look=look, codec="libx264", progress=False
    )
    assert result["frames"] == 8
    with av.open(tmp_path / "film.mp4") as video:
        frames = [f.to_ndarray(format="rgb24") for f in video.decode(video=0)]
    assert len(frames) == 8
    assert np.abs(frames[-1].astype(int) - frames[-2]).sum() > 1000  # Same geometry, changing grain.
