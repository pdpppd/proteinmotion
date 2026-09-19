from types import SimpleNamespace

import pytest

from proteinmotion import _gpu
from proteinmotion.eevee import find_blender


@pytest.fixture
def adapters(monkeypatch):
    for name in (
        "PROTEINMOTION_GPU_BACKEND",
        "PROTEINMOTION_GPU_ADAPTER",
        "WGPU_BACKEND_TYPE",
        "WGPUPY_WGPU_ADAPTER_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_gpu.sys, "platform", "win32")
    infos = [
        ("IntegratedGPU", "Vulkan", "Intel Graphics"),
        ("CPU", "D3D12", "Microsoft Basic Render Driver"),
        ("DiscreteGPU", "D3D12", "NVIDIA RTX"),
        ("DiscreteGPU", "Vulkan", "NVIDIA RTX"),
        ("IntegratedGPU", "Metal", "Apple M3"),
    ]
    values = [SimpleNamespace(info=dict(adapter_type=t, backend_type=b, device=d)) for t, b, d in infos]
    monkeypatch.setattr(_gpu.wgpu.gpu, "enumerate_adapters_sync", lambda: values)
    return values


def test_discrete_vulkan_preferred_over_integrated_and_software(adapters):
    assert _gpu.select_adapter() is adapters[3]
    assert _gpu.select_adapter(backend="dx12", adapter_name="nvidia") is adapters[2]
    assert _gpu.select_adapter(adapter_name="Intel") is adapters[0]
    with pytest.raises(RuntimeError, match="No native GPU"):
        _gpu.select_adapter(adapter_name="Microsoft")
    with pytest.raises(RuntimeError, match="No native GPU"):
        _gpu.select_adapter(adapter_name="missing")
    with pytest.raises(ValueError, match="Unknown GPU backend"):
        _gpu.select_adapter(backend="CUDA")


def test_adapter_overrides_and_macos_compatibility(adapters, monkeypatch):
    monkeypatch.setenv("PROTEINMOTION_GPU_BACKEND", "D3D12")
    assert _gpu.select_adapter() is adapters[2]
    assert _gpu.select_adapter(backend="Vulkan") is adapters[3]
    monkeypatch.delenv("PROTEINMOTION_GPU_BACKEND")
    monkeypatch.setenv("WGPU_BACKEND_TYPE", "D3D12")
    assert _gpu.select_adapter() is adapters[2]
    monkeypatch.delenv("WGPU_BACKEND_TYPE")
    monkeypatch.setenv("PROTEINMOTION_GPU_ADAPTER", "Intel")
    assert _gpu.select_adapter() is adapters[0]
    monkeypatch.delenv("PROTEINMOTION_GPU_ADAPTER")
    monkeypatch.setattr(_gpu.sys, "platform", "darwin")
    assert _gpu.select_adapter() is adapters[4]
    assert _gpu.select_adapter(require_metal=False) is adapters[3]
    with pytest.raises(ValueError, match="conflicts"):
        _gpu.select_adapter(require_metal=True, backend="Vulkan")


def test_windows_blender_discovery_prefers_newest_and_honors_override(tmp_path, monkeypatch):
    from proteinmotion import eevee

    monkeypatch.setattr(eevee.sys, "platform", "win32")
    monkeypatch.setattr(eevee.shutil, "which", lambda _: None)
    monkeypatch.delenv("PROTEINMOTION_BLENDER", raising=False)
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setattr(eevee.os, "access", lambda *args: True)
    paths = []
    for version in ("4.5", "4.9", "4.10"):
        path = tmp_path / "Blender Foundation" / f"Blender {version}" / "blender.exe"
        path.parent.mkdir(parents=True)
        path.touch()
        paths.append(path)
    assert find_blender() == str(paths[-1].resolve())
    assert find_blender(paths[0]) == str(paths[0].resolve())
    monkeypatch.setenv("PROTEINMOTION_BLENDER", str(paths[1]))
    assert find_blender() == str(paths[1].resolve())
    with pytest.raises(RuntimeError, match="Install Blender"):
        find_blender(tmp_path / "missing.exe")
