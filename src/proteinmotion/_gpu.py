"""Native adapter selection shared by rendering and diagnostics."""

import os
import sys

import wgpu

BACKENDS = ("Vulkan", "D3D12", "Metal", "OpenGL")


def select_adapter(*, backend=None, adapter_name=None, require_metal=None):
    backend = backend or os.environ.get("PROTEINMOTION_GPU_BACKEND") or os.environ.get("WGPU_BACKEND_TYPE")
    adapter_name = (
        adapter_name
        or os.environ.get("PROTEINMOTION_GPU_ADAPTER")
        or os.environ.get("WGPUPY_WGPU_ADAPTER_NAME")
    )
    if backend and backend.lower() != "auto":
        names = {name.lower(): name for name in BACKENDS}
        names.update(dx12="D3D12", directx12="D3D12")
        if backend.lower() not in names:
            raise ValueError(f"Unknown GPU backend {backend!r}; choose auto, {', '.join(BACKENDS)}")
        backend = names[backend.lower()]
    else:
        backend = None
    if require_metal is True or (require_metal is None and sys.platform == "darwin" and backend is None):
        if backend not in (None, "Metal"):
            raise ValueError("require_metal=True conflicts with the selected GPU backend")
        backend = "Metal"

    candidates = [
        a
        for a in wgpu.gpu.enumerate_adapters_sync()
        if a.info.get("adapter_type") != "CPU"
        and (backend is None or a.info.get("backend_type") == backend)
        and (
            not adapter_name
            or adapter_name.casefold()
            in " ".join(str(a.info.get(k, "")) for k in ("vendor", "device", "description")).casefold()
        )
    ]
    if not candidates:
        raise RuntimeError(
            f"No native GPU matches backend={backend or 'auto'}, adapter={adapter_name or 'auto'}. "
            "Install the GPU vendor's current driver and run proteinmotion doctor for diagnostics."
        )
    # Prefer a discrete GPU over integrated graphics, then Vulkan on Windows/Linux.
    # Do not accidentally choose Windows' Microsoft Basic Render Driver (CPU).
    types = {"DiscreteGPU": 0, "IntegratedGPU": 1, "VirtualGPU": 2}
    backends = {"Vulkan": 0, "Metal": 0, "D3D12": 1, "OpenGL": 2}
    return min(
        candidates,
        key=lambda a: (types.get(a.info.get("adapter_type"), 3), backends.get(a.info.get("backend_type"), 3)),
    )
