"""Small, explicit right-handed math helpers. Distances are in ångströms."""

import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def rotation(angle, axis=(0, 1, 0)):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or np.linalg.norm(axis) < 1e-10:
        raise ValueError("Rotation axis must be a nonzero 3-vector")
    x, y, z = normalize(axis)
    c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def align_coordinates(moving, reference, indices=None):
    """Proper Kabsch alignment; preserves chirality and returns a new array."""
    # Double precision avoids centroid cancellation in float32 coordinate files.
    moving, reference = np.asarray(moving, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Alignment requires matching (atoms, 3) arrays")
    a, b = (moving, reference) if indices is None else (moving[indices], reference[indices])
    if len(a) < 3:
        raise ValueError("Alignment needs at least three matched atoms")
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    sign = np.diag([1, 1, np.linalg.det(u @ vt)])
    return np.asarray((moving - ac) @ (u @ sign @ vt) + bc, dtype=np.float32)


def look_at(eye, target):
    forward = normalize(np.asarray(target) - eye)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    right = normalize(np.cross(forward, up))
    up = np.cross(right, forward)
    view = np.eye(4)
    view[:3, :3] = np.stack((right, up, -forward))
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov, aspect, near, far):
    f = 1 / np.tan(fov / 2)
    # WebGPU's NDC depth is [0, 1].
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, far / (near - far), far * near / (near - far)],
            [0, 0, -1, 0],
        ]
    )


def color(value):
    if isinstance(value, str):
        s = value.lstrip("#")
        if len(s) != 6:
            raise ValueError("Colors must use #RRGGBB or three values in [0, 1]")
        value = [int(s[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (3,) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError("Color must contain three finite values in [0, 1]")
    return a
