"""Small, explicit right-handed math helpers. Distances are in ångströms.

Proteins, the camera, the renderers and the geometry builders share these helpers,
and with them these conventions:

- Distances are in ångströms. Rotation and camera angles are in radians; chemical
  angles elsewhere in the package (torsions, bond angles) are in degrees.
- Frames are right-handed. The camera looks down its own −z axis with +y up, as in OpenGL.
- 3 × 3 and 4 × 4 matrices act on column vectors (``M @ v``). Point arrays are (n, 3)
  rows, so code applies a matrix to them as ``points @ M[:3, :3].T + M[:3, 3]``.
  WGSL stores matrices column by column, so the renderers upload ``M.T.ravel()``.
- kabsch() is the exception: its rotation acts on row vectors (``points @ R``).
"""

import numpy as np


def normalize(v):
    """Unit vectors along the last axis, as float64. A zero vector stays zero instead of NaN."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def rotation(angle, axis=(0, 1, 0)):
    """3 × 3 matrix that turns column vectors by ``angle`` radians about ``axis``.

    A positive angle turns counterclockwise when the axis points at the viewer (right-hand rule).
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or np.linalg.norm(axis) < 1e-10:
        raise ValueError("Rotation axis must be a nonzero 3-vector")
    # Rodrigues' formula in matrix form, R = c I + s [k]× + t k kᵀ, for the unit axis
    # k = (x, y, z), with c = cos θ, s = sin θ and t = 1 − cos θ.
    x, y, z = normalize(axis)
    c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def kabsch(moving, reference):
    """Best-fit rigid motion between two (n, 3) point sets: ``moving @ R + t ≈ reference``.

    Every superposition in the package uses it. R is a proper rotation (det +1), so a
    mirror image is never accepted as a fit, and points are row vectors. Code that works
    with column vectors (``R.T @ x``) transposes R.
    """
    a, b = np.asarray(moving, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    ac, bc = a.mean(0), b.mean(0)
    # Kabsch: the SVD of the covariance of the centered points gives the rotation.
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    # If u @ vt is a reflection, flip the axis of least variance to get a rotation instead.
    flip = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, flip]) @ vt
    return rotation, bc - ac @ rotation


def align_coordinates(moving, reference, indices=None):
    """Superpose moving on reference (Kabsch, no mirroring) and return a new float32 array.

    ``indices`` limits the fit to those atoms, but the whole of ``moving`` is moved.
    """
    # Double precision avoids centroid cancellation in float32 coordinate files.
    moving, reference = np.asarray(moving, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Alignment requires matching (atoms, 3) arrays")
    a, b = (moving, reference) if indices is None else (moving[indices], reference[indices])
    if len(a) < 3:
        raise ValueError("Alignment needs at least three matched atoms")
    rotation, translation = kabsch(a, b)
    return np.asarray(moving @ rotation + translation, dtype=np.float32)


def perpendiculars(direction):
    """Two unit vectors that make a right-handed frame with the unit vector ``direction``."""
    # Cross with whichever world axis is far from parallel, so the result never degenerates.
    ref = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(direction, u)


def look_at(eye, target):
    """World-to-view 4 × 4 matrix for a camera at ``eye`` looking at ``target``, world +y up."""
    forward = normalize(np.asarray(target) - eye)
    up = np.array([0.0, 1.0, 0.0])
    # Looking almost straight up or down, +y is parallel to the view direction and the
    # cross product below vanishes, so +z serves as up instead.
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    right = normalize(np.cross(forward, up))
    up = np.cross(right, forward)
    # The rows are the camera axes in world space. The camera looks down its −z axis,
    # and the translation moves the eye to the origin.
    view = np.eye(4)
    view[:3, :3] = np.stack((right, up, -forward))
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov, aspect, near, far):
    """Projection matrix for a vertical field of view ``fov`` in radians and width/height ``aspect``.

    View-space depth −near maps to 0 and −far to 1, the WebGPU clip range.
    """
    f = 1 / np.tan(fov / 2)
    # WebGPU's NDC depth is [0, 1], unlike OpenGL's [−1, 1].
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, far / (near - far), far * near / (near - far)],
            [0, 0, -1, 0],
        ]
    )


def color(value):
    """An RGB float32 array in [0, 1] from ``"#RRGGBB"`` or from three numbers in [0, 1]."""
    if isinstance(value, str):
        s = value.lstrip("#")
        if len(s) != 6:
            raise ValueError("Colors must use #RRGGBB or three values in [0, 1]")
        value = [int(s[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (3,) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError("Color must contain three finite values in [0, 1]")
    return a
