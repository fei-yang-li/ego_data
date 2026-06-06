"""Small, dependency-free SE(3) / rotation helpers.

Only numpy is used so the pipeline stays lightweight. Rotations are converted to
axis-angle ("rotation vector") for action representations because that is the
most common rotation parameterization in VLA action spaces.
"""

from __future__ import annotations

import numpy as np


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 SE(3) matrix from a 3x3 rotation and a 3-vector."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) transform (cheaper / more stable than ``np.linalg.inv``)."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def relative_transform(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Transform of frame ``b`` expressed relative to frame ``a``: ``inv(T_a) @ T_b``."""
    return se3_inverse(T_a) @ np.asarray(T_b, dtype=np.float64).reshape(4, 4)


def rotation_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a 3-vector axis-angle (rad)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-6:
        # Near 180 deg: recover axis from the symmetric part for numerical stability.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Fix signs using off-diagonal terms.
        if axis[0] > 0:
            axis[1] = np.copysign(axis[1], A[0, 1])
            axis[2] = np.copysign(axis[2], A[0, 2])
        return axis / (np.linalg.norm(axis) + 1e-12) * theta
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    axis = axis / (2.0 * np.sin(theta))
    return axis * theta


def pose_delta(T_from: np.ndarray, T_to: np.ndarray) -> np.ndarray:
    """6-DoF delta from one pose to the next, expressed in ``T_from``'s frame.

    Returns ``[dx, dy, dz, rx, ry, rz]`` where the translation is the position
    change in the source frame and the rotation is axis-angle.
    """
    rel = relative_transform(T_from, T_to)
    trans = rel[:3, 3]
    rot = rotation_to_axis_angle(rel[:3, :3])
    return np.concatenate([trans, rot])
