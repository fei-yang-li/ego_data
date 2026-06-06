"""Derive action / state vectors from canonical episodes.

Two action spaces are produced, matching the two downstream consumers:

* **End-effector action** (for VLA): per step ``[dx, dy, dz, rx, ry, rz, grip]``
  where the first six are the delta pose of the acting hand's wrist expressed in
  the previous step's wrist frame (axis-angle rotation), and ``grip`` is the
  absolute gripper aperture derived from thumb-index distance.
* **Camera action** (for world models): per step ``[dx, dy, dz, rx, ry, rz]``,
  the delta pose of the head/camera in the previous camera frame — i.e. the
  camera trajectory that the generated video must follow.

Both are length ``T-1`` (one action per transition). A matching
``proprioceptive_state`` (length ``T``) is also provided for VLA observations.
"""

from __future__ import annotations

import numpy as np

from ego_pipeline.geometry import (
    make_se3,
    pose_delta,
    rotation_to_axis_angle,
)
from ego_pipeline.schema import EgoEpisode, HandPose


def wrist_pose(hand: HandPose) -> np.ndarray:
    """Estimate a 4x4 wrist pose (orientation + position) from keypoints.

    The local frame is built from the palm geometry: x points wrist->index MCP,
    the index->pinky MCP direction seeds y, and z = x cross y (palm normal).
    """
    k = hand.keypoints
    wrist = k[0]
    x = k[5] - k[0]  # wrist -> index MCP
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        return make_se3(np.eye(3), wrist)
    x = x / nx
    span = k[17] - k[5]  # index MCP -> pinky MCP
    z = np.cross(x, span)
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        return make_se3(np.eye(3), wrist)
    z = z / nz
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return make_se3(R, wrist)


def _hand_series(episode: EgoEpisode, hand: str) -> list[HandPose | None]:
    attr = "right_hand" if hand == "right" else "left_hand"
    return [getattr(f, attr) for f in episode.frames]


def end_effector_actions(episode: EgoEpisode, hand: str = "right") -> np.ndarray | None:
    """``(T-1, 7)`` end-effector delta-pose + gripper actions, or ``None``."""
    hands = _hand_series(episode, hand)
    if all(h is None for h in hands) or len(hands) < 2:
        return None

    poses = [wrist_pose(h) if h is not None else None for h in hands]
    grips = [h.openness if h is not None else np.nan for h in hands]

    actions = np.zeros((len(hands) - 1, 7), dtype=np.float64)
    last_valid = next((p for p in poses if p is not None), np.eye(4))
    for i in range(1, len(hands)):
        p_prev = poses[i - 1] if poses[i - 1] is not None else last_valid
        p_cur = poses[i] if poses[i] is not None else p_prev
        if poses[i] is not None:
            last_valid = poses[i]
        actions[i - 1, :6] = pose_delta(p_prev, p_cur)
        actions[i - 1, 6] = grips[i] if not np.isnan(grips[i]) else 0.0
    return actions


def proprioceptive_state(episode: EgoEpisode, hand: str = "right") -> np.ndarray | None:
    """``(T, 7)`` absolute wrist position + axis-angle orientation + gripper."""
    hands = _hand_series(episode, hand)
    if all(h is None for h in hands):
        return None
    state = np.zeros((len(hands), 7), dtype=np.float64)
    last = np.eye(4)
    for i, h in enumerate(hands):
        if h is not None:
            last = wrist_pose(h)
            grip = h.openness
        else:
            grip = 0.0
        state[i, :3] = last[:3, 3]
        state[i, 3:6] = rotation_to_axis_angle(last[:3, :3])
        state[i, 6] = grip
    return state


def camera_actions(episode: EgoEpisode) -> np.ndarray | None:
    """``(T-1, 6)`` head/camera delta-pose actions, or ``None`` if no poses."""
    poses = [f.head_pose for f in episode.frames]
    if all(p is None for p in poses) or len(poses) < 2:
        return None
    actions = np.zeros((len(poses) - 1, 6), dtype=np.float64)
    last_valid = next((p for p in poses if p is not None), np.eye(4))
    for i in range(1, len(poses)):
        p_prev = poses[i - 1] if poses[i - 1] is not None else last_valid
        p_cur = poses[i] if poses[i] is not None else p_prev
        if poses[i] is not None:
            last_valid = poses[i]
        actions[i - 1] = pose_delta(p_prev, p_cur)
    return actions


def compute_action_stats(actions_list: list[np.ndarray]) -> dict[str, list[float]]:
    """Aggregate mean/std/min/max/q01/q99 over a list of ``(N, D)`` action arrays.

    These statistics are what VLA training pipelines use to normalize actions.
    """
    if not actions_list:
        return {}
    stacked = np.concatenate([a for a in actions_list if a is not None and len(a)], axis=0)
    return {
        "mean": stacked.mean(0).tolist(),
        "std": (stacked.std(0) + 1e-8).tolist(),
        "min": stacked.min(0).tolist(),
        "max": stacked.max(0).tolist(),
        "q01": np.quantile(stacked, 0.01, axis=0).tolist(),
        "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
        "num_transitions": int(len(stacked)),
    }
