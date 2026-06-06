"""Generate small synthetic egocentric datasets for testing and demos.

Produces an EgoVerse-style on-disk layout (see
:mod:`ego_pipeline.readers.egoverse`) with plausible head trajectories, two
moving hands (21 keypoints each), gaze, camera intrinsics, a language
instruction, and optionally rendered RGB frames. No external datasets required.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ego_pipeline.schema import CameraIntrinsics, HAND_EDGES

_INSTRUCTIONS = [
    "pick up the mug and place it on the shelf",
    "open the drawer and take out the spoon",
    "pour water from the bottle into the glass",
    "fold the towel and put it in the basket",
    "wipe the table with the cloth",
]

# Coarse manipulation phases used as the clip's narration / semantic annotation.
_PHASES = ["approach object", "reach and align", "grasp", "manipulate", "retract hand"]


def _narration_segments(num_frames: int, fps: float) -> list[dict]:
    """Split a clip into evenly spaced narration phases."""
    duration = (num_frames - 1) / fps if num_frames > 1 else 1.0 / fps
    n = min(len(_PHASES), max(1, num_frames))
    bounds = [duration * i / n for i in range(n + 1)]
    return [
        {"start": round(bounds[i], 3), "end": round(bounds[i + 1], 3), "text": _PHASES[i]}
        for i in range(n)
    ]


def _base_hand(offset: np.ndarray) -> np.ndarray:
    """A canonical open hand (21x3) around ``offset`` in the camera frame."""
    pts = np.zeros((21, 3), dtype=np.float64)
    pts[0] = [0.0, 0.0, 0.0]  # wrist
    finger_roots = {1: 0.02, 5: 0.03, 9: 0.03, 13: 0.025, 17: 0.02}
    spread = {1: -0.03, 5: -0.012, 9: 0.0, 13: 0.012, 17: 0.025}
    for root, length in finger_roots.items():
        for j in range(4):
            idx = root + j
            pts[idx] = [spread[root] * (j + 1), 0.0, length * (j + 1)]
    return pts + offset


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width=640, height=480, fx=480.0, fy=480.0, cx=320.0, cy=240.0)


def generate_episode_arrays(
    num_frames: int, fps: float, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(num_frames) / fps
    timestamps = t.astype(np.float64)

    # Smooth head trajectory: gentle forward motion + sinusoidal sway/yaw.
    head_pose = np.zeros((num_frames, 4, 4), dtype=np.float64)
    for i in range(num_frames):
        yaw = 0.25 * np.sin(2 * np.pi * 0.2 * t[i])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pos = np.array([
            0.1 * np.sin(2 * np.pi * 0.1 * t[i]),
            0.02 * np.sin(2 * np.pi * 0.5 * t[i]),
            0.15 * t[i],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        head_pose[i] = T

    # Hands: defined in camera frame then pushed to world via head pose, so they
    # stay roughly in front of the camera as it moves (realistic for ego data).
    left = np.zeros((num_frames, 21, 3))
    right = np.zeros((num_frames, 21, 3))
    gaze = np.zeros((num_frames, 3))
    for i in range(num_frames):
        reach = 0.05 * np.sin(2 * np.pi * 0.3 * t[i])
        l_cam = _base_hand(np.array([-0.12, -0.08, 0.35 + reach]))
        r_cam = _base_hand(np.array([0.12, -0.08, 0.35 - reach]))
        # Animate gripper aperture on the right hand (thumb-index close/open).
        grip = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t[i])
        r_cam[4] = r_cam[8] + (r_cam[4] - r_cam[8]) * grip  # move thumb tip
        T = head_pose[i]
        left[i] = _apply(T, l_cam)
        right[i] = _apply(T, r_cam)
        gaze[i] = _apply(T, np.array([[0.0, 0.0, 1.0]]))[0]

    left += rng.normal(0, 0.001, left.shape)
    right += rng.normal(0, 0.001, right.shape)
    return {
        "timestamps": timestamps,
        "head_pose": head_pose,
        "left_hand": left,
        "right_hand": right,
        "gaze": gaze,
    }


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (T @ h.T).T[:, :3]


def _render_frame(arrays: dict[str, np.ndarray], i: int, intr: CameraIntrinsics, path: str) -> None:
    from PIL import Image, ImageDraw

    from ego_pipeline.geometry import se3_inverse

    img = Image.new("RGB", (intr.width, intr.height), (30, 32, 38))
    draw = ImageDraw.Draw(img)
    cam_from_world = se3_inverse(arrays["head_pose"][i])
    for side, color in (("left_hand", (60, 160, 255)), ("right_hand", (255, 120, 60))):
        pts_w = arrays[side][i]
        h = np.concatenate([pts_w, np.ones((pts_w.shape[0], 1))], axis=1)
        pts_c = (cam_from_world @ h.T).T[:, :3]
        if np.any(pts_c[:, 2] <= 0):
            continue
        px = intr.project(pts_c)
        for a, b in HAND_EDGES:
            draw.line([tuple(px[a]), tuple(px[b])], fill=color, width=2)
        for p in px:
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)
    img.save(path)


def generate_dataset(
    out_dir: str,
    num_episodes: int = 3,
    num_frames: int = 30,
    fps: float = 30.0,
    render_rgb: bool = True,
    seed: int = 0,
) -> str:
    """Write a synthetic EgoVerse-style dataset to ``out_dir``; return the path."""
    os.makedirs(out_dir, exist_ok=True)
    intr = _intrinsics()
    for e in range(num_episodes):
        ep_id = f"episode_{e:04d}"
        ep_dir = os.path.join(out_dir, ep_id)
        os.makedirs(ep_dir, exist_ok=True)
        arrays = generate_episode_arrays(num_frames, fps, seed=seed + e)
        np.savez(os.path.join(ep_dir, "poses.npz"), **arrays)
        meta = {
            "language_instruction": _INSTRUCTIONS[e % len(_INSTRUCTIONS)],
            "fps": fps,
            "source": "synthetic",
            "intrinsics": intr.to_dict(),
            "narrations": _narration_segments(num_frames, fps),
            "metadata": {"synthetic_seed": seed + e},
        }
        with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        if render_rgb:
            rgb_dir = os.path.join(ep_dir, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            for i in range(num_frames):
                _render_frame(
                    arrays, i, intr, os.path.join(rgb_dir, f"frame_{i:06d}.jpg")
                )
    return out_dir
