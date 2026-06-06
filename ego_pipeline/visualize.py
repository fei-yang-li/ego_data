"""Visualization utilities for inspecting canonical egocentric episodes.

Two complementary views are provided:

* :func:`plot_episode_summary` — a multi-panel matplotlib figure (modality
  availability, head trajectory, hand-openness/gripper signal, derived action
  magnitude) for a quick "does this episode look sane?" check.
* :func:`render_frame_overlay` — draws projected hand skeletons and the gaze
  point on top of an RGB frame (or a blank canvas when the image is missing),
  which catches calibration / coordinate-frame mistakes.

Matplotlib runs head-less (``Agg``) so this works on servers without a display.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ego_pipeline.geometry import se3_inverse  # noqa: E402
from ego_pipeline.schema import (  # noqa: E402
    HAND_EDGES,
    EgoEpisode,
    EgoFrame,
    HandPose,
    PoseConvention,
)

_HAND_COLORS = {"left": (60, 160, 255), "right": (255, 120, 60)}


def _hand_pixels(
    hand: HandPose, frame: EgoFrame
) -> np.ndarray | None:
    """Project a hand's keypoints into image pixels, or ``None`` if impossible."""
    if frame.intrinsics is None:
        return None
    kpts = hand.keypoints
    if hand.convention == PoseConvention.WORLD:
        if frame.head_pose is None:
            return None
        cam_from_world = se3_inverse(frame.head_pose)
        kh = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)
        kpts = (cam_from_world @ kh.T).T[:, :3]
    if np.any(kpts[:, 2] <= 0):
        return None
    return frame.intrinsics.project(kpts)


def render_frame_overlay(
    frame: EgoFrame, out_path: str, canvas_size: tuple[int, int] = (640, 480)
) -> str:
    """Render one frame with hand skeleton + gaze overlay to ``out_path``."""
    from PIL import Image, ImageDraw

    img = None
    if frame.rgb_path and os.path.isfile(frame.rgb_path):
        try:
            img = Image.open(frame.rgb_path).convert("RGB")
        except OSError:
            img = None
    if frame.rgb is not None and img is None:
        img = Image.fromarray(np.asarray(frame.rgb).astype("uint8"))
    if img is None:
        w, h = canvas_size
        if frame.intrinsics is not None:
            w, h = frame.intrinsics.width, frame.intrinsics.height
        img = Image.new("RGB", (w, h), (24, 24, 28))

    draw = ImageDraw.Draw(img)
    for hand in frame.hands():
        px = _hand_pixels(hand, frame)
        if px is None:
            continue
        color = _HAND_COLORS[hand.handedness.value]
        for a, b in HAND_EDGES:
            draw.line([tuple(px[a]), tuple(px[b])], fill=color, width=2)
        for p in px:
            draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)

    if frame.gaze is not None and frame.intrinsics is not None and len(frame.gaze) >= 3:
        gaze = frame.gaze
        if frame.head_pose is not None:
            gh = np.append(gaze[:3], 1.0)
            gaze = (se3_inverse(frame.head_pose) @ gh)[:3]
        if gaze[2] > 0:
            uv = frame.intrinsics.project(gaze.reshape(1, 3))[0]
            draw.ellipse(
                [uv[0] - 8, uv[1] - 8, uv[0] + 8, uv[1] + 8],
                outline=(0, 255, 0),
                width=3,
            )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


def plot_episode_summary(episode: EgoEpisode, out_path: str) -> str:
    """Render a multi-panel diagnostic figure for an episode."""
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"{episode.source} · {episode.episode_id} · "
        f"{len(episode)} frames · {episode.duration:.2f}s · {episode.fps:.1f} fps\n"
        f"instruction: {episode.language_instruction[:90]}",
        fontsize=11,
    )

    t = episode.timestamps
    t = t - t[0] if len(t) else t

    # Panel 1: modality availability heatmap.
    ax1 = fig.add_subplot(2, 2, 1)
    mods = ["rgb", "head_pose", "left_hand", "right_hand", "gaze"]
    avail = np.zeros((len(mods), len(episode)))
    for j, f in enumerate(episode.frames):
        avail[0, j] = 1.0 if f.rgb_path or f.rgb is not None else 0.0
        avail[1, j] = 1.0 if f.head_pose is not None else 0.0
        avail[2, j] = 1.0 if f.left_hand is not None else 0.0
        avail[3, j] = 1.0 if f.right_hand is not None else 0.0
        avail[4, j] = 1.0 if f.gaze is not None else 0.0
    ax1.imshow(avail, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax1.set_yticks(range(len(mods)))
    ax1.set_yticklabels(mods)
    ax1.set_xlabel("frame index")
    ax1.set_title("modality availability")

    # Panel 2: head/camera trajectory (top-down X-Z) with start/end markers.
    ax2 = fig.add_subplot(2, 2, 2)
    pos = episode.head_positions()
    if not np.all(np.isnan(pos)):
        ax2.plot(pos[:, 0], pos[:, 2], "-o", ms=2, color="#3b6fb0")
        ax2.scatter(pos[0, 0], pos[0, 2], c="green", s=60, label="start", zorder=5)
        ax2.scatter(pos[-1, 0], pos[-1, 2], c="red", s=60, label="end", zorder=5)
        ax2.legend(loc="best", fontsize=8)
        ax2.set_aspect("equal", adjustable="datalim")
    else:
        ax2.text(0.5, 0.5, "no head pose", ha="center", va="center")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_title("head trajectory (top-down)")

    # Panel 3: gripper / hand openness over time.
    ax3 = fig.add_subplot(2, 2, 3)
    plotted = False
    for side, color in (("left_hand", "#3ca0ff"), ("right_hand", "#ff7838")):
        vals = [
            getattr(f, side).openness if getattr(f, side) is not None else np.nan
            for f in episode.frames
        ]
        if not np.all(np.isnan(vals)):
            ax3.plot(t, vals, label=side, color=color)
            plotted = True
    if plotted:
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "no hand pose", ha="center", va="center")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("openness (thumb-index / hand size)")
    ax3.set_title("gripper signal")

    # Panel 4: per-step action magnitude (translation + rotation).
    ax4 = fig.add_subplot(2, 2, 4)
    from ego_pipeline.exporters.actions import camera_actions, end_effector_actions

    cam = camera_actions(episode)
    ee = end_effector_actions(episode, hand="right")
    if cam is not None:
        ax4.plot(t[1:], np.linalg.norm(cam[:, :3], axis=1), label="camera Δtrans", color="#7a4fb0")
    if ee is not None:
        ax4.plot(t[1:], np.linalg.norm(ee[:, :3], axis=1), label="right-hand Δtrans", color="#ff7838")
    if cam is None and ee is None:
        ax4.text(0.5, 0.5, "no pose-derived action", ha="center", va="center")
    else:
        ax4.legend(fontsize=8)
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("per-step translation")
    ax4.set_title("derived action magnitude")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def visualize_episode(
    episode: EgoEpisode, out_dir: str, num_overlay_frames: int = 4
) -> list[str]:
    """Write a summary figure plus a few frame overlays; return output paths."""
    os.makedirs(out_dir, exist_ok=True)
    outputs = [
        plot_episode_summary(episode, os.path.join(out_dir, f"{episode.episode_id}_summary.png"))
    ]
    if num_overlay_frames > 0 and len(episode) > 0:
        idxs = np.linspace(0, len(episode) - 1, num=min(num_overlay_frames, len(episode)))
        for k in np.unique(idxs.astype(int)):
            outputs.append(
                render_frame_overlay(
                    episode.frames[int(k)],
                    os.path.join(out_dir, f"{episode.episode_id}_frame_{int(k):04d}.png"),
                )
            )
    return outputs
