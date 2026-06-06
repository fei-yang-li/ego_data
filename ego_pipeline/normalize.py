"""Normalization and temporal-alignment transforms over canonical episodes.

These transforms make episodes from different sources directly comparable:

* **Temporal resampling** to a fixed target FPS (uniform time grid).
* **Trajectory recentring** so each episode starts at the world origin and,
  optionally, with an identity head orientation (device frame == world frame at
  ``t = 0``). This removes arbitrary capture-rig placement.
* **Hand frame conversion** from world to the (egocentric) camera frame, which
  is what action models usually consume.

All transforms return *new* episodes and never mutate the inputs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ego_pipeline.geometry import se3_inverse
from ego_pipeline.schema import EgoEpisode, EgoFrame, HandPose, PoseConvention


@dataclass
class NormalizeConfig:
    target_fps: float | None = None
    recenter_trajectory: bool = True
    align_start_orientation: bool = False
    hands_to_camera: bool = False
    min_frames: int = 2


def _interp_vec(t_new: np.ndarray, t_old: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per-channel linear interpolation of ``(T, D)`` values onto ``t_new``."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(t_old), -1)
    out = np.empty((len(t_new), flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_new, t_old, flat[:, c])
    return out.reshape((len(t_new),) + values.shape[1:])


def resample_episode(episode: EgoEpisode, target_fps: float) -> EgoEpisode:
    """Resample an episode onto a uniform ``target_fps`` time grid.

    Poses, keypoints and gaze are linearly interpolated; ``rgb_path`` and
    narration are taken from the nearest original frame (images cannot be
    interpolated by path).
    """
    if len(episode) < 2 or target_fps <= 0:
        return episode

    t_old = episode.timestamps
    t0, t1 = float(t_old[0]), float(t_old[-1])
    n_new = int(round((t1 - t0) * target_fps)) + 1
    n_new = max(n_new, 2)
    t_new = t0 + np.arange(n_new) / target_fps

    # Pre-stack available modalities (NaN-fill missing so interp still runs).
    head = np.stack(
        [f.head_pose if f.head_pose is not None else np.full((4, 4), np.nan) for f in episode.frames]
    )
    has_head = not np.all(np.isnan(head))

    def stack_hand(side: str) -> tuple[np.ndarray, bool, object]:
        present = [getattr(f, side) for f in episode.frames]
        any_present = any(h is not None for h in present)
        ref = next((h for h in present if h is not None), None)
        arr = np.stack(
            [
                h.keypoints if h is not None else np.full((21, 3), np.nan)
                for h in present
            ]
        )
        return arr, any_present, ref

    left_arr, has_left, left_ref = stack_hand("left_hand")
    right_arr, has_right, right_ref = stack_hand("right_hand")

    has_gaze = any(f.gaze is not None for f in episode.frames)
    gaze_dim = next((len(f.gaze) for f in episode.frames if f.gaze is not None), 3)
    gaze = np.stack(
        [f.gaze if f.gaze is not None else np.full(gaze_dim, np.nan) for f in episode.frames]
    )

    head_i = _interp_vec(t_new, t_old, head) if has_head else None
    left_i = _interp_vec(t_new, t_old, left_arr) if has_left else None
    right_i = _interp_vec(t_new, t_old, right_arr) if has_right else None
    gaze_i = _interp_vec(t_new, t_old, gaze) if has_gaze else None

    nn_idx = np.clip(np.searchsorted(t_old, t_new), 0, len(t_old) - 1)

    new_frames: list[EgoFrame] = []
    for k in range(n_new):
        src = episode.frames[int(nn_idx[k])]
        nf = EgoFrame(
            timestamp=float(t_new[k]),
            rgb_path=src.rgb_path,
            intrinsics=src.intrinsics,
            narration=src.narration,
            head_pose=_renorm_se3(head_i[k]) if head_i is not None else None,
            gaze=gaze_i[k] if gaze_i is not None else None,
            extra=copy.deepcopy(src.extra),
        )
        if left_i is not None and left_ref is not None:
            nf.left_hand = HandPose(left_ref.handedness, left_i[k], left_ref.convention)
        if right_i is not None and right_ref is not None:
            nf.right_hand = HandPose(right_ref.handedness, right_i[k], right_ref.convention)
        new_frames.append(nf)

    out = EgoEpisode(
        episode_id=episode.episode_id,
        frames=new_frames,
        language_instruction=episode.language_instruction,
        source=episode.source,
        metadata={**episode.metadata, "resampled_fps": target_fps},
    )
    return out


def _renorm_se3(T: np.ndarray) -> np.ndarray:
    """Re-orthonormalize the rotation block after interpolation (SVD projection)."""
    T = np.array(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    u, _, vt = np.linalg.svd(R)
    Rn = u @ vt
    if np.linalg.det(Rn) < 0:
        u[:, -1] *= -1
        Rn = u @ vt
    T[:3, :3] = Rn
    T[3, :] = [0, 0, 0, 1]
    return T


def recenter_trajectory(
    episode: EgoEpisode, align_orientation: bool = False
) -> EgoEpisode:
    """Shift (and optionally rotate) so the first head pose sits at the origin."""
    first = next((f for f in episode.frames if f.head_pose is not None), None)
    if first is None:
        return episode
    T0_inv = se3_inverse(first.head_pose)
    if not align_orientation:
        # Translation-only recentring keeps world orientation intact.
        t0 = first.head_pose[:3, 3]
        T0_inv = np.eye(4)
        T0_inv[:3, 3] = -t0

    new = copy.deepcopy(episode)
    for f in new.frames:
        if f.head_pose is not None:
            f.head_pose = T0_inv @ f.head_pose
        for hand in (f.left_hand, f.right_hand):
            if hand is not None and hand.convention == PoseConvention.WORLD:
                kpts_h = np.concatenate(
                    [hand.keypoints, np.ones((hand.keypoints.shape[0], 1))], axis=1
                )
                hand.keypoints = (T0_inv @ kpts_h.T).T[:, :3]
    new.metadata = {**episode.metadata, "recentered": True}
    return new


def hands_to_camera_frame(episode: EgoEpisode) -> EgoEpisode:
    """Express world-frame hand keypoints in the per-frame camera frame."""
    new = copy.deepcopy(episode)
    for f in new.frames:
        if f.head_pose is None:
            continue
        cam_from_world = se3_inverse(f.head_pose)
        for hand in (f.left_hand, f.right_hand):
            if hand is not None and hand.convention == PoseConvention.WORLD:
                kpts_h = np.concatenate(
                    [hand.keypoints, np.ones((hand.keypoints.shape[0], 1))], axis=1
                )
                hand.keypoints = (cam_from_world @ kpts_h.T).T[:, :3]
                hand.convention = PoseConvention.CAMERA
    return new


def normalize_episode(episode: EgoEpisode, config: NormalizeConfig) -> EgoEpisode | None:
    """Apply the full normalization stack; return ``None`` if too short."""
    out = episode
    if config.target_fps:
        out = resample_episode(out, config.target_fps)
    if config.recenter_trajectory:
        out = recenter_trajectory(out, align_orientation=config.align_start_orientation)
    if config.hands_to_camera:
        out = hands_to_camera_frame(out)
    if len(out) < config.min_frames:
        return None
    return out
