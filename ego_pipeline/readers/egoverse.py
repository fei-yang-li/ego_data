"""Reader for an EgoVerse-style pose-rich layout.

EgoVerse-class datasets provide RGB plus 3D hand keypoints (21 per hand),
6-DoF head pose and dense language annotations. The exact on-disk container
varies (HDF5 / EgoDB / npz); this reader targets a self-describing per-episode
directory layout that captures the same content and is trivial to produce::

    root/
      <episode_id>/
        meta.json     # {language_instruction, fps?, intrinsics?, source?, metadata?}
        poses.npz     # arrays: timestamps (T,), head_pose (T,4,4),
                      #         left_hand (T,21,3), right_hand (T,21,3), gaze (T,3)
        rgb/                 # optional image frames frame_000000.jpg ...

Any array in ``poses.npz`` is optional; missing modalities simply produce
``None`` fields on the canonical frames.

Options
-------
rgb_template: str
    ``str.format`` template given ``episode_dir`` and ``index`` (0-based).
    Default ``{episode_dir}/rgb/frame_{index:06d}.jpg``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import numpy as np

from ego_pipeline.readers.base import BaseReader, register_reader
from ego_pipeline.schema import (
    CameraIntrinsics,
    EgoEpisode,
    EgoFrame,
    Handedness,
    HandPose,
    PoseConvention,
)


@register_reader("egoverse")
class EgoVerseReader(BaseReader):
    DEFAULT_RGB_TEMPLATE = "{episode_dir}/rgb/frame_{index:06d}.jpg"

    def read(self) -> Iterator[EgoEpisode]:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"dataset root not found: {self.root}")

        rgb_template = self.options.get("rgb_template", self.DEFAULT_RGB_TEMPLATE)
        episode_dirs = sorted(
            d
            for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )
        for ep_id in episode_dirs:
            ep_dir = os.path.join(self.root, ep_id)
            poses_path = os.path.join(ep_dir, "poses.npz")
            if not os.path.isfile(poses_path):
                continue
            yield self._build_episode(ep_id, ep_dir, poses_path, rgb_template)

    def _build_episode(
        self, ep_id: str, ep_dir: str, poses_path: str, rgb_template: str
    ) -> EgoEpisode:
        meta_path = os.path.join(ep_dir, "meta.json")
        meta: dict = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

        intrinsics = (
            CameraIntrinsics.from_dict(meta["intrinsics"])
            if meta.get("intrinsics")
            else None
        )

        with np.load(poses_path) as data:
            arrays = {k: data[k] for k in data.files}

        timestamps = arrays.get("timestamps")
        if timestamps is None:
            # Derive from fps if absent.
            fps = float(meta.get("fps", 30.0))
            n = _infer_length(arrays)
            timestamps = np.arange(n, dtype=np.float64) / fps
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        n = len(timestamps)

        head_pose = arrays.get("head_pose")
        left_hand = arrays.get("left_hand")
        right_hand = arrays.get("right_hand")
        gaze = arrays.get("gaze")

        frames: list[EgoFrame] = []
        for i in range(n):
            rgb_path = rgb_template.format(episode_dir=ep_dir, index=i)
            if not os.path.isfile(rgb_path):
                rgb_path = None

            frames.append(
                EgoFrame(
                    timestamp=float(timestamps[i]),
                    rgb_path=rgb_path,
                    intrinsics=intrinsics,
                    head_pose=head_pose[i] if head_pose is not None else None,
                    left_hand=(
                        HandPose(Handedness.LEFT, left_hand[i], PoseConvention.WORLD)
                        if left_hand is not None
                        else None
                    ),
                    right_hand=(
                        HandPose(Handedness.RIGHT, right_hand[i], PoseConvention.WORLD)
                        if right_hand is not None
                        else None
                    ),
                    gaze=gaze[i] if gaze is not None else None,
                )
            )

        return EgoEpisode(
            episode_id=ep_id,
            frames=frames,
            language_instruction=meta.get("language_instruction", ""),
            source=meta.get("source", "egoverse"),
            metadata=meta.get("metadata", {}),
        )


def _infer_length(arrays: dict[str, np.ndarray]) -> int:
    for key in ("head_pose", "left_hand", "right_hand", "gaze"):
        if key in arrays:
            return len(arrays[key])
    return 0
