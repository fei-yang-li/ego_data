"""Export canonical episodes to a video world-model training format.

World models for egocentric video (e.g. EgoVid-5M style) are trained on
(video, action, caption) triplets where the *action* is the camera kinematics
that the generated frames must obey. We emit, per clip::

    <out>/
      meta/
        info.json        # action space + schema
        index.jsonl      # one record per clip: frames[], caption, action file,
                         #                       fps, num_frames
        stats.json       # camera-action normalization statistics
      actions/
        <episode_id>.npy # (T-1, 6) camera delta-pose actions [dx dy dz rx ry rz]

Frames are referenced by their on-disk paths (no copying), keeping exports cheap
for multi-TB datasets. Episodes without head pose are skipped (no camera action
can be derived) and reported in the summary.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ego_pipeline.exporters.actions import camera_actions, compute_action_stats
from ego_pipeline.schema import EgoEpisode


class WorldModelExporter:
    def __init__(self, out_dir: str, fps: float | None = None) -> None:
        self.out_dir = out_dir
        self.fps = fps
        self.meta_dir = os.path.join(out_dir, "meta")
        self.actions_dir = os.path.join(out_dir, "actions")

    def export(self, episodes: list[EgoEpisode]) -> dict:
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.actions_dir, exist_ok=True)

        all_actions: list[np.ndarray] = []
        records = []
        written = 0

        for ep in episodes:
            cam = camera_actions(ep)
            if cam is None:
                continue

            safe_id = ep.episode_id.replace("/", "_")
            action_path = os.path.join(self.actions_dir, f"{safe_id}.npy")
            np.save(action_path, cam.astype(np.float32))

            frames = [f.rgb_path for f in ep.frames if f.rgb_path]
            records.append(
                {
                    "clip_id": safe_id,
                    "source": ep.source,
                    "caption": ep.language_instruction,
                    "num_frames": len(ep),
                    "num_action_steps": int(len(cam)),
                    "fps": self.fps or round(ep.fps, 3),
                    "frames": frames,
                    "action_file": os.path.relpath(action_path, self.out_dir),
                    "has_frames": len(frames) > 0,
                }
            )
            all_actions.append(cam)
            written += 1

        info = {
            "format": "world_model",
            "version": "ego_pipeline-wm-0.1",
            "action_space": {
                "dtype": "float32",
                "shape": [6],
                "names": ["dx", "dy", "dz", "rx", "ry", "rz"],
                "description": "per-frame camera delta pose in previous camera frame "
                "(translation + axis-angle rotation)",
            },
            "total_clips": written,
        }
        with open(os.path.join(self.meta_dir, "info.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        with open(os.path.join(self.meta_dir, "index.jsonl"), "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as fh:
            json.dump({"action": compute_action_stats(all_actions)}, fh, indent=2)

        return {
            "format": "world_model",
            "out_dir": self.out_dir,
            "clips_written": written,
            "clips_skipped": len(episodes) - written,
        }
