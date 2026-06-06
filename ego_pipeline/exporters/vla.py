"""Export canonical episodes to a Vision-Language-Action (VLA) dataset.

The layout follows the conventions used by LeRobot / RLDS style pipelines so the
output can be loaded by common VLA training stacks (OpenVLA, Octo, pi0, ...)::

    <out>/
      meta/
        info.json       # feature schema, fps, action/state dims
        episodes.jsonl  # one record per episode (length, instruction)
        stats.json      # action & state normalization statistics
      data/
        episode_000000.parquet   # per-step: index, timestamp, state, action,
        episode_000001.parquet   #           image path, task instruction
        ...

Per step we store the right-hand end-effector action
``[dx, dy, dz, rx, ry, rz, grip]`` (see
:mod:`ego_pipeline.exporters.actions`), the absolute proprioceptive state, the
RGB image path and the language instruction. Actions are length ``T-1``; the
final observation has no action (padded with the conventional terminal action).
"""

from __future__ import annotations

import json
import os

import numpy as np

from ego_pipeline.exporters.actions import (
    compute_action_stats,
    end_effector_actions,
    proprioceptive_state,
)
from ego_pipeline.schema import EgoEpisode


class VLAExporter:
    def __init__(
        self,
        out_dir: str,
        hand: str = "right",
        image_key: str = "observation.images.ego",
        fps: float | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.hand = hand
        self.image_key = image_key
        self.fps = fps
        self.data_dir = os.path.join(out_dir, "data")
        self.meta_dir = os.path.join(out_dir, "meta")

    def export(self, episodes: list[EgoEpisode]) -> dict:
        import pandas as pd

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

        all_actions: list[np.ndarray] = []
        all_states: list[np.ndarray] = []
        episode_records = []
        global_index = 0
        written = 0

        for ep_idx, ep in enumerate(episodes):
            actions = end_effector_actions(ep, hand=self.hand)
            states = proprioceptive_state(ep, hand=self.hand)
            if actions is None or states is None:
                # No hand pose -> cannot build an action-labelled VLA episode.
                continue

            # Pad the action sequence to length T with a zero terminal action.
            T = len(ep)
            padded = np.zeros((T, actions.shape[1]), dtype=np.float64)
            padded[: T - 1] = actions

            rows = []
            for i, f in enumerate(ep.frames):
                rows.append(
                    {
                        "index": global_index,
                        "episode_index": written,
                        "frame_index": i,
                        "timestamp": float(f.timestamp),
                        "observation.state": states[i].tolist(),
                        "action": padded[i].tolist(),
                        self.image_key: f.rgb_path or "",
                        "task": ep.language_instruction,
                        "is_terminal": bool(i == T - 1),
                    }
                )
                global_index += 1

            df = pd.DataFrame(rows)
            out_path = os.path.join(self.data_dir, f"episode_{written:06d}.parquet")
            df.to_parquet(out_path, index=False)

            all_actions.append(actions)
            all_states.append(states)
            episode_records.append(
                {
                    "episode_index": written,
                    "original_id": ep.episode_id,
                    "source": ep.source,
                    "length": T,
                    "tasks": [ep.language_instruction],
                }
            )
            written += 1

        info = {
            "codebase_version": "ego_pipeline-vla-0.1",
            "robot_type": "egocentric_hand",
            "fps": self.fps,
            "total_episodes": written,
            "total_frames": global_index,
            "features": {
                "action": {"dtype": "float32", "shape": [7],
                            "names": ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]},
                "observation.state": {"dtype": "float32", "shape": [7],
                                       "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
                self.image_key: {"dtype": "image_path", "shape": []},
                "task": {"dtype": "string"},
            },
        }
        with open(os.path.join(self.meta_dir, "info.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        with open(os.path.join(self.meta_dir, "episodes.jsonl"), "w", encoding="utf-8") as fh:
            for rec in episode_records:
                fh.write(json.dumps(rec) + "\n")
        stats = {
            "action": compute_action_stats(all_actions),
            "observation.state": compute_action_stats(all_states),
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

        return {
            "format": "vla",
            "out_dir": self.out_dir,
            "episodes_written": written,
            "episodes_skipped": len(episodes) - written,
            "total_frames": global_index,
        }
