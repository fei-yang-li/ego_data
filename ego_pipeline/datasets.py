"""Torch-free dataset loaders for the exported VLA / world-model formats.

These classes implement ``__len__`` / ``__getitem__`` so they plug straight into
``torch.utils.data.DataLoader`` (which only requires those two methods), yet they
have **no hard dependency on PyTorch** — handy for inspection and for CI without
a heavy torch install. If torch is present they also subclass
``torch.utils.data.Dataset`` so ``isinstance`` checks pass.

Usage
-----
>>> from ego_pipeline.datasets import VLADataset, WorldModelDataset
>>> ds = VLADataset("out/vla", load_images=True, action_horizon=4)
>>> sample = ds[0]
>>> sample["action_chunk"].shape   # (4, 7)
>>> # then: torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

try:  # optional: subclass torch Dataset when available
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore

    _Base = _TorchDataset
except Exception:  # pragma: no cover - torch usually absent here
    _Base = object


def _load_image(path: str) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


class VLADataset(_Base):
    """Flat transition dataset over an exported VLA directory.

    Each item is one time step::

        {
          "image": np.ndarray | str,    # H,W,3 if load_images else path
          "state": (7,) float32,
          "action": (7,) float32,        # action at this step
          "action_chunk": (H, 7) float32,# next H actions (action chunking)
          "task": str,
          "timestamp": float,
          "is_terminal": bool,
          "episode_index": int,
        }

    Parameters
    ----------
    root: path to the ``out/vla`` directory.
    load_images: decode RGB into arrays (else return the image path string).
    action_horizon: length of the returned action chunk (1 = single action).
    normalize: if True, normalize ``state``/``action`` with the saved q01/q99
        statistics to roughly [-1, 1].
    image_key: feature key holding the image path (matches the exporter).
    """

    def __init__(
        self,
        root: str,
        load_images: bool = False,
        action_horizon: int = 1,
        normalize: bool = False,
        image_key: str = "observation.images.ego",
    ) -> None:
        import pandas as pd

        self.root = root
        self.load_images = load_images
        self.action_horizon = max(1, int(action_horizon))
        self.normalize = normalize
        self.image_key = image_key

        with open(os.path.join(root, "meta", "info.json"), encoding="utf-8") as fh:
            self.info = json.load(fh)
        stats_path = os.path.join(root, "meta", "stats.json")
        self.stats = json.load(open(stats_path, encoding="utf-8")) if os.path.isfile(stats_path) else {}

        files = sorted(glob.glob(os.path.join(root, "data", "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet files under {root}/data")
        self._episodes: list[Any] = [pd.read_parquet(f) for f in files]

        # Flat (episode_idx, row_idx) index across all episodes.
        self._index: list[tuple[int, int]] = []
        for ei, df in enumerate(self._episodes):
            self._index.extend((ei, ri) for ri in range(len(df)))

    def __len__(self) -> int:
        return len(self._index)

    def _norm(self, vec: np.ndarray, key: str) -> np.ndarray:
        s = self.stats.get(key)
        if not s:
            return vec
        lo = np.asarray(s["q01"], dtype=np.float64)
        hi = np.asarray(s["q99"], dtype=np.float64)
        rng = np.clip(hi - lo, 1e-6, None)
        return np.clip(2.0 * (vec - lo) / rng - 1.0, -1.0, 1.0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ei, ri = self._index[idx]
        df = self._episodes[ei]
        row = df.iloc[ri]

        state = np.asarray(row["observation.state"], dtype=np.float32)
        action = np.asarray(row["action"], dtype=np.float32)

        # Action chunk: this step + following steps, clamped within the episode.
        chunk = np.stack(
            [
                np.asarray(df.iloc[min(ri + k, len(df) - 1)]["action"], dtype=np.float32)
                for k in range(self.action_horizon)
            ]
        )

        if self.normalize:
            state = self._norm(state, "observation.state").astype(np.float32)
            action = self._norm(action, "action").astype(np.float32)
            chunk = np.stack([self._norm(a, "action") for a in chunk]).astype(np.float32)

        img_path = str(row[self.image_key])
        image: Any = _load_image(img_path) if self.load_images else img_path

        return {
            "image": image,
            "state": state,
            "action": action,
            "action_chunk": chunk,
            "task": str(row["task"]),
            "timestamp": float(row["timestamp"]),
            "is_terminal": bool(row["is_terminal"]),
            "episode_index": int(row["episode_index"]),
        }


class WorldModelDataset(_Base):
    """Clip dataset over an exported world-model directory.

    Each item is a (sub)clip::

        {
          "frames": list[str] | np.ndarray,  # paths, or (L,H,W,3) if load_images
          "actions": (L-1, 6) float32,        # camera delta-pose per step
          "caption": str,
          "fps": float,
          "clip_id": str,
        }

    Parameters
    ----------
    root: path to the ``out/world_model`` directory.
    clip_len: if set, sample fixed-length windows of this many frames from each
        clip (stride = ``clip_len``); otherwise yield whole clips.
    load_images: decode frames into a stacked array.
    normalize: normalize the camera actions with the saved q01/q99 stats.
    """

    def __init__(
        self,
        root: str,
        clip_len: int | None = None,
        load_images: bool = False,
        normalize: bool = False,
    ) -> None:
        self.root = root
        self.clip_len = clip_len
        self.load_images = load_images
        self.normalize = normalize

        stats_path = os.path.join(root, "meta", "stats.json")
        self.stats = json.load(open(stats_path, encoding="utf-8")) if os.path.isfile(stats_path) else {}

        index_path = os.path.join(root, "meta", "index.jsonl")
        self._clips: list[dict] = []
        with open(index_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._clips.append(json.loads(line))

        # Build sample windows: (clip_idx, start_frame, length).
        self._windows: list[tuple[int, int, int]] = []
        for ci, clip in enumerate(self._clips):
            n = clip["num_frames"]
            if self.clip_len is None:
                self._windows.append((ci, 0, n))
            else:
                for start in range(0, max(1, n - self.clip_len + 1), self.clip_len):
                    self._windows.append((ci, start, self.clip_len))

    def __len__(self) -> int:
        return len(self._windows)

    def _load_actions(self, clip: dict) -> np.ndarray:
        return np.load(os.path.join(self.root, clip["action_file"])).astype(np.float32)

    def _norm(self, actions: np.ndarray) -> np.ndarray:
        s = self.stats.get("action")
        if not s:
            return actions
        lo = np.asarray(s["q01"], dtype=np.float64)
        hi = np.asarray(s["q99"], dtype=np.float64)
        rng = np.clip(hi - lo, 1e-6, None)
        return np.clip(2.0 * (actions - lo) / rng - 1.0, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ci, start, length = self._windows[idx]
        clip = self._clips[ci]

        frames = clip["frames"][start : start + length]
        actions = self._load_actions(clip)[start : start + max(0, length - 1)]
        if self.normalize:
            actions = self._norm(actions)

        if self.load_images:
            imgs = [_load_image(p) for p in frames]
            imgs = [im for im in imgs if im is not None]
            frames_out: Any = np.stack(imgs) if imgs else np.empty((0,))
        else:
            frames_out = frames

        return {
            "frames": frames_out,
            "actions": actions,
            "caption": clip["caption"],
            "fps": float(clip["fps"]),
            "clip_id": clip["clip_id"],
        }
