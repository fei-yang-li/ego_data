"""Example: load the exported VLA / world-model data for training.

Run from the repo root::

    python examples/torch_dataloader_example.py

It uses the committed sample fixtures under ``examples/sample_output/`` and works
with or without PyTorch installed (the datasets implement ``__len__`` /
``__getitem__`` so a torch ``DataLoader`` can wrap them directly).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, so `ego_pipeline` imports

from ego_pipeline.datasets import VLADataset, WorldModelDataset  # noqa: E402

VLA_DIR = os.path.join(HERE, "sample_output", "vla")
WM_DIR = os.path.join(HERE, "sample_output", "world_model")


def vla_demo() -> None:
    print("=" * 64)
    print("VLA dataset (single-step transitions, action chunking)")
    print("=" * 64)
    ds = VLADataset(VLA_DIR, load_images=True, action_horizon=4, normalize=True)
    print(f"transitions: {len(ds)}  |  action dim: {ds.info['features']['action']['shape']}")
    sample = ds[1]
    img = sample["image"]
    print("sample[1]:")
    print(f"  task           : {sample['task']}")
    print(f"  image          : {type(img).__name__} shape="
          f"{getattr(img, 'shape', None)}")
    print(f"  state    (7,)  : {np.round(sample['state'], 3)}")
    print(f"  action   (7,)  : {np.round(sample['action'], 3)}")
    print(f"  action_chunk   : shape {sample['action_chunk'].shape}")
    print(f"  is_terminal    : {sample['is_terminal']}")

    # Wrap in a torch DataLoader if torch is available.
    try:
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"  torch batch action shape: {tuple(batch['action'].shape)}")
    except Exception as exc:  # torch not installed
        print(f"  (torch DataLoader skipped: {exc.__class__.__name__})")


def world_model_demo() -> None:
    print()
    print("=" * 64)
    print("World-model dataset (video clips + camera actions)")
    print("=" * 64)
    ds = WorldModelDataset(WM_DIR, load_images=True, normalize=False)
    print(f"clips: {len(ds)}")
    clip = ds[0]
    frames = clip["frames"]
    print("clip[0]:")
    print(f"  caption        : {clip['caption']}")
    print(f"  clip_id / fps  : {clip['clip_id']} / {clip['fps']}")
    print(f"  frames         : {type(frames).__name__} shape="
          f"{getattr(frames, 'shape', len(frames))}")
    print(f"  camera actions : shape {clip['actions'].shape}  (Δpose per step)")
    print(f"  action[0]      : {np.round(clip['actions'][0], 4)}")


if __name__ == "__main__":
    vla_demo()
    world_model_demo()
