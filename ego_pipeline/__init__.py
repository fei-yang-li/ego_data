"""ego_pipeline: a general-purpose egocentric data processing pipeline.

The pipeline has three stages that mirror the typical needs when working with
egocentric (first-person) datasets such as Ego4D, EPIC-KITCHENS, Ego-Exo4D and
EgoVerse:

1. **Normalized reading** (``ego_pipeline.readers``): adapters that load
   heterogeneous on-disk layouts into one canonical in-memory schema
   (:mod:`ego_pipeline.schema`).
2. **Visualization / inspection** (``ego_pipeline.visualize``): render frames,
   hand keypoints, gaze, camera trajectories and action timelines so the data
   can be sanity-checked.
3. **Export** (``ego_pipeline.exporters``): convert the canonical episodes into
   action-labelled formats consumed by VLA (Vision-Language-Action) models and
   by video world models.

See :mod:`ego_pipeline.pipeline` for the high level orchestration and
``python -m ego_pipeline`` for the command line interface.
"""

from ego_pipeline.schema import (
    CameraIntrinsics,
    EgoEpisode,
    EgoFrame,
    HandPose,
    PoseConvention,
)

__all__ = [
    "CameraIntrinsics",
    "EgoEpisode",
    "EgoFrame",
    "HandPose",
    "PoseConvention",
]

__version__ = "0.1.0"
