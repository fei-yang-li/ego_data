"""Canonical in-memory schema for egocentric data.

Every dataset reader normalizes its source into the structures defined here so
that downstream visualization and export code can be written once, against a
single representation, regardless of where the data came from.

Conventions
-----------
* All time stamps are in **seconds**.
* Camera / head poses are stored as 4x4 homogeneous ``SE(3)`` matrices mapping a
  point in the *device* (camera) frame to the *world* frame, i.e.
  ``p_world = T_world_cam @ p_cam``.
* Hand keypoints follow the 21-point MANO / MediaPipe ordering with index 0 the
  wrist. They may live in the world or the camera frame; the active frame is
  recorded on :class:`HandPose`.
* Right-handed coordinate frames are assumed throughout.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 21-keypoint hand topology (wrist + 5 fingers x 4 joints), MANO/MediaPipe order.
HAND_NUM_KEYPOINTS = 21
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
)
THUMB_TIP = 4
INDEX_TIP = 8


class PoseConvention(str, enum.Enum):
    """Which coordinate frame a quantity is expressed in."""

    WORLD = "world"
    CAMERA = "camera"


class Handedness(str, enum.Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics (pixels)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project ``(N, 3)`` camera-frame points to ``(N, 2)`` pixel coords."""
        points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = np.clip(points_cam[:, 2:3], 1e-6, None)
        uv = points_cam[:, :2] / z
        u = uv[:, 0] * self.fx + self.cx
        v = uv[:, 1] * self.fy + self.cy
        return np.stack([u, v], axis=-1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraIntrinsics":
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
        )


@dataclass
class HandPose:
    """3D hand keypoints for a single hand."""

    handedness: Handedness
    keypoints: np.ndarray  # (21, 3) float
    convention: PoseConvention = PoseConvention.WORLD
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64).reshape(-1, 3)
        if self.keypoints.shape[0] != HAND_NUM_KEYPOINTS:
            raise ValueError(
                f"expected {HAND_NUM_KEYPOINTS} keypoints, got {self.keypoints.shape[0]}"
            )
        if isinstance(self.handedness, str):
            self.handedness = Handedness(self.handedness)
        if isinstance(self.convention, str):
            self.convention = PoseConvention(self.convention)

    @property
    def wrist(self) -> np.ndarray:
        return self.keypoints[0]

    @property
    def openness(self) -> float:
        """Normalized thumb-index tip distance, a proxy for gripper aperture.

        Scaled by hand size (wrist -> middle finger base) so it is roughly
        invariant to absolute scale / units.
        """
        tip_dist = float(np.linalg.norm(self.keypoints[THUMB_TIP] - self.keypoints[INDEX_TIP]))
        hand_scale = float(np.linalg.norm(self.keypoints[9] - self.keypoints[0])) + 1e-6
        return tip_dist / hand_scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "handedness": self.handedness.value,
            "keypoints": self.keypoints.tolist(),
            "convention": self.convention.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HandPose":
        return cls(
            handedness=Handedness(d["handedness"]),
            keypoints=np.asarray(d["keypoints"], dtype=np.float64),
            convention=PoseConvention(d.get("convention", "world")),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class EgoFrame:
    """A single synchronized time step of an egocentric recording."""

    timestamp: float
    rgb_path: str | None = None
    rgb: np.ndarray | None = None  # (H, W, 3) uint8, optional in-memory image
    intrinsics: CameraIntrinsics | None = None
    # 4x4 SE(3): device/camera-to-world transform.
    head_pose: np.ndarray | None = None
    left_hand: HandPose | None = None
    right_hand: HandPose | None = None
    gaze: np.ndarray | None = None  # (3,) gaze direction or point, world/camera frame
    narration: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.head_pose is not None:
            self.head_pose = np.asarray(self.head_pose, dtype=np.float64).reshape(4, 4)
        if self.gaze is not None:
            self.gaze = np.asarray(self.gaze, dtype=np.float64).reshape(-1)

    def hands(self) -> list[HandPose]:
        return [h for h in (self.left_hand, self.right_hand) if h is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rgb_path": self.rgb_path,
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
            "head_pose": self.head_pose.tolist() if self.head_pose is not None else None,
            "left_hand": self.left_hand.to_dict() if self.left_hand else None,
            "right_hand": self.right_hand.to_dict() if self.right_hand else None,
            "gaze": self.gaze.tolist() if self.gaze is not None else None,
            "narration": self.narration,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EgoFrame":
        return cls(
            timestamp=float(d["timestamp"]),
            rgb_path=d.get("rgb_path"),
            intrinsics=(
                CameraIntrinsics.from_dict(d["intrinsics"]) if d.get("intrinsics") else None
            ),
            head_pose=(
                np.asarray(d["head_pose"], dtype=np.float64) if d.get("head_pose") else None
            ),
            left_hand=HandPose.from_dict(d["left_hand"]) if d.get("left_hand") else None,
            right_hand=HandPose.from_dict(d["right_hand"]) if d.get("right_hand") else None,
            gaze=np.asarray(d["gaze"], dtype=np.float64) if d.get("gaze") else None,
            narration=d.get("narration"),
            extra=d.get("extra", {}),
        )


@dataclass
class EgoEpisode:
    """An ordered sequence of :class:`EgoFrame` plus episode-level metadata."""

    episode_id: str
    frames: list[EgoFrame]
    language_instruction: str = ""
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep frames strictly ordered in time; readers may emit them unsorted.
        self.frames.sort(key=lambda f: f.timestamp)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([f.timestamp for f in self.frames], dtype=np.float64)

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return float(self.frames[-1].timestamp - self.frames[0].timestamp)

    @property
    def fps(self) -> float:
        if len(self.frames) < 2 or self.duration <= 0:
            return 0.0
        return float((len(self.frames) - 1) / self.duration)

    def head_positions(self) -> np.ndarray:
        """``(T, 3)`` world positions of the head/camera (NaN where missing)."""
        out = np.full((len(self.frames), 3), np.nan, dtype=np.float64)
        for i, f in enumerate(self.frames):
            if f.head_pose is not None:
                out[i] = f.head_pose[:3, 3]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "language_instruction": self.language_instruction,
            "source": self.source,
            "metadata": self.metadata,
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EgoEpisode":
        return cls(
            episode_id=str(d["episode_id"]),
            frames=[EgoFrame.from_dict(f) for f in d["frames"]],
            language_instruction=d.get("language_instruction", ""),
            source=d.get("source", "unknown"),
            metadata=d.get("metadata", {}),
        )
