"""Reader for EPIC-KITCHENS-100 style annotations.

EPIC-KITCHENS ships action segments as CSV rows (``EPIC_100_train.csv`` etc.)
referencing extracted RGB frames on disk. There are no hand/head poses, so the
resulting episodes carry RGB + a language instruction (the narration). This
reader demonstrates how an "RGB + language" dataset maps onto the canonical
schema.

Relevant CSV columns: ``narration_id, participant_id, video_id, start_frame,
stop_frame, start_timestamp, stop_timestamp, narration``.

Options
-------
annotations: str
    Path to the CSV file (default ``<root>/EPIC_100_train.csv``).
frames_template: str
    ``str.format`` template for a frame image given ``participant``, ``video``
    and ``frame`` (1-based index). Default mirrors the official layout:
    ``{root}/{participant}/rgb_frames/{video}/frame_{frame:010d}.jpg``.
fps: float
    Frame rate used to derive timestamps when not present (default 50.0).
max_frames_per_segment: int
    Cap on frames sampled per action segment (default 32).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from ego_pipeline.readers.base import BaseReader, register_reader
from ego_pipeline.schema import EgoEpisode, EgoFrame


def _parse_timestamp(value: str) -> float | None:
    """Parse ``HH:MM:SS.mmm`` into seconds; return None on failure."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    try:
        parts = value.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(value)
    except (ValueError, TypeError):
        return None


@register_reader("epic_kitchens")
class EpicKitchensReader(BaseReader):
    DEFAULT_TEMPLATE = "{root}/{participant}/rgb_frames/{video}/frame_{frame:010d}.jpg"

    def read(self) -> Iterator[EgoEpisode]:
        import csv

        annotations = self.options.get(
            "annotations", os.path.join(self.root, "EPIC_100_train.csv")
        )
        template = self.options.get("frames_template", self.DEFAULT_TEMPLATE)
        fps = float(self.options.get("fps", 50.0))
        max_frames = int(self.options.get("max_frames_per_segment", 32))

        if not os.path.isfile(annotations):
            raise FileNotFoundError(f"annotations CSV not found: {annotations}")

        with open(annotations, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield self._row_to_episode(row, template, fps, max_frames)

    def _row_to_episode(
        self, row: dict[str, str], template: str, fps: float, max_frames: int
    ) -> EgoEpisode:
        narration_id = row.get("narration_id") or row.get("uid") or "segment"
        participant = row.get("participant_id", "")
        video = row.get("video_id", "")
        narration = (row.get("narration") or "").strip()

        start_frame = int(float(row.get("start_frame", 1) or 1))
        stop_frame = int(float(row.get("stop_frame", start_frame) or start_frame))
        stop_frame = max(stop_frame, start_frame)

        start_ts = _parse_timestamp(row.get("start_timestamp"))
        stop_ts = _parse_timestamp(row.get("stop_timestamp"))

        total = stop_frame - start_frame + 1
        stride = max(1, total // max_frames)
        frame_indices = list(range(start_frame, stop_frame + 1, stride))[:max_frames]

        frames: list[EgoFrame] = []
        for fi in frame_indices:
            if start_ts is not None and stop_ts is not None and total > 1:
                frac = (fi - start_frame) / (total - 1)
                ts = start_ts + frac * (stop_ts - start_ts)
            else:
                ts = fi / fps
            rgb_path = template.format(
                root=self.root, participant=participant, video=video, frame=fi
            )
            frames.append(
                EgoFrame(timestamp=ts, rgb_path=rgb_path, narration=narration)
            )

        return EgoEpisode(
            episode_id=str(narration_id),
            frames=frames,
            language_instruction=narration,
            source="epic_kitchens",
            metadata={
                "participant_id": participant,
                "video_id": video,
                "verb": row.get("verb"),
                "noun": row.get("noun"),
            },
        )
