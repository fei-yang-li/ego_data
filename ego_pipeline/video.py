"""Render a full episode to a playable video with semantic-annotation captions.

Each output frame stacks the RGB image (with hand-skeleton + gaze overlay from
:mod:`ego_pipeline.visualize`) and a bottom caption bar showing:

* the clip-level **language instruction**,
* the **active narration** segment at the current time (the clip's semantic
  annotation track), and
* a progress/time read-out.

MP4 is written via OpenCV (system ffmpeg). If that backend is unavailable the
renderer transparently falls back to an animated GIF (PIL only), so it works in
minimal environments too.
"""

from __future__ import annotations

import os

import numpy as np

from ego_pipeline.schema import EgoEpisode
from ego_pipeline.visualize import render_frame_overlay_image

_BAR_BG = (18, 18, 22)
_INSTR_COLOR = (120, 220, 255)
_NARR_COLOR = (255, 215, 120)
_TIME_COLOR = (170, 170, 175)


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def compose_video_frame(
    episode: EgoEpisode, index: int, width: int | None = None, bar_height: int = 84
):
    """Build a single annotated video frame (PIL image) for ``episode.frames[index]``."""
    from PIL import Image, ImageDraw

    frame = episode.frames[index]
    img = render_frame_overlay_image(frame).convert("RGB")
    if width and img.width != width:
        h = int(img.height * width / img.width)
        img = img.resize((width, h))
    w, h = img.size

    canvas = Image.new("RGB", (w, h + bar_height), _BAR_BG)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    t = frame.timestamp - episode.frames[0].timestamp
    narration = episode.narration_at(frame.timestamp) or "—"
    max_chars = max(20, w // 7)

    y = h + 6
    instr_lines = _wrap(f"TASK: {episode.language_instruction or '(none)'}", max_chars)
    draw.text((8, y), instr_lines[0], fill=_INSTR_COLOR)
    y += 14
    draw.text((8, y), _wrap(f"NARRATION: {narration}", max_chars)[0], fill=_NARR_COLOR)
    y += 14
    draw.text(
        (8, y),
        f"t={t:6.2f}s  frame {index + 1}/{len(episode)}  src={episode.source}",
        fill=_TIME_COLOR,
    )

    # Thin progress bar along the very bottom.
    prog = (index + 1) / max(1, len(episode))
    draw.rectangle([0, h + bar_height - 4, int(w * prog), h + bar_height - 1], fill=_INSTR_COLOR)
    return canvas


def _write_mp4_cv2(frames: list[np.ndarray], out_path: str, fps: float) -> bool:
    try:
        import cv2
    except Exception:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, max(1.0, fps), (w, h))
    if not writer.isOpened():
        return False
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def _write_gif(frames: list[np.ndarray], out_path: str, fps: float) -> str:
    from PIL import Image

    imgs = [Image.fromarray(fr) for fr in frames]
    duration_ms = max(20, int(1000 / max(1.0, fps)))
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def render_episode_video(
    episode: EgoEpisode,
    out_path: str,
    fps: float | None = None,
    width: int | None = 640,
    fmt: str = "auto",
) -> str:
    """Render ``episode`` to a captioned video file; return the written path.

    ``fmt`` is ``"mp4"``, ``"gif"`` or ``"auto"`` (mp4 if possible, else gif).
    The returned path may differ from ``out_path`` when falling back to GIF.
    """
    if len(episode) == 0:
        raise ValueError("cannot render an empty episode")

    fps = fps or (episode.fps if episode.fps > 0 else 10.0)
    frames = [
        np.asarray(compose_video_frame(episode, i, width=width))
        for i in range(len(episode))
    ]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    base, _ = os.path.splitext(out_path)

    if fmt in ("auto", "mp4"):
        mp4_path = base + ".mp4"
        if _write_mp4_cv2(frames, mp4_path, fps):
            return mp4_path
        if fmt == "mp4":
            raise RuntimeError("mp4 backend unavailable (no working OpenCV/ffmpeg)")

    gif_path = base + ".gif"
    return _write_gif(frames, gif_path, fps)
