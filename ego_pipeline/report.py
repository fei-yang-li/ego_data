"""Build a browsable clip-annotation report for a set of episodes.

Produces, under ``out_dir``:

* ``<id>.mp4`` / ``.gif`` — the full annotated data video per clip,
* ``<id>_summary.png`` — the diagnostic figure per clip,
* ``clip_annotations.json`` — machine-readable semantic annotations (clip
  instruction + temporal narration segments + modality coverage),
* ``index.html`` — a self-contained gallery to eyeball videos alongside their
  semantic annotations.

This is the "查看完整数据视频 + clip 语义标注" entry point.
"""

from __future__ import annotations

import html
import json
import os

from ego_pipeline.schema import EgoEpisode
from ego_pipeline.video import render_episode_video
from ego_pipeline.visualize import plot_episode_summary


def _modalities(episode: EgoEpisode) -> list[str]:
    f = episode.frames[0] if episode.frames else None
    mods = []
    if f is None:
        return mods
    if any(fr.rgb_path or fr.rgb is not None for fr in episode.frames):
        mods.append("rgb")
    if any(fr.head_pose is not None for fr in episode.frames):
        mods.append("head_pose")
    if any(fr.left_hand or fr.right_hand for fr in episode.frames):
        mods.append("hands")
    if any(fr.gaze is not None for fr in episode.frames):
        mods.append("gaze")
    if any(fr.narration for fr in episode.frames):
        mods.append("narration")
    return mods


def clip_annotation(episode: EgoEpisode) -> dict:
    """Structured semantic annotation for a single clip."""
    return {
        "episode_id": episode.episode_id,
        "source": episode.source,
        "language_instruction": episode.language_instruction,
        "duration_s": round(episode.duration, 3),
        "fps": round(episode.fps, 3),
        "num_frames": len(episode),
        "modalities": _modalities(episode),
        "narration_segments": [
            {
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
                "text": s["text"],
                "num_frames": s["num_frames"],
            }
            for s in episode.narration_segments()
        ],
        "metadata": episode.metadata,
    }


def _video_tag(rel: str) -> str:
    if rel.endswith(".mp4"):
        return (
            f'<video controls loop muted width="100%" src="{html.escape(rel)}">'
            f"</video>"
        )
    return f'<img src="{html.escape(rel)}" width="100%" alt="clip video"/>'


def _clip_card(ann: dict, video_rel: str | None, summary_rel: str | None) -> str:
    seg_rows = "".join(
        f"<tr><td>{s['start']:.2f}</td><td>{s['end']:.2f}</td>"
        f"<td>{html.escape(s['text'])}</td></tr>"
        for s in ann["narration_segments"]
    ) or '<tr><td colspan="3"><i>no narration segments</i></td></tr>'

    media = _video_tag(video_rel) if video_rel else "<i>no video</i>"
    summary = (
        f'<img src="{html.escape(summary_rel)}" width="100%" alt="summary"/>'
        if summary_rel
        else ""
    )
    mods = " ".join(f'<span class="tag">{m}</span>' for m in ann["modalities"])

    return f"""
    <div class="card">
      <h2>{html.escape(ann['episode_id'])} <span class="src">{html.escape(ann['source'])}</span></h2>
      <p class="instr">TASK: {html.escape(ann['language_instruction'] or '(none)')}</p>
      <p class="meta">{ann['num_frames']} frames · {ann['duration_s']}s · {ann['fps']} fps · {mods}</p>
      <div class="media">{media}</div>
      <details open><summary>语义标注 / narration segments</summary>
        <table><tr><th>start (s)</th><th>end (s)</th><th>narration</th></tr>{seg_rows}</table>
      </details>
      <details><summary>诊断图 / diagnostics</summary>{summary}</details>
    </div>
    """


_HTML_HEAD = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Ego clip annotations</title><style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#0f1116;color:#e6e6e6}
header{padding:16px 24px;background:#161922;border-bottom:1px solid #262b36}
h1{margin:0;font-size:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;padding:20px}
.card{background:#161922;border:1px solid #262b36;border-radius:10px;padding:14px}
.card h2{font-size:15px;margin:0 0 6px}
.src{font-size:11px;color:#7f8a9a;font-weight:normal}
.instr{color:#78dcff;margin:4px 0;font-size:13px}
.meta{color:#9aa3b2;font-size:12px;margin:4px 0}
.tag{background:#23314a;color:#a9c7ff;border-radius:4px;padding:1px 6px;font-size:11px;margin-right:2px}
.media{margin:8px 0}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{border:1px solid #2a3140;padding:4px 6px;text-align:left}
th{background:#1d2230}
details{margin-top:8px}
summary{cursor:pointer;color:#ffd778;font-size:13px}
</style></head><body>
<header><h1>Egocentric 数据视频 &amp; clip 语义标注</h1></header>
<div class="grid">
"""


def build_report(
    episodes: list[EgoEpisode],
    out_dir: str,
    with_video: bool = True,
    with_summary: bool = True,
    video_fmt: str = "auto",
    video_fps: float | None = None,
    video_width: int | None = 640,
) -> dict:
    """Render videos + summaries and write ``index.html`` / ``clip_annotations.json``."""
    os.makedirs(out_dir, exist_ok=True)
    annotations = []
    cards = []

    for ep in episodes:
        ann = clip_annotation(ep)
        annotations.append(ann)
        safe = ep.episode_id.replace("/", "_")

        video_rel = None
        if with_video:
            try:
                path = render_episode_video(
                    ep,
                    os.path.join(out_dir, f"{safe}.mp4"),
                    fps=video_fps,
                    width=video_width,
                    fmt=video_fmt,
                )
                video_rel = os.path.basename(path)
            except Exception as exc:  # keep report generation resilient
                ann["video_error"] = f"{type(exc).__name__}: {exc}"

        summary_rel = None
        if with_summary:
            sp = plot_episode_summary(ep, os.path.join(out_dir, f"{safe}_summary.png"))
            summary_rel = os.path.basename(sp)

        cards.append(_clip_card(ann, video_rel, summary_rel))

    with open(os.path.join(out_dir, "clip_annotations.json"), "w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=2, ensure_ascii=False)

    html_path = os.path.join(out_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(_HTML_HEAD + "\n".join(cards) + "\n</div></body></html>")

    return {
        "out_dir": out_dir,
        "index_html": html_path,
        "annotations_json": os.path.join(out_dir, "clip_annotations.json"),
        "num_clips": len(annotations),
    }
