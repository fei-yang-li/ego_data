"""Interactive single-page HTML data viewer.

The committed template ``assets/viewer_template.html`` is a self-contained
vanilla-JS app (search, source/modality filters, per-clip video + narration
*timeline* visualization). :func:`render_viewer` injects the clip records into
that template so the resulting ``index.html`` works by simply double-clicking it
— the data is inlined, so no local web server / fetch is required.

Each record is a clip annotation (see :func:`ego_pipeline.report.clip_annotation`)
optionally augmented with ``_video`` / ``_summary`` media file names (relative to
the HTML file) so the viewer can embed them.
"""

from __future__ import annotations

import json
import os
from typing import Any

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "assets", "viewer_template.html")
_PLACEHOLDER = "__CLIP_DATA__"


def load_template() -> str:
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def render_viewer(records: list[dict[str, Any]], out_path: str) -> str:
    """Write an interactive viewer HTML embedding ``records``; return its path."""
    template = load_template()
    # ``</`` is escaped so an annotation string can never break out of the
    # inlined <script> block.
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace(_PLACEHOLDER, data)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


def viewer_from_annotations_file(
    annotations_path: str, out_path: str, media_dir: str | None = None
) -> str:
    """Regenerate the viewer from an existing ``clip_annotations.json``.

    If ``media_dir`` is given (defaults to the annotations' directory), matching
    ``<id>.mp4`` / ``<id>.gif`` / ``<id>_summary.png`` files are linked in.
    """
    with open(annotations_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    media_dir = media_dir or os.path.dirname(os.path.abspath(annotations_path))
    out_dir = os.path.dirname(os.path.abspath(out_path))

    for rec in records:
        safe = str(rec.get("episode_id", "")).replace("/", "_")
        for ext in (".mp4", ".gif"):
            cand = os.path.join(media_dir, safe + ext)
            if os.path.isfile(cand):
                rec["_video"] = os.path.relpath(cand, out_dir)
                break
        summ = os.path.join(media_dir, safe + "_summary.png")
        if os.path.isfile(summ):
            rec["_summary"] = os.path.relpath(summ, out_dir)
    return render_viewer(records, out_path)
