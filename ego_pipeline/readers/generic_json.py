"""Reader for the pipeline's own JSON interchange format.

This is the simplest reader: it loads episodes that were serialized via
:meth:`EgoEpisode.to_dict`. It is useful as a stable interchange format and as
the format produced by the synthetic data generator used in tests/demos.

Layout (either is accepted)::

    root/episode_0001.json          # one EgoEpisode dict per file
    root/episode_0002.json
    ...

or a single file containing a list of episode dicts::

    root.json  -> [ {episode...}, {episode...} ]
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

from ego_pipeline.readers.base import BaseReader, register_reader
from ego_pipeline.schema import EgoEpisode


@register_reader("generic_json")
class GenericJsonReader(BaseReader):
    def read(self) -> Iterator[EgoEpisode]:
        if os.path.isfile(self.root):
            with open(self.root, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            episodes = data if isinstance(data, list) else [data]
            for ep in episodes:
                yield EgoEpisode.from_dict(ep)
            return

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"path not found: {self.root}")

        files = sorted(
            f for f in os.listdir(self.root) if f.endswith(".json")
        )
        for name in files:
            with open(os.path.join(self.root, name), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for ep in data:
                    yield EgoEpisode.from_dict(ep)
            else:
                yield EgoEpisode.from_dict(data)
