"""Dataset readers that normalize heterogeneous sources into the canonical schema.

Each reader subclasses :class:`~ego_pipeline.readers.base.BaseReader` and yields
:class:`~ego_pipeline.schema.EgoEpisode` objects. New datasets are added by
implementing a single ``read`` method and registering the class.
"""

from ego_pipeline.readers.base import BaseReader, get_reader, list_readers, register_reader
from ego_pipeline.readers.egoverse import EgoVerseReader
from ego_pipeline.readers.epic_kitchens import EpicKitchensReader
from ego_pipeline.readers.generic_json import GenericJsonReader

__all__ = [
    "BaseReader",
    "EgoVerseReader",
    "EpicKitchensReader",
    "GenericJsonReader",
    "get_reader",
    "list_readers",
    "register_reader",
]
