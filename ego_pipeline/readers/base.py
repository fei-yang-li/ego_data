"""Base class and registry for dataset readers."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any

from ego_pipeline.schema import EgoEpisode

_REGISTRY: dict[str, type["BaseReader"]] = {}


def register_reader(name: str):
    """Class decorator registering a reader under ``name``."""

    def _wrap(cls: type["BaseReader"]) -> type["BaseReader"]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"reader '{name}' already registered")
        _REGISTRY[key] = cls
        cls.reader_name = name
        return cls

    return _wrap


def get_reader(name: str) -> type["BaseReader"]:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown reader '{name}'. available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_readers() -> list[str]:
    return sorted(_REGISTRY)


class BaseReader(abc.ABC):
    """Abstract base for all dataset readers.

    Parameters
    ----------
    root:
        Path to the dataset root on disk.
    options:
        Reader-specific keyword options (forwarded from config / CLI).
    """

    reader_name: str = "base"

    def __init__(self, root: str, **options: Any) -> None:
        self.root = root
        self.options = options

    @abc.abstractmethod
    def read(self) -> Iterator[EgoEpisode]:
        """Yield canonical episodes one at a time (lazy where possible)."""
        raise NotImplementedError

    def read_all(self) -> list[EgoEpisode]:
        return list(self.read())

    def __iter__(self) -> Iterator[EgoEpisode]:
        return self.read()
