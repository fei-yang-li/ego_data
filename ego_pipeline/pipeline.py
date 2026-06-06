"""High-level orchestration tying readers, normalization, viz and exporters.

A :class:`PipelineConfig` fully specifies a run and can be loaded from YAML so
the same processing is reproducible across datasets::

    reader:
      name: egoverse
      root: /data/egoverse
      options: {}
    normalize:
      target_fps: 10
      recenter_trajectory: true
      hands_to_camera: false
    visualize:
      enabled: true
      out_dir: out/viz
      num_overlay_frames: 4
    export:
      vla:   {enabled: true, out_dir: out/vla, hand: right}
      world_model: {enabled: true, out_dir: out/world_model}
    limit: null
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ego_pipeline.normalize import NormalizeConfig, normalize_episode
from ego_pipeline.readers import get_reader
from ego_pipeline.schema import EgoEpisode


@dataclass
class ReaderConfig:
    name: str = "generic_json"
    root: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizeConfig:
    enabled: bool = False
    out_dir: str = "out/viz"
    num_overlay_frames: int = 4
    max_episodes: int | None = 8


@dataclass
class ExportTarget:
    enabled: bool = False
    out_dir: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    vla: ExportTarget = field(default_factory=ExportTarget)
    world_model: ExportTarget = field(default_factory=ExportTarget)
    limit: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        reader = ReaderConfig(**d.get("reader", {}))
        normalize = NormalizeConfig(**d.get("normalize", {}))
        visualize = VisualizeConfig(**d.get("visualize", {}))
        export = d.get("export", {})
        vla = ExportTarget(
            enabled=export.get("vla", {}).get("enabled", False),
            out_dir=export.get("vla", {}).get("out_dir", "out/vla"),
            options={k: v for k, v in export.get("vla", {}).items()
                     if k not in ("enabled", "out_dir")},
        )
        world_model = ExportTarget(
            enabled=export.get("world_model", {}).get("enabled", False),
            out_dir=export.get("world_model", {}).get("out_dir", "out/world_model"),
            options={k: v for k, v in export.get("world_model", {}).items()
                     if k not in ("enabled", "out_dir")},
        )
        return cls(
            reader=reader,
            normalize=normalize,
            visualize=visualize,
            vla=vla,
            world_model=world_model,
            limit=d.get("limit"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})


def load_episodes(config: PipelineConfig) -> list[EgoEpisode]:
    """Read + normalize episodes according to ``config`` (the '归一化读取' stage)."""
    reader_cls = get_reader(config.reader.name)
    reader = reader_cls(config.reader.root, **config.reader.options)

    episodes: list[EgoEpisode] = []
    for ep in reader.read():
        norm = normalize_episode(ep, config.normalize)
        if norm is None:
            continue
        episodes.append(norm)
        if config.limit is not None and len(episodes) >= config.limit:
            break
    return episodes


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute the full read -> normalize -> visualize -> export pipeline."""
    episodes = load_episodes(config)
    report: dict[str, Any] = {
        "reader": config.reader.name,
        "root": config.reader.root,
        "num_episodes": len(episodes),
        "total_frames": sum(len(e) for e in episodes),
        "outputs": {},
    }

    if config.visualize.enabled and episodes:
        from ego_pipeline.visualize import visualize_episode

        subset = episodes
        if config.visualize.max_episodes is not None:
            subset = episodes[: config.visualize.max_episodes]
        viz_paths: list[str] = []
        for ep in subset:
            viz_paths += visualize_episode(
                ep, config.visualize.out_dir, config.visualize.num_overlay_frames
            )
        report["outputs"]["visualize"] = {
            "out_dir": config.visualize.out_dir,
            "num_files": len(viz_paths),
        }

    if config.vla.enabled and episodes:
        from ego_pipeline.exporters import VLAExporter

        exporter = VLAExporter(
            config.vla.out_dir,
            fps=config.normalize.target_fps,
            **config.vla.options,
        )
        report["outputs"]["vla"] = exporter.export(episodes)

    if config.world_model.enabled and episodes:
        from ego_pipeline.exporters import WorldModelExporter

        exporter = WorldModelExporter(
            config.world_model.out_dir,
            fps=config.normalize.target_fps,
            **config.world_model.options,
        )
        report["outputs"]["world_model"] = exporter.export(episodes)

    return report


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
