"""Command line interface for ego_pipeline.

Subcommands::

    python -m ego_pipeline synth     --out DIR [--episodes N --frames N --no-rgb]
    python -m ego_pipeline inspect   --reader NAME --root PATH [--limit N]
    python -m ego_pipeline visualize --reader NAME --root PATH --out DIR [--limit N]
    python -m ego_pipeline run       --config config.yaml
    python -m ego_pipeline run       --reader NAME --root PATH --out DIR \
                                     [--fps F] [--vla] [--world-model] [--viz]
"""

from __future__ import annotations

import argparse
import json
import sys

from ego_pipeline.normalize import NormalizeConfig
from ego_pipeline.pipeline import (
    ExportTarget,
    PipelineConfig,
    ReaderConfig,
    VisualizeConfig,
    load_episodes,
    run_pipeline,
)
from ego_pipeline.readers import get_reader, list_readers


def _parse_options(items: list[str] | None) -> dict:
    opts: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--option expects key=value, got '{item}'")
        key, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass  # keep as string
        opts[key] = val
    return opts


def cmd_synth(args: argparse.Namespace) -> int:
    from ego_pipeline.synthetic import generate_dataset

    path = generate_dataset(
        args.out,
        num_episodes=args.episodes,
        num_frames=args.frames,
        fps=args.fps,
        render_rgb=not args.no_rgb,
        seed=args.seed,
    )
    print(f"synthetic dataset written to: {path}")
    print(f"  reader=egoverse episodes={args.episodes} frames={args.frames} rgb={not args.no_rgb}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    reader = get_reader(args.reader)(args.root, **_parse_options(args.option))
    count = 0
    total_frames = 0
    for ep in reader.read():
        count += 1
        total_frames += len(ep)
        mods = []
        f0 = ep.frames[0] if ep.frames else None
        if f0:
            if f0.rgb_path or f0.rgb is not None:
                mods.append("rgb")
            if f0.head_pose is not None:
                mods.append("head")
            if f0.left_hand or f0.right_hand:
                mods.append("hands")
            if f0.gaze is not None:
                mods.append("gaze")
        print(
            f"[{count:04d}] {ep.episode_id:>24}  frames={len(ep):4d}  "
            f"dur={ep.duration:6.2f}s  fps={ep.fps:5.1f}  "
            f"mods={'+'.join(mods) or 'none':<20}  '{ep.language_instruction[:50]}'"
        )
        if args.limit and count >= args.limit:
            break
    print(f"\ntotal: {count} episodes, {total_frames} frames")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    from ego_pipeline.visualize import visualize_episode

    config = PipelineConfig(
        reader=ReaderConfig(args.reader, args.root, _parse_options(args.option)),
        normalize=NormalizeConfig(target_fps=args.fps),
        limit=args.limit,
    )
    episodes = load_episodes(config)
    n = 0
    for ep in episodes:
        paths = visualize_episode(ep, args.out, args.overlay_frames)
        n += len(paths)
    print(f"wrote {n} visualization files for {len(episodes)} episodes to {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        if not args.reader or not args.root:
            raise SystemExit("either --config or both --reader and --root are required")
        out = args.out or "out"
        config = PipelineConfig(
            reader=ReaderConfig(args.reader, args.root, _parse_options(args.option)),
            normalize=NormalizeConfig(
                target_fps=args.fps,
                recenter_trajectory=not args.no_recenter,
                hands_to_camera=args.hands_to_camera,
            ),
            visualize=VisualizeConfig(enabled=args.viz, out_dir=f"{out}/viz"),
            vla=ExportTarget(enabled=args.vla, out_dir=f"{out}/vla"),
            world_model=ExportTarget(enabled=args.world_model, out_dir=f"{out}/world_model"),
            limit=args.limit,
        )
    report = run_pipeline(config)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ego_pipeline",
        description="General-purpose egocentric data processing pipeline "
        f"(readers: {', '.join(list_readers())}).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_synth = sub.add_parser("synth", help="generate a synthetic dataset")
    p_synth.add_argument("--out", required=True)
    p_synth.add_argument("--episodes", type=int, default=3)
    p_synth.add_argument("--frames", type=int, default=30)
    p_synth.add_argument("--fps", type=float, default=30.0)
    p_synth.add_argument("--no-rgb", action="store_true", help="skip rendering RGB frames")
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.set_defaults(func=cmd_synth)

    p_inspect = sub.add_parser("inspect", help="list/normalize-read episodes")
    p_inspect.add_argument("--reader", required=True, choices=list_readers())
    p_inspect.add_argument("--root", required=True)
    p_inspect.add_argument("--option", action="append", help="reader option key=value")
    p_inspect.add_argument("--limit", type=int)
    p_inspect.set_defaults(func=cmd_inspect)

    p_viz = sub.add_parser("visualize", help="render inspection figures/overlays")
    p_viz.add_argument("--reader", required=True, choices=list_readers())
    p_viz.add_argument("--root", required=True)
    p_viz.add_argument("--out", required=True)
    p_viz.add_argument("--option", action="append")
    p_viz.add_argument("--fps", type=float, default=None)
    p_viz.add_argument("--overlay-frames", type=int, default=4)
    p_viz.add_argument("--limit", type=int)
    p_viz.set_defaults(func=cmd_visualize)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument("--config", help="YAML config path")
    p_run.add_argument("--reader", choices=list_readers())
    p_run.add_argument("--root")
    p_run.add_argument("--out", help="output root dir (flag mode)")
    p_run.add_argument("--option", action="append")
    p_run.add_argument("--fps", type=float, default=None)
    p_run.add_argument("--no-recenter", action="store_true")
    p_run.add_argument("--hands-to-camera", action="store_true")
    p_run.add_argument("--viz", action="store_true")
    p_run.add_argument("--vla", action="store_true")
    p_run.add_argument("--world-model", action="store_true")
    p_run.add_argument("--limit", type=int)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
