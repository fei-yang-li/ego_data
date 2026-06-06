"""End-to-end and unit tests for ego_pipeline (run with: python -m pytest)."""

import json
import os

import numpy as np

from ego_pipeline.exporters.actions import (
    camera_actions,
    end_effector_actions,
    proprioceptive_state,
)
from ego_pipeline.geometry import make_se3, pose_delta, rotation_to_axis_angle
from ego_pipeline.normalize import NormalizeConfig, normalize_episode, resample_episode
from ego_pipeline.pipeline import (
    ExportTarget,
    PipelineConfig,
    ReaderConfig,
    VisualizeConfig,
    run_pipeline,
)
from ego_pipeline.readers import GenericJsonReader, get_reader, list_readers
from ego_pipeline.schema import EgoEpisode, EgoFrame
from ego_pipeline.synthetic import generate_dataset


def test_geometry_axis_angle_roundtrip():
    theta = 0.7
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    aa = rotation_to_axis_angle(R)
    assert np.isclose(np.linalg.norm(aa), theta, atol=1e-6)
    assert np.allclose(aa / np.linalg.norm(aa), [0, 0, 1], atol=1e-6)


def test_pose_delta_translation():
    Ta = make_se3(np.eye(3), [0, 0, 0])
    Tb = make_se3(np.eye(3), [1, 2, 3])
    d = pose_delta(Ta, Tb)
    assert np.allclose(d[:3], [1, 2, 3])
    assert np.allclose(d[3:], 0, atol=1e-8)


def test_reader_registry():
    readers = list_readers()
    for name in ("generic_json", "egoverse", "epic_kitchens"):
        assert name in readers
        assert get_reader(name) is not None


def test_synthetic_and_egoverse_reader(tmp_path):
    root = generate_dataset(
        str(tmp_path / "synth"), num_episodes=2, num_frames=12, fps=30, render_rgb=False
    )
    reader = get_reader("egoverse")(root)
    episodes = reader.read_all()
    assert len(episodes) == 2
    ep = episodes[0]
    assert len(ep) == 12
    assert ep.frames[0].head_pose is not None
    assert ep.frames[0].right_hand is not None
    assert ep.language_instruction


def test_normalize_resample_changes_fps():
    frames = [EgoFrame(timestamp=i / 30.0, head_pose=make_se3(np.eye(3), [i * 0.1, 0, 0]))
              for i in range(30)]
    ep = EgoEpisode("e", frames)
    out = resample_episode(ep, target_fps=10.0)
    assert abs(out.fps - 10.0) < 1.0
    # First/last head positions preserved after resampling.
    assert np.allclose(out.frames[0].head_pose[:3, 3], [0, 0, 0], atol=1e-6)


def test_recenter_moves_start_to_origin():
    frames = [EgoFrame(timestamp=i / 10.0, head_pose=make_se3(np.eye(3), [5 + i, 2, 1]))
              for i in range(5)]
    ep = EgoEpisode("e", frames)
    out = normalize_episode(ep, NormalizeConfig(recenter_trajectory=True))
    assert np.allclose(out.frames[0].head_pose[:3, 3], [0, 0, 0], atol=1e-8)


def test_action_derivation(tmp_path):
    root = generate_dataset(
        str(tmp_path / "synth"), num_episodes=1, num_frames=20, fps=30, render_rgb=False
    )
    ep = get_reader("egoverse")(root).read_all()[0]
    cam = camera_actions(ep)
    ee = end_effector_actions(ep, hand="right")
    state = proprioceptive_state(ep, hand="right")
    assert cam.shape == (len(ep) - 1, 6)
    assert ee.shape == (len(ep) - 1, 7)
    assert state.shape == (len(ep), 7)
    # Camera moves forward -> nonzero translation actions.
    assert np.linalg.norm(cam[:, :3]) > 0


def test_json_roundtrip(tmp_path):
    root = generate_dataset(
        str(tmp_path / "synth"), num_episodes=1, num_frames=8, fps=30, render_rgb=False
    )
    ep = get_reader("egoverse")(root).read_all()[0]
    d = ep.to_dict()
    path = tmp_path / "ep.json"
    path.write_text(json.dumps(d))
    loaded = GenericJsonReader(str(path)).read_all()[0]
    assert len(loaded) == len(ep)
    assert np.allclose(loaded.frames[3].head_pose, ep.frames[3].head_pose)


def test_epic_kitchens_reader(tmp_path):
    csv = tmp_path / "EPIC_100_train.csv"
    csv.write_text(
        "narration_id,participant_id,video_id,start_frame,stop_frame,"
        "start_timestamp,stop_timestamp,narration,verb,noun\n"
        "P01_01_0,P01,P01_01,100,160,00:00:02.00,00:00:03.20,open the fridge,open,fridge\n"
    )
    reader = get_reader("epic_kitchens")(str(tmp_path), max_frames_per_segment=8)
    episodes = reader.read_all()
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.language_instruction == "open the fridge"
    assert len(ep) == 8
    assert ep.frames[0].rgb_path and ep.frames[0].rgb_path.endswith(".jpg")
    # No poses -> VLA/world-model exporters skip these.
    assert camera_actions(ep) is None
    assert end_effector_actions(ep) is None


def test_full_pipeline_exports(tmp_path):
    root = generate_dataset(
        str(tmp_path / "synth"), num_episodes=2, num_frames=16, fps=30, render_rgb=True
    )
    out = tmp_path / "out"
    config = PipelineConfig(
        reader=ReaderConfig("egoverse", root),
        normalize=NormalizeConfig(target_fps=10, recenter_trajectory=True),
        visualize=VisualizeConfig(enabled=True, out_dir=str(out / "viz"), num_overlay_frames=2),
        vla=ExportTarget(enabled=True, out_dir=str(out / "vla")),
        world_model=ExportTarget(enabled=True, out_dir=str(out / "wm")),
    )
    report = run_pipeline(config)
    assert report["num_episodes"] == 2
    # VLA outputs
    assert os.path.isfile(out / "vla" / "meta" / "info.json")
    assert os.path.isfile(out / "vla" / "meta" / "stats.json")
    assert report["outputs"]["vla"]["episodes_written"] == 2
    # World model outputs
    assert os.path.isfile(out / "wm" / "meta" / "index.jsonl")
    assert report["outputs"]["world_model"]["clips_written"] == 2
    # Visualization outputs
    assert report["outputs"]["visualize"]["num_files"] > 0

    # Stats are well-formed.
    stats = json.loads((out / "vla" / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 7


def test_narration_segments(tmp_path):
    root = generate_dataset(
        str(tmp_path / "synth"), num_episodes=1, num_frames=20, fps=30, render_rgb=False
    )
    ep = get_reader("egoverse")(root).read_all()[0]
    segs = ep.narration_segments()
    assert len(segs) >= 2
    assert all(s["text"] for s in segs)
    # Segments are time-ordered and non-overlapping-ish.
    assert segs[0]["start"] <= segs[-1]["end"]
    assert ep.narration_at(ep.frames[0].timestamp) is not None


def test_render_episode_video_and_report(tmp_path):
    from ego_pipeline.report import build_report
    from ego_pipeline.video import render_episode_video

    root = generate_dataset(
        str(tmp_path / "synth"), num_episodes=2, num_frames=10, fps=30, render_rgb=True
    )
    episodes = get_reader("egoverse")(root).read_all()

    vid = render_episode_video(episodes[0], str(tmp_path / "clip.mp4"), fps=10, width=320)
    assert os.path.isfile(vid)
    assert os.path.getsize(vid) > 0
    assert vid.endswith((".mp4", ".gif"))

    # GIF path always works (no codec dependency).
    gif = render_episode_video(episodes[0], str(tmp_path / "clip2.gif"), fps=10, fmt="gif")
    assert gif.endswith(".gif") and os.path.isfile(gif)

    result = build_report(episodes, str(tmp_path / "report"))
    assert os.path.isfile(result["index_html"])
    anns = json.loads(open(result["annotations_json"], encoding="utf-8").read())
    assert len(anns) == 2
    assert anns[0]["narration_segments"]
    assert "narration" in anns[0]["modalities"]
    html = open(result["index_html"], encoding="utf-8").read()
    assert "NARRATION" not in html or "narration segments" in html
    assert "TASK" in html


def test_dataset_loaders(tmp_path):
    from ego_pipeline.datasets import VLADataset, WorldModelDataset

    root = generate_dataset(
        str(tmp_path / "synth"), num_episodes=2, num_frames=12, fps=30, render_rgb=True
    )
    out = tmp_path / "out"
    config = PipelineConfig(
        reader=ReaderConfig("egoverse", root),
        normalize=NormalizeConfig(target_fps=10),
        vla=ExportTarget(enabled=True, out_dir=str(out / "vla")),
        world_model=ExportTarget(enabled=True, out_dir=str(out / "wm")),
    )
    run_pipeline(config)

    vla = VLADataset(str(out / "vla"), load_images=True, action_horizon=4, normalize=True)
    assert len(vla) > 0
    item = vla[1]
    assert item["state"].shape == (7,)
    assert item["action"].shape == (7,)
    assert item["action_chunk"].shape == (4, 7)
    assert item["image"].ndim == 3  # decoded RGB
    # Normalized values stay within [-1, 1].
    assert np.all(np.abs(item["action"]) <= 1.0 + 1e-6)

    wm = WorldModelDataset(str(out / "wm"), load_images=False)
    assert len(wm) == 2
    clip = wm[0]
    assert clip["actions"].shape[1] == 6
    assert len(clip["frames"]) == clip["actions"].shape[0] + 1
    assert clip["caption"]

