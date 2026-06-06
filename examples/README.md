# examples — 样例训练数据与加载器

这里提供**已生成好的样例产物**（无需自己跑生成），以及一个 PyTorch 风格的加载器示例。

## 目录

```
examples/
├── README.md
├── torch_dataloader_example.py   # 加载样例数据的可运行示例
├── sample_dataset/               # 合成的 EgoVerse 风格源数据（2 episodes × 8 帧，含 narration）
│   └── episode_000{0,1}/{meta.json, poses.npz, rgb/*.jpg}
├── sample_output/                # pipeline 导出的训练数据
│   ├── vla/                      # VLA 格式（LeRobot/RLDS 风格）
│   │   ├── meta/{info.json, episodes.jsonl, stats.json}
│   │   └── data/episode_*.parquet
│   └── world_model/              # 世界模型格式（EgoVid 风格）
│       ├── meta/{info.json, index.jsonl, stats.json}
│       └── actions/episode_*.npy
└── sample_report/                # 完整数据视频 + clip 语义标注
    ├── index.html                # 交互式可视化（搜索/过滤/narration 时间轴，双击打开）
    ├── gallery.html              # 无 JS 的静态后备版
    ├── clip_annotations.json     # 机器可读的语义标注
    ├── episode_*.mp4             # 带字幕条的完整数据视频
    └── episode_*_summary.png
```

直接用浏览器打开 `examples/sample_report/index.html` 就能交互式浏览每条 clip 的视频、
指令、按时间分段的语义标注时间轴与诊断图（支持搜索与按来源/模态过滤）。

## 运行加载器示例

```bash
python examples/torch_dataloader_example.py
```

不依赖 PyTorch 也能运行（加载器实现了 `__len__/__getitem__`，可直接被
`torch.utils.data.DataLoader` 包裹）。装了 torch 时示例会额外演示用 `DataLoader`
取一个 batch。

## VLA 加载器 `VLADataset`

把 `out/vla` 当作**逐步 transition** 数据集，每个样本：

```python
from ego_pipeline.datasets import VLADataset
ds = VLADataset("examples/sample_output/vla",
                load_images=True,    # 解码 RGB 为 (H,W,3) 数组；False 时返回路径
                action_horizon=4,    # 返回未来 4 步的 action chunk
                normalize=True)      # 用 q01/q99 归一化到 [-1,1]
sample = ds[0]
# {image, state(7,), action(7,), action_chunk(4,7), task, timestamp, is_terminal, episode_index}
```

## 世界模型加载器 `WorldModelDataset`

把 `out/world_model` 当作**视频 clip** 数据集，每个样本：

```python
from ego_pipeline.datasets import WorldModelDataset
ds = WorldModelDataset("examples/sample_output/world_model",
                       clip_len=None,    # 设为整数则按固定长度切窗
                       load_images=True) # 解码为 (L,H,W,3)；False 时返回路径列表
clip = ds[0]
# {frames, actions(L-1,6), caption, fps, clip_id}
```

## 自己重新生成这份样例

```bash
python - <<'PY'
from ego_pipeline.synthetic import generate_dataset
from ego_pipeline.pipeline import PipelineConfig, ReaderConfig, ExportTarget, run_pipeline
from ego_pipeline.normalize import NormalizeConfig
generate_dataset("examples/sample_dataset", num_episodes=2, num_frames=6, render_rgb=True)
run_pipeline(PipelineConfig(
    reader=ReaderConfig("egoverse", "examples/sample_dataset"),
    normalize=NormalizeConfig(recenter_trajectory=True),
    vla=ExportTarget(enabled=True, out_dir="examples/sample_output/vla"),
    world_model=ExportTarget(enabled=True, out_dir="examples/sample_output/world_model"),
))
PY
```
