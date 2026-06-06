# ego_pipeline — 通用 Egocentric 数据处理 Pipeline

一个轻量、可扩展的第一人称（egocentric）数据处理管线，把异构的 ego 数据集
（Ego4D / EPIC-KITCHENS / Ego-Exo4D / EgoVerse 等）统一成一种规范结构，并支持：

1. **数据归一化读取**：不同数据集 → 统一的规范 schema（`ego_pipeline.schema`）
2. **数据可视化检查**：轨迹 / 模态可用性 / 手部骨架 / 注视点 / 动作幅度
3. **导出为可训练格式**：
   - **VLA**（Vision-Language-Action，LeRobot / RLDS 风格）
   - **世界模型 action**（视频 + 相机运动 action + caption，EgoVid 风格）

整个库只依赖 `numpy / matplotlib / pillow / pandas / pyarrow / pyyaml`，无需 GPU。

---

## 架构

```
原始数据集 ──▶ Reader ──▶ 规范 EgoEpisode ──▶ Normalize ──▶ Visualize（检查）
（异构布局）   归一化读取    (schema.py)        归一化/对齐         │
                                                                ├──▶ VLAExporter        (out/vla)
                                                                └──▶ WorldModelExporter (out/world_model)
```

### 规范 schema（`schema.py`）

| 结构 | 内容 |
|------|------|
| `EgoEpisode` | `episode_id`、有序 `frames`、`language_instruction`、`source`、`metadata` |
| `EgoFrame` | `timestamp`、`rgb_path/rgb`、`intrinsics`、`head_pose(4x4 SE3)`、`left/right_hand`、`gaze`、`narration` |
| `HandPose` | 21 个 MANO/MediaPipe 关键点 `(21,3)`、左右手、坐标系、`openness`（夹爪开合代理） |
| `CameraIntrinsics` | 针孔内参，提供 `project()` 投影 |

约定：时间戳单位为秒；位姿为 4x4 `SE(3)`（device→world）；右手坐标系。

---

## 安装

```bash
pip install -r requirements.txt
```

## 快速上手（用内置合成数据，无需下载真实数据集）

```bash
# 1) 生成一个合成 EgoVerse 风格数据集（含位姿 + 渲染 RGB）
python -m ego_pipeline synth --out out/synthetic --episodes 3 --frames 30 --fps 30

# 2) 归一化读取并打印每条 episode 的统计信息
python -m ego_pipeline inspect --reader egoverse --root out/synthetic

# 3) 跑通完整管线（归一化 + 可视化 + 导出 VLA + 导出世界模型格式）
python -m ego_pipeline run --config configs/example_egoverse.yaml
```

输出：

```
out/viz/            每条 episode 的诊断图 + 若干帧手部骨架/注视点叠加图
out/vla/            VLA 数据集（meta/info.json, meta/stats.json, data/*.parquet）
out/world_model/    世界模型数据（meta/index.jsonl, actions/*.npy, meta/stats.json）
```

也可以不写 config，直接用命令行开关：

```bash
python -m ego_pipeline run --reader egoverse --root out/synthetic --out out \
    --fps 10 --viz --vla --world-model
```

---

## 支持的 Reader（数据归一化读取）

| 名称 | 适用数据集 | 输入布局 | 模态 |
|------|-----------|----------|------|
| `egoverse` | EgoVerse 等位姿丰富数据集 | `<ep>/poses.npz` + `meta.json`（+ `rgb/`） | RGB + 头部位姿 + 双手 + 注视 |
| `epic_kitchens` | EPIC-KITCHENS-100 | `EPIC_100_*.csv` + 抽帧目录 | RGB + 语言 |
| `generic_json` | 本库 JSON 交换格式 | 由 `EgoEpisode.to_dict()` 序列化 | 任意 |

新增数据集只需继承 `BaseReader` 实现一个 `read()`，并用 `@register_reader("name")`
注册即可：

```python
from ego_pipeline.readers.base import BaseReader, register_reader
from ego_pipeline.schema import EgoEpisode, EgoFrame

@register_reader("my_dataset")
class MyReader(BaseReader):
    def read(self):
        for ...:
            yield EgoEpisode(episode_id=..., frames=[EgoFrame(...), ...],
                             language_instruction=..., source="my_dataset")
```

---

## 归一化（`normalize.py`）

- **时间重采样** `target_fps`：把不同帧率统一到固定时间网格（位姿/关键点线性插值，
  旋转矩阵插值后用 SVD 重正交化；RGB 路径取最近邻）。
- **轨迹重定心** `recenter_trajectory`：每条 episode 从世界原点开始（可选 `align_start_orientation`
  让 `t=0` 时 device 系与 world 系对齐），消除采集设备摆放差异。
- **手部坐标系转换** `hands_to_camera`：将世界系手部关键点转换到相机系（动作模型常用）。

---

## 动作派生（`exporters/actions.py`）

| 用途 | 函数 | 维度 | 含义 |
|------|------|------|------|
| VLA 动作 | `end_effector_actions` | `(T-1, 7)` | 手腕相对位姿增量 `[dx,dy,dz,rx,ry,rz]`（轴角）+ 夹爪开合 |
| VLA 状态 | `proprioceptive_state` | `(T, 7)` | 手腕绝对位置 + 轴角姿态 + 夹爪 |
| 世界模型动作 | `camera_actions` | `(T-1, 6)` | 相机/头部相对位姿增量（即视频需遵循的相机运动） |

手腕姿态由手部关键点的掌面几何估计；夹爪开合由拇指-食指尖距离归一化得到。
导出时会自动统计 `mean/std/min/max/q01/q99`（VLA 训练常用的动作归一化统计量）。

---

## 导出格式

### VLA（`out/vla/`，LeRobot / RLDS 风格）

```
meta/info.json       特征 schema、fps、动作/状态维度
meta/episodes.jsonl  每条 episode（长度、指令）
meta/stats.json      动作 & 状态归一化统计
data/episode_*.parquet
    每步：index, episode_index, frame_index, timestamp,
          observation.state(7), action(7), observation.images.ego(路径),
          task(语言指令), is_terminal
```

### 世界模型（`out/world_model/`，EgoVid 风格）

```
meta/info.json    相机 action 空间定义
meta/index.jsonl  每个 clip：frames[]（图像路径）、caption、num_frames、action_file、fps
meta/stats.json   相机 action 归一化统计
actions/<id>.npy  (T-1, 6) 相机位姿增量
```

> 帧图像通过路径引用、不复制，因此导出对 TB 级数据集依然轻量。
> 没有手部位姿的数据集（如 EPIC-KITCHENS）VLA 导出会自动跳过并在 summary 中报告。

---

## 可视化（`visualize.py`）

- `plot_episode_summary`：四宫格诊断图——模态可用性热图、头部轨迹俯视图、
  夹爪开合信号、派生动作幅度曲线。
- `render_frame_overlay`：将手部骨架投影叠加到 RGB 帧（或空白画布）+ 注视点，
  用于排查标定 / 坐标系错误。

---

## 命令行总览

```bash
python -m ego_pipeline synth     --out DIR [--episodes N --frames N --fps F --no-rgb]
python -m ego_pipeline inspect   --reader NAME --root PATH [--limit N --option k=v]
python -m ego_pipeline visualize --reader NAME --root PATH --out DIR [--fps F --limit N]
python -m ego_pipeline run        --config config.yaml
python -m ego_pipeline run        --reader NAME --root PATH --out DIR [--fps F --viz --vla --world-model]
```

## 测试

```bash
python -m pytest tests/ -q
```
