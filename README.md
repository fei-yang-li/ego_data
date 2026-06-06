# Egocentric Datasets Research

开源第一人称（Egocentric）视觉数据集调研报告，涵盖 **2012–2026** 年主流数据集。

## 内容概览

- **经典大规模数据集**: Ego4D、EPIC-KITCHENS-100、Ego-Exo4D
- **2025-2026 新发布**: EgoLife、Ego-1K、WAGIBench、EgoDrive、OSMO 等
- **领域专项**: 工业、医疗、眼动、手势、音乐、360°、VLM 评测
- **对比分析**: 规模、交互密度、模态覆盖、设备趋势
- **选型指南**: 按场景推荐最佳数据集
- **许可与合规**: 各数据集许可证对比

## 文件说明

```
egocentric-datasets/
├── README.md                          # 项目说明
├── egocentric-datasets-research.md    # 完整调研报告
├── ego_pipeline/                      # 通用 ego 数据处理 pipeline（见下）
│   └── README.md                      # pipeline 使用说明
├── configs/                           # pipeline 示例配置
└── tests/                             # pipeline 测试
```

## 通用 Ego 数据处理 Pipeline

`ego_pipeline/` 提供了一个轻量、可扩展的第一人称数据处理管线，支持：

- **数据归一化读取**：把异构数据集（EgoVerse / EPIC-KITCHENS 等）统一成规范 schema
- **数据可视化检查**：轨迹、模态可用性、手部骨架、注视点、动作幅度
- **导出可训练格式**：VLA（LeRobot/RLDS 风格）与世界模型 action（视频 + 相机运动）

快速体验（无需下载真实数据）：

```bash
pip install -r requirements.txt
python -m ego_pipeline synth --out out/synthetic --episodes 3 --frames 30
python -m ego_pipeline run --config configs/example_egoverse.yaml
```

详见 [`ego_pipeline/README.md`](ego_pipeline/README.md)。

## License

CC BY 4.0
