# DynamicSceneLab

面向室内移动机器人的**动态仿真与数据生成系统**，用于复现人物横穿、遮挡出现等动态人机交互，支持重复测试与仿真数据采集。

系统基于 **Isaac Sim、3DGS 与 Nav2**，将真实场景外观、物理几何、动态高斯人物和机器人感知控制连接起来，提供从场景构建、候选轨迹执行到闭环评测与场景变体生成的完整流程。

## Demo

https://github.com/user-attachments/assets/e236be10-a6e4-4bcd-8dc2-fa759f822936

多人物仿真 · 遮挡交互 · CrowdES 轨迹回放 · 场景变体对比（35 秒，1080p）

## Features

### 1. Real2Sim 场景构建

- **3DGS＋Mesh 混合表示**：3DGS 承载真实室内场景外观，Mesh 提供碰撞与导航所需的物理几何。
- **尺度与坐标对齐**：统一场景、机器人和传感器的尺度与坐标，配置碰撞体及 NavMesh，将场景资产转化为可导航、可感知的仿真环境。

### 2. 动态人物与导航闭环

- **动态高斯人物**：接入 LHM 可动画人物，以统一状态关联人体动作、世界运动和感知／碰撞代理；支持 HuNav 社会力交互与指定轨迹回放两类运动模式。
- **感知—规划—控制闭环**：人物通过 LiDAR 进入 ROS2 / Local Costmap，由 Nav2 控制 Carter 响应；支持多人物、横穿与遮挡出现等交互场景，记录机器人减速、停车及恢复导航的实际行为。

### 3. 场景生成、质控与评测

- **候选场景与轨迹质控**：接入参数化事件及 CrowdES 预训练生成轨迹，完成图像坐标到世界坐标、模型帧到仿真时间的转换，并检查可行走区域、空间间隙和运动连续性。
- **统一记录与评测**：记录场景配置、传感器观测、人物与机器人轨迹、控制及运行结果；通过 `EpisodeEvaluator` 输出 TTC、最小代理间隙、制动及停车等指标。
- **父场景变体生成**：保持人物空间轨迹，通过时间偏移和速度缩放生成可执行变体，保留父场景与参数信息，支持同一运动模式下不同交互强度的重复测试与对比。

## Architecture

```text
参数化场景 / CrowdES 轨迹
          │
    场景配置与质量检查
          │
          ▼
Isaac Sim ── LiDAR / 状态 ──► ROS2 / Nav2
    ▲                             │
    └──────── 控制指令 ─────────────┘
          │
          ▼
回合记录 → EpisodeEvaluator → 交互指标
          │
父轨迹 + 时间偏移 / 速度缩放 → 场景变体
```

## Quick Start

### 离线评测器

仅需 Python 标准库：

```bash
git clone https://github.com/hahaha678-up/DynamicSceneLab.git
cd DynamicSceneLab
python -B hunav-core/crossing/test_episode_evaluator.py
```

安装 NumPy、PyYAML 后可运行全部 40 项离线测试：

```bash
python -B -m unittest discover -s hunav-core/crossing -p 'test_*.py' -v
```

### Isaac / Nav2 仿真

按 [运行文档](docs/RUNNING.md) 准备容器、场景和模型资源后，在 Linux 主机运行：

```bash
export DEMO_NAME=demo_my_run
python3 -B hunav-core/demo_suite.py --phase simulate --only occlusion
python3 -B hunav-core/demo_suite.py --phase render --only occlusion
```

`--only` 支持 `multi`、`crossing`、`occlusion`、`crowdes`、`variant_a`、`variant_b`、`variant_c`。输出位于 `mobile-navigation/output/<DEMO_NAME>/`。

## Repository

```text
mobile-navigation/             场景几何、坐标配置与 Carter
  crowdes-b/                   CrowdES 推理与轨迹检查
hunav-core/
  crossing/                    场景执行、ROS2 / Nav2、评测与渲染
lhm-human/                     动态高斯人物与步态
examples/
  scenarios/                   场景配置
  trajectories/                示例人物轨迹
  results.json                 实际回合结果
docs/                          环境与运行说明
```

## Documentation

- [环境配置与完整运行流程](docs/RUNNING.md)
- [场景示例与参数化变体](examples/README.md)
- [评测指标与数据格式](hunav-core/crossing/EPISODE_EVALUATOR.md)
- [演示回合结果](examples/results.json)

## Acknowledgements

感谢相关开源项目与社区的支持。详见[参考资料](THIRD_PARTY.md)。
