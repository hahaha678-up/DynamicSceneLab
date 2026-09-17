# DynamicSceneLab

面向室内移动机器人的动态仿真与数据生成系统。基于 **3DGS、Isaac Sim 和 Nav2**，支持动态人物场景构建、生成轨迹执行、导航闭环评测与参数化场景扩增。

## Demo

https://github.com/user-attachments/assets/e236be10-a6e4-4bcd-8dc2-fa759f822936

多人物仿真 · 遮挡交互 · CrowdES 轨迹回放 · 场景变体对比（35 秒，1080p）

## Features

- **Real2Sim 场景**：3DGS 外观与 Mesh 物理几何结合，统一场景、机器人和传感器坐标，支持碰撞、导航及多视角渲染。
- **动态人物**：可动画高斯人物与感知 / 碰撞代理同步；支持 HuNav 社会力交互、指定轨迹回放、横穿和遮挡出现。
- **导航闭环**：LiDAR → ROS2 → Local Costmap → Nav2 → Carter，记录人物运动引起的减速、停车与恢复导航。
- **数据生成与评测**：支持参数化场景和 CrowdES 生成轨迹，提供轨迹有效性检查、回合记录、TTC / 间隙评测及时间 / 速度变体生成。

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
