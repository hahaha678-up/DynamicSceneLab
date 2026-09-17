# 运行说明

## 两种使用范围

1. **离线评测与源码检查**：Python 标准库即可运行评测器；NumPy、PyYAML 用于事件与几何测试。
2. **完整仿真与渲染**：需要已有 Isaac / ROS2 / 模型环境与下列资源。本次发布没有重建全新的容器镜像；轻量测试已执行，演示来自原有完整环境的真实记录。

本仓库是 Re3Sim 场景上的集成层，保留实验使用的容器路径 `/repo`、`/work`。这些是容器内路径，不是用户机器的固定目录。

## 外部资源

| 资源 | 项目内预期位置或用途 |
|---|---|
| Re3Sim `681da6f5d842f0de6c573a077926fe3722924708` | 将其 `re3sim` 目录链接或挂载到仓库根目录 `re3sim/`，提供 `gaussian_splatting` 渲染模块；保留上游许可证 |
| Habitat-GS scene55 | `mobile-navigation/assets/scene55/scene55.gs.ply`、`scene55.mesh.ply`、`scene55.navmesh` |
| NVIDIA Nova Carter | `mobile-navigation/assets/` 内的本地 USD 与相对依赖；按 NVIDIA 许可取得，具体根路径见 `run_demo.py` / `demo_run.py` |
| CrowdES `0e663594604533d77ca98998bb93b63f1d509982` | 源码放置在 `mobile-navigation/crowdes-b/vendor/`；ETH 预训练包解压后位于 `models/checkpoints/eth/`，含 emitter_pre、emitter、simulator |
| LHM | 源码 `lhm-human/LHM/`；模型配置和权重位于 `lhm-human/downloads/config.json`、`model.safetensors`；人体先验按上游要求取得 |
| 人物动作 | 合法取得的 SMPL-X 动作放入 `lhm-human/downloads/habitat_walk.npz`；`run_avatar.py`、`prepare_habitat_walk.py` 生成渲染服务所需状态 |
| HuNavSim / lightsfm / BehaviorTree.CPP | 编译到 `hunav-core/ws/` 和 `vendor/` 对应位置；版本见 `hunav-core/versions/`，行为树见 `behavior_trees/` |

模型、人体先验、机器人资产与场景资产均未随仓库分发。资源来源见 [第三方说明](../THIRD_PARTY.md)。原实验对 HuNav 构建配置作过适配：增加 `behavior_tree_dir` 参数与可选 Groot 开关、显式声明 tf2_geometry_msgs、去掉未使用的 Nav2 依赖；未改写 Regular/SFM 力计算。该外部构建环境不是本仓库的一键安装内容。

## 环境与挂载约定

实际导航实验使用 Re3Sim 的 Isaac Sim 4.0 容器环境、Python 3.10，以及独立的 ROS2 Humble / Nav2 容器；不要把本机其他 Isaac Lab 环境的版本直接当成本项目运行版本。LHM 和 CrowdES 各使用其已配置依赖。

| 既有容器名称 | 挂载关系 |
|---|---|
| `re3sim-mobile-scene64` | 仓库根目录 → `/repo`；`mobile-navigation/` → `/work` |
| `re3sim-nav2` | `hunav-core/` → `/work`；安装 ROS2 Humble、Nav2 及消息依赖 |
| `re3sim-hunav-core` | `hunav-core/` → `/work`；HuNav 工作空间已编译 |
| `re3sim-lhm-human` | `lhm-human/` → `/work`；已有 `/work/venv/bin/python` |

容器名称沿用早期实验，但当前场景是 **scene55**。运行入口使用 Unix socket 交换状态，相关目录必须可写并共享。ROS2 与模型服务需要按各自环境配置，代码没有把人物真值直接注入被遮挡障碍的导航感知。

`python_env.sh` 沿用 Re3Sim 镜像中的 `/root/miniconda` 和 `/isaac-sim` 环境。如果使用其他镜像，需调整这个入口。主机脚本继承当前 Docker context / `DOCKER_HOST`，不再固定个人服务器的 socket；Shell 入口可通过 `ISAAC_DOCKER_ENV` 指定已有环境脚本。

## 1. 准备场景几何

下列命令均要求相应容器与资产已经准备好，在仓库根目录的 Linux 主机上执行：

```bash
mkdir -p hunav-core/runtime mobile-navigation/output lhm-human/output
docker exec re3sim-mobile-scene64 bash /work/python_env.sh /work/prepare_scene.py --scene scene55
docker exec re3sim-mobile-scene64 bash /work/python_env.sh /work/check_geometry.py
```

`scene_config.json` 保存当前房间的 GS → 世界坐标变换与路线；`prepare_scene.py` 写出可行走网格和几何缓存。更换房间时必须重新准备几何、相机、Carter 路线和坐标变换，不能直接套用 scene55 的数值。

## 2. 生成 CrowdES 候选轨迹

```bash
docker exec re3sim-mobile-scene64 bash /work/crowdes-b/python_env.sh /work/crowdes-b/prepare_room.py --exact-navmesh
docker exec re3sim-mobile-scene64 bash /work/crowdes-b/python_env.sh /work/crowdes-b/infer.py --scene scene55 --run scene55_trial --seeds 20 --first-seed 0 --seconds 35
docker exec re3sim-mobile-scene64 bash /work/crowdes-b/python_env.sh /work/crowdes-b/analyze.py scene55_trial
```

推理默认请求单人、使用预测密度图，不传 GT 密度图；保留预训练速度控制。世界坐标转换使用输入单应变换和原点偏移，时间由模型 frame / fps 得到。`collect_clean.py` 提供更严格的几何覆盖、速度、停滞和重复轨迹检查；只有通过实际检查的候选才适合继续回放，不能把请求数量当成合格数量。

## 3. 执行导航回合

已有示例使用 CSV 或显式 `[time_s, x_m, y_m]` 点列，因此可以跳过重新推理。先确保 `re3sim-nav2`、`re3sim-mobile-scene64` 已运行：

```bash
export DEMO_NAME=demo_my_run
python3 -B hunav-core/demo_suite.py --phase simulate --only occlusion
python3 -B hunav-core/demo_suite.py --phase simulate --only crowdes
```

场景配置位于 `examples/scenarios/`。`--only` 还可选择 `multi`、`crossing`、`variant_a`、`variant_b`、`variant_c`。省略 `--only` 会执行全部示例；新的配置应使用新的 `DEMO_NAME`，避免混合已有记录。初始化时间、轨迹和实际响应以新回合输出为准，不保证跨设备逐帧一致。

HuNav 社会力模式的独立入口：

```bash
bash hunav-core/run_episode.sh crossing/scenarios/gap0_person1.0.yaml
python3 -B hunav-core/run_nav2_episode.py
```

前者为固定控制器机制基线，后者使用 Nav2；两者依赖 HuNav bridge 和既有容器。不要同时运行多个占用相同 socket 的 episode。

## 4. 离线评测

```bash
python3 -B hunav-core/crossing/episode_evaluator.py --episode mobile-navigation/output/demo_my_run/occlusion --summary mobile-navigation/output/demo_my_run/summary.csv
```

`demo_suite.py` 已在回合结束后调用同一评测器。上面命令适合独立检查回合；需要替换已有自动生成结果时显式使用 `--replace-generated`。评测器读取 `trajectory.jsonl`、`observations.jsonl`、`metrics.json`、`resolved.json`、可选 `scenario.json` 和 Nav2 配置，写出 `result.json` 与汇总 CSV。完整数据契约见 [评测器文档](../hunav-core/crossing/EPISODE_EVALUATOR.md)。

## 5. 渲染记录

人物模型与动作缓存准备好之后：

```bash
python3 -B hunav-core/demo_suite.py --phase render --only occlusion
```

`demo_render.py` 依据实际回合状态渲染 GS 房间、LHM 人物与 Carter，可输出宏观视角及机器人视角。原始 `--phase compose` 脚本用于全部场景的长版拼接，要求所有示例均已完成仿真和渲染；首页的 35 秒版本是这些记录的人工剪辑，删去了独立 Crossing 展示段及等待片段。

## 验证范围

- 发布副本的 40 项离线测试通过，包括事件配置、遮挡几何、TTC、扫掠间隙、坏数据、时序缺失、终止状态及 CLI 幂等性。
- 已检查示例轨迹、配置引用和源码语法；首页视频在原有完整环境录制。
- 发布时没有新建环境或重新运行全部 GPU 仿真；完整环境从零安装的复现尚未验收。
