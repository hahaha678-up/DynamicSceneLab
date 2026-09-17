# DynamicSceneLab

**室内动态人物场景生成与机器人导航闭环评测。**

基于已有 3DGS / Mesh 场景，接入 Isaac Sim、动态人物和 Nav2，将参数化场景与 CrowdES 生成轨迹转换为可执行的导航测试回合，记录交互指标，并围绕同一人物轨迹构造不同时间、速度条件的场景变体。

## 演示

https://github.com/user-attachments/assets/e236be10-a6e4-4bcd-8dc2-fa759f822936

35 秒 · 1080p · 原速关键片段：**多人物动态环境 → 遮挡出现与导航响应 → CrowdES 轨迹回放 → 同一父场景的三组变体**。

## 已实现的能力

- **动态场景接入**：在 Habitat-GS scene55 中配置 Carter 与多个人物，使用 Mesh / PhysX 进行物理与传感器计算，结合房间高斯和 LHM 人体高斯呈现动态画面。
- **导航闭环**：模拟 LiDAR 经 ROS2 进入局部 Costmap 与 Nav2，控制指令驱动 Carter；支持 Crossing、Occlusion 及轨迹回放场景。
- **生成式候选轨迹**：使用 CrowdES 官方预训练模型，适配房间可行走区域，检查候选轨迹的几何与运动有效性，转换世界坐标和时间后在 Isaac 中执行。
- **统一评测与参数化扩增**：记录目标到达、最小代理间隙、TTC、制动及停车时间；通过 `time_shift_s` 和 `speed_scale` 修改同一父轨迹的时序，保留场景来源、配置及父子关系。

```text
参数化事件 / CrowdES 候选轨迹
              ↓
      场景配置与轨迹有效性检查
              ↓
   Isaac + LiDAR → ROS2 / Nav2 → Carter
              ↓
      EpisodeEvaluator / 回合结果
              ↓
       间隙与交互响应分析

同一父轨迹 → 时间偏移 / 速度缩放 → 新场景 → 再评测
```

HuNavSim 社会力交互和生成轨迹回放是两种人物运动模式。CrowdES 回放保留原轨迹，不再交给 HuNavSim 改写。导航回合运行时记录物理、传感器和控制状态；演示中的 GS / LHM 画面依据记录状态离线渲染，不能据此推断实时渲染帧率。

## 代码结构

| 目录 | 内容 |
|---|---|
| [`mobile-navigation/`](mobile-navigation/) | 场景几何、坐标配置、Carter 与 GS 场景接入 |
| [`mobile-navigation/crowdes-b/`](mobile-navigation/crowdes-b/) | CrowdES 房间输入、推理、轨迹检查与回放适配 |
| [`hunav-core/crossing/`](hunav-core/crossing/) | 事件配置、Isaac Runner、ROS2 / Nav2 接口、评测与渲染 |
| [`lhm-human/`](lhm-human/) | 人体高斯服务、步态准备与场景合成 |
| [`examples/`](examples/) | 实际生成轨迹、可执行场景配置及精选回合指标 |

## 快速检查

评测器仅依赖 Python 标准库，可独立运行其测试，无需 Isaac、ROS 或 GPU：

```bash
python -B hunav-core/crossing/test_episode_evaluator.py
```

具备 NumPy 和 PyYAML 后，可运行全部 40 项轻量测试：

```bash
python -B -m unittest discover -s hunav-core/crossing -p 'test_*.py' -v
```

完整仿真依赖预先配置的 Linux、NVIDIA GPU、Isaac Sim、ROS2 Humble / Nav2，以及第三方场景与模型资源。仓库提供项目集成源码，**不包含商业仿真软件、模型权重或完整场景资产**。挂载关系、资源准备和运行命令见 [运行说明](docs/RUNNING.md)；指标定义见 [EpisodeEvaluator](hunav-core/crossing/EPISODE_EVALUATOR.md)。

## 已记录的结果

下表来自同一父场景的三组演示配置，均完成导航任务。参数由人工选定，程序按配置生成变体。

| 变体 | 时间偏移 | 速度倍率 | 最小代理间隙 |
|---|---:|---:|---:|
| 视频 A | +0.8 s | 1.15 | 0.258 m |
| 视频 B | 0.0 s | 1.00 | 0.158 m |
| 视频 C | −1.2 s | 0.80 | −0.378 m |

完整数值和其他演示回合见 [`examples/results.json`](examples/results.json)。视频 A / B / C 分别对应配置 `variant_c` / `variant_b` / `variant_a`，按测得的间隙由大到小排列。

### 当前范围

- 间隙基于二维圆形代理；负值表示代理重叠，不能直接等同于物理接触。TTC 使用当前相对速度外推。
- `rho = min_clearance_m - 0.25`；0.25 m 是固定版本的项目实验阈值，不是安全标准。无效回合不参与反馈。
- 多人物演示的主要交互指标针对指定人物与 Carter，尚不是所有人物对的聚合评测。
- 已实现参数化扩增；尚未实现 CEM 风险反馈搜索，也未证明大规模长尾发现效率。
- 首次检测时间、反应延迟与缺失的物理接触记录保持为空，不以推测补齐。

## 参考与致谢

感谢 [Re3Sim](https://github.com/InternRobotics/Re3Sim)、[CrowdES](https://github.com/InhwanBae/Crowd-Behavior-Generation)、[HuNavSim](https://github.com/robotics-upo/hunav_sim)、[LHM](https://github.com/aigc3d/LHM)、[Nav2](https://github.com/ros-navigation/navigation2) 与 [Habitat-GS](https://huggingface.co/datasets/RukawaY/gs_scenes) 等开源项目及资源为本项目提供支持。相关来源与许可见 [第三方资源说明](THIRD_PARTY.md)。
