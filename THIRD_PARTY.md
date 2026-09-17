# 第三方资源与贡献范围

本仓库发布动态导航项目的适配、编排、评测、渲染集成代码及少量示例轨迹。未随仓库重新分发完整上游源码、商业仿真软件、模型权重、人体先验或场景模型；使用外部资源须遵循各自项目和资源的许可。

| 上游 | 在本项目中的作用 |
|---|---|
| [Re3Sim](https://github.com/InternRobotics/Re3Sim) | Isaac 场景与 Gaussian Splatting 渲染基础；实验版本 `681da6f5d842f0de6c573a077926fe3722924708` |
| [CrowdES](https://github.com/InhwanBae/Crowd-Behavior-Generation) | 预训练人群发射与轨迹生成；实验源码版本 `0e663594604533d77ca98998bb93b63f1d509982`，ETH 预训练权重；本项目未训练该模型 |
| [HuNavSim](https://github.com/robotics-upo/hunav_sim) / [lightsfm](https://github.com/robotics-upo/lightsfm) | 人物社会力交互，具体版本见 `hunav-core/versions/` |
| [BehaviorTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP) | HuNav 行为树依赖 |
| [Nav2](https://github.com/ros-navigation/navigation2) / ROS2 Humble | 导航控制、局部地图与消息接口 |
| [LHM](https://github.com/aigc3d/LHM) | 姿态驱动的人体高斯表示与变形；人物服务调用官方模型与渲染代码 |
| [Habitat-GS 场景](https://huggingface.co/datasets/RukawaY/gs_scenes) | scene55 房间外观、几何和导航资源；使用版本 `ed92c666c45c16750b8bb062a835aac21326f51e` |
| NVIDIA Isaac Sim / Nova Carter | 物理仿真、传感器、机器人资产，按 NVIDIA 相应条款独立获取 |
| SMPL-X 与动作资源 | 人体动作参数与先验，按对应资源许可独立获取 |

上游模型、导航算法、社会力模型与重建场景不作为本项目原创成果。项目贡献集中于房间输入适配、坐标与时间转换、场景配置与人物执行、传感器导航闭环、统一指标及来源追踪、参数化变体和演示集成。

本次代码发布未新增统一开源许可证，也不改变任何第三方资源的许可；公开可见不代表对所有代码、资产和模型授予统一的再分发或商用许可。
