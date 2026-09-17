# 人物交互与导航回合

本目录包含 HuNav bridge、事件配置执行器、ROS2 / Nav2 接口和轨迹回放编排。

| 入口 | 用途 |
|---|---|
| `run_episode.sh` | HuNav 人物 + 固定路线控制器的机制基线 |
| `run_nav2_episode.py` | HuNav 人物 + Nav2 控制的 Crossing 验证 |
| `run_ab_study.py` | 参数化 Crossing / Occlusion 的控制器对照 |
| `demo_suite.py` | 读取 `../examples/scenarios/`，执行回放、评测与离线渲染 |
| `bridge.py` / `hunav_client.py` | Unix socket 与官方 HuNav 服务适配 |
| `crossing/` | 场景规则、Isaac Runner、传感器接口、指标与渲染 |

HuNav 的 Regular/SFM 根据实际机器人状态更新人物；轨迹回放模式直接执行选定路径，两者分开使用。人物真值用于环境推进和离线评测，Nav2 障碍输入来自模拟 LiDAR。

运行前按 [运行说明](../docs/RUNNING.md) 准备容器、资源与共享目录。`versions/` 记录原实验依赖版本，第三方完整源码和构建产物未打包。
