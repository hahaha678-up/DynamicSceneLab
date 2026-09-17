# 人物高斯与场景合成

本目录包含 LHM 模型调用、动作准备、人体高斯服务与房间合成适配。

- `run_avatar.py`：调用官方模型和预处理，生成人物状态。
- `prepare_habitat_walk.py`：将 SMPL-X 动作转换到目标人物状态。
- `serve_social_avatar.py`：根据姿态变形人体高斯，通过 Unix socket 提供渲染数据。
- `run_social_room.py`：原有 HuNav 人物交互和房间合成入口。

导航演示的离线渲染入口是 `../hunav-core/crossing/demo_render.py`。人物运动由 HuNav 或给定轨迹决定，LHM 负责外观和姿态变形。

模型、人体先验和动作须独立取得，遵循各自许可。见 [运行说明](../docs/RUNNING.md) 和 [第三方说明](../THIRD_PARTY.md)。
