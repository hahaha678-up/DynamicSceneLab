# 场景执行与闭环评测

| 模块 | 职责 |
|---|---|
| `spec.py` / `scenarios/` | Crossing 参数与人物起终点、速度、出发时间 |
| `occlusion.py` / `visibility.py` | 遮挡规则、射线可见性和事件检查 |
| `runner.py` | 在 Isaac 中推进 Carter 与 HuNav 人物并记录状态 |
| `demo_tracks.py` / `demo_run.py` | CSV / 点列轨迹、时间偏移、速度缩放、多人物执行 |
| `carter_ros.py` / `ros_endpoint.py` | 模拟 LiDAR、状态与 ROS2 控制指令交换 |
| `nav2_session.py` / `nav2_params.yaml` | Nav2 控制及局部路径记录 |
| `fixed_session.py` | 固定控制器对照 |
| `episode_evaluator.py` | 独立离线评测与数据质量判断 |
| `demo_render.py` / `render_episode.py` | 依据记录状态渲染人物、房间和机器人 |
| `demo_compose.py` | 原始长版多视图视频拼接 |

在仓库根目录运行轻量测试：

```bash
python -B -m unittest discover -s hunav-core/crossing -p 'test_*.py' -v
```

事件与遮挡测试需要 NumPy、PyYAML；评测器测试仅依赖标准库。

`trajectory.jsonl` 保存仿真时间、Carter 与指定人物状态；`observations.jsonl` 保存传感器和控制快照。多人物回放在 `people` 中保存全部人物，但当前配对指标针对指定人物。

完整指标见 [EPISODE_EVALUATOR.md](EPISODE_EVALUATOR.md)，运行环境见 [RUNNING.md](../../docs/RUNNING.md)。
