# 示例场景与轨迹

- `trajectories/person_*.csv`：原有 CrowdES 预训练采样中用于演示的轨迹，`time_s` 为秒、`x/y` 为 Isaac 世界坐标米；未手工改写轨迹形状。
- `trajectories/crossing_reference.csv`：从已记录的参数化 Crossing 回合提取的实际人物轨迹，用于多人物演示和 Crossing 参考。
- `trajectories/parent.csv`：从同一父回合提取的实际人物轨迹，用于三组时间 / 速度变体。
- `scenarios/`：`demo_suite.py` 读取的场景配置。路径使用容器内的 `/repo/examples/`；CSV 由 `demo_tracks.Track` 读取。
- `results.json`：原完整环境中六个实际演示回合的精选指标，不是重新运行示例后的保证值，也不是大规模基准成绩。

## 变体定义

保持空间轨迹不变，用以下公式变换轨迹时间：

```text
t_new = t_start + time_shift_s + (t_original - t_start) / speed_scale
```

三组参数为手工选定的演示配置，执行器自动完成轨迹变换与回放。配置保留来源和父场景 ID；本版本没有风险反馈驱动的参数搜索。

源轨迹校验与完整实验记录保留在原实验环境；仓库只提供运行示例所需的数据，不携带模型、房间资产或完整传感器日志。
