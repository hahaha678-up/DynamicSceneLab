# CrowdES 房间轨迹适配

`prepare_room.py` 从几何准备可行走区域、分割输入、尺度与 NavMesh；`infer.py` 调用官方预训练 CrowdES，输出像素和世界坐标轨迹。

默认请求单人，使用预测密度图；不微调模型，不手工改写生成的空间轨迹。不同 seed 可能得到无效或无关候选，不能预先保证横穿或高风险。

- `analyze.py`：几何与运动检查。
- `collect_clean.py`：速度、加速度、停滞、网格覆盖和重复路径筛选。
- `replay_nine.py`：原有候选集的 Isaac 回放检查。
- `run_batch.sh` / `run_clean40.py`：原实验批次管理，需要既有依赖与指定运行目录。

`clean40` 是原实验目标批次名称，不表示已公开 40 条合格轨迹。演示使用的样本见 [examples](../../examples/)。

模型与第三方依赖未打包；配置与使用见 [运行说明](../../docs/RUNNING.md)。
