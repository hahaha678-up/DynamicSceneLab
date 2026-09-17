# 场景与 Carter 接入

示例使用 Habitat-GS scene55。`scene_config.json` 保存高斯到 Isaac 世界坐标的变换、地面高度、导航路线及相机配置。

- `prepare_scene.py`：解析场景资产与 NavMesh，准备可行走网格和渲染数据。
- `check_geometry.py`：检查地面支撑及障碍间距。
- `run_demo.py`：静态房间中 Carter 的基础物理与视觉验证。
- `crowdes-b/`：预训练候选人物轨迹的输入、推理、验收与回放适配。

`run_demo.py` 使用固定路线控制器；动态导航闭环入口在 `../hunav-core/`。打开单独的 USD 不会自动启用外部高斯合成。

资源与容器路径见 [运行说明](../docs/RUNNING.md)，来源见 [第三方资源说明](../THIRD_PARTY.md)。
