# 真机运行说明

1. 按官方 SO-101 文档配置电机、装配和校准。`lerobot-find-port` 可辅助定位串口，`lerobot-find-cameras opencv` 可辅助确认相机。
2. 复制示例配置到 `configs/so101.local.json`，填写实际串口与相机索引。相机 `front` / `wrist` 是示例命名，模型训练与执行必须保持一致。
3. 先预览 `so101-run teleoperate --config configs/so101.local.json`，核对参数后加入 `--execute`。真机运行期间保留现场操作人员和设备急停。
4. `record` 保存遥操作示范。每条示范保持一致的语言任务描述；要覆盖不同任务，可以分别采集，再依据 LeRobot 数据工具合并。
5. 训练时为 SmolVLA、ACT、Diffusion Policy 使用相同的数据定义与评测任务；根据显存调整 batch size。示例训练步数不保证收敛。
6. 在 `evaluation.policy_path` 指向实际检查点目录；确认其中存在 `config.json`。评测输出换用新目录与新 repo_id。LeRobot 0.6.1 将策略执行移至 `lerobot-rollout`，本工具使用 `episodic` 模式保存回合，采集仍使用 `lerobot-record`。
7. 真机轨迹记录不自动产生任务成功标签。逐回合检查视频或现场记录，写入 `evaluation.jsonl`，再运行 `so101-report`。

更换语言表达、目标位置或物体组合时，在评测记录的 `condition` 中区分。中途扰动应记录发生时刻、扰动方式与失败阶段，成功条件应在评测前定义。

如 SmolVLA 的预训练相机键与采集相机键不同，填写 `training.rename_map`，评测时填写对应的 `evaluation.rename_map`。格式是 `{ "原始观测键": "模型观测键" }`，例如实际模型要求 camera1 时可把 `observation.images.front` 映射到 `observation.images.camera1`。以实际模型 `config.json` 为准，不默认假设相机对应关系。
