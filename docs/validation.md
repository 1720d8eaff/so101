# 验证范围

本机已完成：

- Python 包可编辑安装，四个命令入口可调用。
- 离线测试覆盖原有数据/启动流程，以及因果历史、失败标注、训练集难例采样、异步推理、过期动作、状态漂移、关节限幅、SE(3) 编解码、IK 不收敛、组合留出、适配器训练和检查点往返。精确数量以 `python -m unittest discover -s tests -v` 输出为准；未安装学习扩展时适配器测试会跳过。
- 命令字段对照 LeRobot 0.6.1 的配置与源码；录制采用 `lerobot-record`，策略运行采用 `lerobot-rollout` 的 episodic 模式。

测试中的数值与 episode 仅在临时目录生成，用来验证程序行为，不是实验数据。GitHub Actions 使用 CPU 环境执行同一套离线测试。

学习部分通过 `pip install -e ".[learning]"` 安装；`so101-offline-demo --output-dir outputs/smoke-run` 提供端到端合成数据演示。Python调度器的虚拟时间测试不测量物理延迟，也未验证30Hz硬实时。

尚未验证：实际串口通信、相机采集、GPU 训练、模型收敛、真机执行、Sim2Real 场景接入。仓库没有训练权重或任务成功率结论。硬件 / 模型路径入口通过配置检查和参数核对，不代表已经在机器人上运行。
