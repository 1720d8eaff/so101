# 验证范围

本机已完成：

- Python 包可编辑安装，四个命令入口可调用。
- 离线测试覆盖原有数据/启动流程，以及因果历史、失败标注、训练集难例采样、异步推理、过期动作、状态漂移、关节限幅、SE(3) 编解码、IK 不收敛、组合留出、适配器训练和检查点往返。精确数量以 `python -m unittest discover -s tests -v` 输出为准；未安装学习扩展时适配器测试会跳过。
- 命令字段对照 LeRobot 0.6.1 的配置与源码；录制采用 `lerobot-record`，策略运行采用 `lerobot-rollout` 的 episodic 模式。

测试中的数值与 episode 仅在临时目录生成，用来验证程序行为，不是实验数据。GitHub Actions 使用 CPU 环境执行同一套离线测试。

学习部分通过 `pip install -e ".[learning]"` 安装；`so101-offline-demo --output-dir outputs/smoke-run` 提供端到端合成数据演示。Python调度器的虚拟时间测试不测量物理延迟，也未验证30Hz硬实时。

追加验证：62项Python测试通过（本机安装torch和MuJoCo）；内部动作时间嵌入条件模块与完整训练器在小型合成flow模型上通过前向/反向/保存测试；完整ROS2 Jazzy节点严格编译并通过合成消息联调；官方Panda与UR5e模型各32次FK/IK回环通过。详见live-interfaces.md、internal-conditioning.md及model-kinematics.md。

尚未验证：实际串口通信、相机采集、GPU训练、真实SmolVLA权重前向或收敛、真机执行、跨本体任务与Sim2Real。已保存的输出适配器权重来自合成缓存训练；没有真机任务成功率结论。硬件 / 模型路径入口通过配置检查和参数核对，不代表已经在机器人上运行。
