# 实现、接口和复现边界

## 失败建模与因果历史

`history_windows(states, length)` 每次只接收一条 episode，窗口结束于当前帧。左侧零填充配合布尔 mask；不会读取未来状态或上一条 episode。`action_targets` 的未来动作只作训练监督，尾部越界动作被 mask。

`FailureEvent(start, stop, failure_type, recovered)` 使用 `[start,stop)`。失败类包括定位、抓取、执行偏移、目标变化、提前终止、超时、碰撞及其他。区间重叠或越界直接报错；未标注帧为 -1，不自动当成成功。`recovery_windows` 合并关键回放区间；`training_weights` 仅返回训练 episode 的索引及难例权重。

进度必须由数据生产方标注。当前不会把帧序号或总时长伪装为真实任务完成进度。合成演示中的进度专门标记为合成标签。

## 动作输出适配器

`HistoryFailureAdapter` 输入只有历史状态、历史 mask 和冻结策略的动作块。MLP 历史编码器产生失败 logits 和预测进度；failure-aware 模式将自己的预测反馈给残差动作头。人工失败/进度标签仅进入辅助损失，不进入 `forward`。输出头零初始化，初始输出与冻结策略相同。

三组比较口径：original 为冻结基座输出；history 为历史残差；failure-aware 为历史+预测失败/进度残差。`history` 训练的辅助损失权重为0。三组需使用相同基座、训练 episode、随机种子、调参预算与评测场景。这是一个新的可运行实现，不能据此认定复现了简历所述实验。

动作监督采用有效 horizon 点的均方误差，未知辅助标签跳过。训练按 episode 划分后，仅在训练窗口中做难例重采样；验证损失按未加权的有效动作点汇总，不报告机器人任务成功率。

## 冻结策略缓存契约

NPZ 使用 `allow_pickle=False`，包含且仅包含以下数组：

| 键 | 形状 / 类型 | 含义 |
|---|---|---|
| history_state | N,L,S / float | 基座预处理约定下的因果状态历史 |
| history_mask | N,L / bool | 每个历史帧是否有效，当前帧必须有效 |
| base_actions | N,H,A / float | 冻结基座对该观测的预测，禁止用目标动作冒充 |
| target_actions | N,H,A / float | 与基座相同动作空间中的监督 |
| action_mask | N,H / bool | episode 尾部有效监督 |
| failure_target | N / int64 | FAILURE_CLASSES 索引；-1 未知 |
| progress_target | N / float | [0,1] 标注进度；-1 未知 |
| episode_id | N / Unicode 或整数 | 整条 episode 的划分单位 |

`split.json` 为 `{"train":[...],"validation":[...]}`，必须无交集且完整覆盖缓存中的 episode。元数据需要 `base_checkpoint_sha256`、`normalization_training_episodes` 和 `action_space="upstream_processed_absolute_joint_targets"`。归一化 IDs 必须与训练划分一致。

这些元数据是缓存生产方的声明；训练器不能只凭一个散列证明缓存确由该 checkpoint 推理产生。正式实验应另外保留缓存导出脚本、基座配置、原始 episode 清单、预处理统计及 checkpoint 文件清单。`run.json` 保存输入缓存/划分散列、超参数和训练/验证 IDs。

## LeRobot 包装

`AdaptedChunkPolicy(base_policy, adapter)` 使用 LeRobot 风格的 `predict_action_chunk(batch)`，单机器人 batch。调用方先使用原基座预处理器，再把输出交给原动作后处理器；历史状态、base_actions 和 target_actions 的单位/归一化必须一致。每条 episode 开始调用 `reset()`。加载适配器使用 `weights_only=True`；它不替代 SmolVLA 权重和配置。

包装器在输出动作块上学习残差，不修改 SmolVLA 的 flow-matching 内部计算，也不声称与未公开的隐层注入方案等价。当前已在合成基座和 CPU 训练中验证，尚未加载真实 SmolVLA checkpoint 验证整个预处理/模型/后处理链路。

上游接口参考：[LeRobot 0.6.1 SmolVLA 源码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py)。

## 异步调度

`AsyncInference` 保持至多一个后台预测。控制线程用 `future.done()` 轮询，推理未完成时不等待，也不积压新的请求。策略异常被记录到 `last_error` 和计数器，不会把失败输出发送给调度器。

`StateAlignedBuffer` 中的动作是绝对目标。明确约定第 k 个样本时间为 `observation_timestamp + (k+1)*sample_dt`。接入上游策略时必须核对第一个动作的时间语义并显式转换，不能直接混用 k=0 对应当前帧的训练契约。所有时间戳都来自同一单调时钟；ROS 时间需在传输边界转换并处理复位。

新块按时间丢弃已过期前缀，仅替换未来区间；已有更早的未来动作可保留。请求序号倒退、观测过期或观测初态与当前状态的无穷范数偏差超过阈值时拒绝。队列长度有上限，低水位触发重规划。执行时再次检查动作/观测年龄；关节限位和每 tick 最大变化量由调用方显式提供。单位由调用方声明，默认阈值不是机器人安全标定值。

当前模块不打开串口、不发布 ROS2 命令，也不保证硬实时30Hz。纯 Python 调度测试不能支持简历中的 C++/ROS2、175→58ms 或真机成功率数字。要测延迟，需在硬件管线中记录采集、预处理、推理、通信、排队、执行各段，分别报告 wall-clock 阻塞与观测到执行的动作年龄。

## 跨本体动作

统一动作 `[dx,dy,dz,rotvec_x,rotvec_y,rotvec_z,gripper]`：平移为米、旋转为弧度，body frame 下 `T_target = T_current @ Delta`；夹爪是归一化绝对开度，0闭合/1张开。动作序列若逐步积分，不能当成相对同一观测的动作块解码。

`KinematicsAdapter` 需要调用方提供真实 FK 和关节限位。数值 IK 采用带阻尼的有限差分雅可比、关节区间和单次迭代步长限制；不收敛就抛出 `IKConvergenceError`，不返回一个被称为成功的近似命令。测试使用声明的笛卡尔运动学替身，未提供虚构的 Panda/UR5e/SO-101 URDF。

`robot_task_holdout` 按完整 `(robot_id,task)` 组合留出 episode。少样本适应中的 support 数据应再与 query 评测集隔离，不能把40条目标示范同时当作训练和最终测试。尚需真实场景资产、专家轨迹、机器人 FK/IK 标定和统一数据转换，才能验证跨本体迁移效果。
