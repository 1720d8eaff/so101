# 数据格式与处理

## 1. 采集与训练格式

`so101-run record` 调用 LeRobot 0.6.1 采集流程，写入其原生数据集目录，包括 `meta/info.json`、分片数据和视频等。`so101-run train-*` 使用这个目录。检查 `dataset.root` 与 `dataset.repo_id` 对应同一份数据。

## 2. 离线轨迹中间格式

本工具的轨迹校验与筛选接口使用每条轨迹一个 NPZ 文件，至少包含：

| NPZ 字段 | 形状 | 含义 |
| --- | --- | --- |
| `observation_state` | `[T, state_dim]` | 状态向量，关节顺序与单位由你的导出器明确 |
| `action` | `[T, action_dim]` | 同时刻的动作标签，需明确位置 / 速度 / 增量等定义 |

这里只校验数值、维度与时间长度，不假设 SO-101 的某个通道必然是夹爪，也不推断动作单位。RGB 视频、末端位姿可另外保存；当前工具不验证 RGB 时间对齐，也不会完成 LeRobot 与 NPZ 的自动互转。

`episodes.jsonl` 每行记录一个 episode。字段定义如下，填写真实值，不把下面的字段说明当作数据：

| 字段 | 类型 | 要求 |
| --- | --- | --- |
| `episode_id` | 字符串或整数 | 全文件唯一；建议包含任务和运行标识 |
| `npz_path` | 字符串 | 相对 manifest 所在目录的路径，或绝对路径 |
| `success` | 布尔值 | 经任务成功条件验证 |
| `timeout` | 布尔值 | 是否超时 |
| `collision` | 布尔值 | 是否出现任务定义中的无效碰撞 |
| `abnormal_action` | 布尔值 | 是否违反预先定义的动作规则 |
| `task` / `source` | 字符串，可选 | 语言任务与 sim / real 来源 |

缺失的失败标记按“未知”处理，默认不会进入有效轨迹集合。`--keep-unsuccessful` 仅放宽 success 条件，不放宽碰撞、超时、异常动作检查。输出文件默认必须是新路径，避免覆盖历史记录。

## 3. 固定划分与统计

`split` 按完整 episode 划分，使用固定种子和排序后的 ID；改变 manifest 行顺序不会改变结果。`stats` 检查训练 / 验证完全覆盖数据且没有重叠，以逐 episode 合并矩的方式计算均值、总体标准差、最小值和最大值，只读取训练轨迹。

常量通道的标准差保留为 0；后续归一化时自行采用经过记录的 epsilon。不同来源的动作单位、维度或相机定义不一致时，应先做转换，再合并数据。

LeRobot `training.episodes` 使用其自身的整数 episode 索引，不是任意 manifest 字符串 ID。选择训练索引时必须核对映射。LeRobot 可能使用数据集元数据里的全量统计，因此仅设置 `dataset.episodes` 不足以证明归一化没有泄漏。严格对照应建立仅含训练 episodes 的原生数据集并重新计算统计；本仓库的统计 JSON 尚未接入 LeRobot processor。

## 4. 评测记录

`evaluation.jsonl` 每行对应一次实际执行，要求：`policy`、`checkpoint`、`task`、`condition`、`seed`、`trial_id`、`success`、`failure_type`。

- `seed` 表示训练种子；不同训练种子分组报告。
- `condition` 可填写 `fixed`、`language_variant`、`position_shift`、`object_combination`、`midway_perturbation` 等你实际使用的测试条件。
- `success` 必须为布尔值，不从视频文件存在或机器人未报错推断成功。
- 成功时 `failure_type` 为 `null`；失败时填写 `localization`、`grasp`、`execution_deviation`、`early_stop`、`timeout`、`collision` 或 `other`。
- 同一模型、检查点、任务、条件与训练种子下的 `trial_id` 不得重复。

评测汇总不计算不存在的标签，不把不同训练种子的回合混作独立训练实验。
