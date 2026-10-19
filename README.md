# SO-101 VLA 操作实验工具

新增 [模型、ROS2/C++ 与跨本体接线](docs/live-interfaces.md)：SmolVLA 检查点及前后处理器加载、带图像的异步快照、ROS2 类型化动作块/C++缓冲节点、MuJoCo命名关节FK后端。接口测试不代表真实模型收益、ROS2整机联调或真机实验已经完成。

已提交[合成缓存上的离线适配器训练样例](examples/synthetic-offline-training/README.md)，包含实际CPU训练生成的检查点、损失和来源元信息；未使用真实SmolVLA权重或机器人数据。

面向 SO-101 桌面操作的实验代码：将语言条件策略、模仿学习对照、操作数据检查和真机评测组织为可配置的运行流程。

当前包含实验入口、数据处理、因果历史窗口、可训练的历史/失败感知动作输出适配器、异步状态对齐调度原型及统一末端动作接口。仓库不包含历史真机数据、SmolVLA 微调权重或历史成功率的原始记录。新增代码与简历中既往实验的关系见 [研究证据清单](docs/research-evidence.md)。

## 包含什么

| 模块 | 内容 |
| --- | --- |
| `so101-run` | SO-101 校准、遥操作、数据采集、SmolVLA / ACT / Diffusion Policy 训练与真机执行 |
| `so101-data` | 轨迹检查、失败过滤、按 episode 划分、仅训练集的状态 / 动作统计 |
| `so101-report` | 按模型、检查点、任务、测试条件与训练种子汇总成功率及失败类型 |
| `so101-augment` | 基于训练集动作统计生成 clean / noisy / 因果 EMA filtered 三组离线标签 |
| `so101-train-adapter` | 用冻结策略的缓存动作训练 history / failure-aware 输出残差，按 episode 隔离验证集 |
| `so101-offline-demo` | CPU 合成数据端到端训练和虚拟时钟调度演示 |
| `so101-evidence` | 采集episode/frame分开计数、来源隔离、文件散列、配对初态和逐阶段实测延迟核查 |
| `runtime.py` | 单个在途推理、过期前缀丢弃、状态漂移拒绝、限幅和低水位重规划 |
| `embodiment.py` | body-frame EEF ΔPose + 绝对夹爪开度，显式 FK、限位和失败返回的数值 IK |
| `configs/` | 双视角相机、串口、任务、训练与评测的配置示例 |
| `docs/` | 数据格式、真机流程、对照实验设计和后续展示材料 |
| `tests/` | 数据泄漏、无效轨迹、命令生成与结果统计的离线测试 |

策略实现来自 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)，本仓库实现配置入口、数据处理与评测汇总。VLA 使用 SmolVLA，ACT 和 Diffusion Policy 用作操作策略对照。

## 快速开始：不需要机械臂

工具部分支持 Python 3.10+。在仓库根目录执行：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 使用：.\.venv\Scripts\Activate.ps1
python -m pip install -e .
so101-run train-smolvla --config configs/so101.example.json
python -m unittest discover -s tests -v
```

运行入口默认只打印参数列表；加入 `--execute` 才执行采集、训练或真机命令。上述预览不下载模型，不启动训练，也不连接串口。

## 历史、失败建模与异步执行

安装学习扩展后运行完整的离线演示：

```bash
python -m pip install -e ".[learning]"
so101-offline-demo --output-dir outputs/offline-smoke
python -m unittest discover -s tests -v
```

演示生成明确标记的合成缓存、训练/验证划分、适配器检查点、损失日志和虚拟时钟动作轨迹。它验证代码链路，不测量真机任务成功率或推理延迟。每次运行使用新的输出目录。

在真实冻结策略缓存上训练：

```bash
so101-train-adapter --cache data/cache.npz --split data/split.json \
  --metadata data/cache-metadata.json --mode failure_aware \
  --output-dir checkpoints/failure-aware-seed0 --epochs 20 --seed 0
```

缓存格式、归一化约定、LeRobot `predict_action_chunk` 包装和调度接线见 [实现与复现](docs/implementation.md)。动作输出残差与在 SmolVLA 隐层注入 token/adapter 是不同实现；当前 Python 调度原型也不是已经验证的 C++/ROS2 控制器。

历史采集数量、任务成功率和延迟应使用真实记录核查，入口与字段见[原始证据核查](docs/evidence-records.md)。该工具不生成实验数据，也不将记录一致性自动判为真机真实性。

日志格式演示见[明确标注的模拟日志](examples/synthetic-resume-logs/README.md)，包含按简历约数构造的任务结果、恢复记录、规划等待和虚构损失曲线。它们不属于历史实验结果；每条记录均标有synthetic，生成配方和统计计算可复现。

## 采集、训练与真机运行

使用 **Python 3.12** 创建环境，并安装固定版本的 LeRobot 0.6.1：

```bash
python -m pip install -e ".[robot]"
so101-run doctor
```

推荐将训练与硬件流程放在配置好 CUDA、串口和视频依赖的 Linux 环境。PyTorch / CUDA、FFmpeg 等系统依赖按 [LeRobot 安装说明](https://huggingface.co/docs/lerobot/v0.6.1/installation) 配置。

复制 `configs/so101.example.json` 为 `configs/so101.local.json`，修改 follower / leader 串口、相机索引、任务描述、数据目录和训练设备。Windows 串口通常填写 `COM3` 等实际端口。示例中的端口、采集数、步数与 batch size 是起步配置，均不表示已完成的实验。

```bash
so101-run calibrate-follower --config configs/so101.local.json --execute
so101-run calibrate-leader --config configs/so101.local.json --execute
so101-run teleoperate --config configs/so101.local.json --execute
so101-run record --config configs/so101.local.json --execute
so101-run train-smolvla --config configs/so101.local.json --execute
so101-run train-act --config configs/so101.local.json --execute
so101-run train-diffusion --config configs/so101.local.json --execute
so101-run evaluate-robot --config configs/so101.local.json --execute
```

先按 [SO-101 官方说明](https://huggingface.co/docs/lerobot/v0.6.1/so101) 完成电机配置与校准。相机名称、顺序和动作单位应与训练数据一致。训练使用本地 LeRobot 数据集；首次 SmolVLA 训练需要下载基础模型。每次训练 / 采集换用新的输出目录，脚本拒绝覆盖既有实验。`evaluate-robot` 使用本地训练后的 `pretrained_model` 目录，默认不上传数据或模型。

详细步骤见 [真机运行说明](docs/hardware.md)，数据处理见 [数据格式与处理](docs/data.md)，实验比较见 [实验方案](docs/experiments.md)。

已完成的检查与尚未验证部分见 [验证范围](docs/validation.md)。

## 离线数据处理

以下输入路径是你后续提供的文件，仓库不附带数据。中间 NPZ 格式与 LeRobot 训练数据格式分别说明于 [docs/data.md](docs/data.md)。

```bash
so101-data validate --manifest data/episodes.jsonl --output outputs/validation.json
so101-data filter --manifest data/episodes.jsonl --output outputs/selected.jsonl --rejected outputs/rejected.json
so101-data split --manifest outputs/selected.jsonl --output outputs/split.json --train-fraction 0.8 --seed 0
so101-data stats --manifest outputs/selected.jsonl --split outputs/split.json --output outputs/train-stats.json
so101-report --input data/evaluation.jsonl --output outputs/evaluation-summary.json
so101-augment --input data/episode.npz --train-stats outputs/train-stats.json --output-dir outputs/variants/episode --channels 0 1 2 --seed 0
```

离线统计按训练 episode 计算，验证 episode 不参与。统计 JSON 是本工具的中间产物，不会自动替换 LeRobot 的归一化统计，也不能直接作为训练数据输入。

加噪示例的通道索引仅展示参数格式，必须按实际动作定义修改并排除夹爪通道。每条 episode 单独初始化因果滤波；保持原始观测不变，因此这些是离线动作标签变体，不是重新执行后的物理轨迹。

## 仓库接下来放哪些材料

无需把原始数据全部公开。优先补充：硬件与相机摆位照片、任务描述、一次成功和一次失败的演示视频、脱敏后的运行配置、带样本量与检查点版本的评测表。对应目录规划和填写要求见 [展示材料清单](docs/artifacts.md)。

## 来源与版本

- [LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1)，上游 commit：`7e241bd630a3719a56157a497ce5d08f244784f1`
- [SmolVLA 文档](https://huggingface.co/docs/lerobot/v0.6.1/smolvla)
- [ACT 项目](https://github.com/tonyzhaozh/act)：双臂 Transfer Cube 是独立仿真基线，不等同于 SO-101 桌面任务。

本仓库不内嵌上游模型实现或复制其他项目的历史结果。SO-101 仿真场景、URDF / MJCF、专家策略与仿真数据到 LeRobot 的转换器尚需根据实际模型接入，见实验方案中的 Sim2Real 部分。
