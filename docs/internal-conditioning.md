# 动作专家内部历史与失败条件

`internal_conditioning.py` 为 LeRobot 0.6.1 SmolVLA 的 `policy.model.action_time_mlp_out` 注册可移除 hook。因果状态窗口编码为隐向量，Failure / Progress 辅助头预测作为内部残差的输入，再加到动作时间嵌入。它直接参与基座 flow loss 的计算；原有 `adapter.py` 则是输出动作残差，两者作为明确的互斥模式保留。

没有插入额外 token，因而不扩展注意力掩码或动作块长度。残差投影零初始化，契约测试验证初始输出与基座逐值一致；真实预训练权重尚未加载，不能据此声称完成真实 SmolVLA 训练或任务收益评测。

失败标签和进度目标只进入辅助损失，不进入推理条件。推理条件来自历史状态及预测的 Failure / Progress。每次 reset 清空历史，缺少显式历史上下文时报错，禁止静默使用旧 episode。当前只支持单观测 batch 的流式包装，禁用 `compile_model`，梯度检查点重入或其他模型版本需另外验证。

## 训练入口

`train_internal.train` 接受已加载的 policy、conditioner、按 episode 标识的 `Example` 列表、完整 train/validation 划分和来源元数据。每个 Example 包含基座处理器生成的图像/状态/语言/动作 batch，以及同一处理空间的因果 history 和 mask；history 最后一帧必须与当前状态相等。未来动作与人工标签仅作为监督。

```python
from so101_lab.internal_conditioning import InternalConfig, InternalConditioner
from so101_lab.train_internal import Example, train

conditioner = InternalConditioner(InternalConfig(state_dim, expert_hidden_size)).to(device)
losses = train(policy, conditioner, processed_examples, episode_split, metadata,
               "checkpoints/new-internal-run", epochs=20)
```

变量由真实数据加载器提供。`expert_hidden_size` 必须等于原投影输出维度。元数据需声明 `source`、基座 `checkpoint_sha256` 和 `normalization_fit_episodes`；哈希声明由调用方核实，训练器不替调用方证明数据来源。基座参数冻结、使用其实际 forward 损失，仅优化内部模块；验证 episode 不参与训练。保存独立 adapter、逐 epoch 实际损失与参数，拒绝覆盖输出。损失不是任务成功率。

本次实际运行的是小型合成 flow 模型上的前向、反向、完整训练器和序列化契约测试，验证梯度到达内部模块、冻结基座不变、episode 隔离、零初始化与重置。未把玩具模型检查点标成 SmolVLA 权重，也没有生成伪造的收敛曲线。

## 部署入口

```python
stream = SmolVLAStream.from_pretrained(
    checkpoint, device="cuda:0", internal_adapter_path="checkpoints/new-internal-run/internal-adapter.pt")
```

调用方仍需保证基座、相机模态、状态顺序、预处理统计、动作表示与训练完全一致。不要将内部模块和输出残差同时安装。当前实现是逐观测条件残差，不是已完成的历史图像 token 模块，也没有论文级消融结论。

对应固定源码：[LeRobot v0.6.1 SmolVLA](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py)。
