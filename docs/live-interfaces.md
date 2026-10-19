# 模型、ROS2/C++ 与跨本体接口

这些是2026-10-19新增接线代码；不追认简历中的历史训练、采集数量、成功率、样本效率或延迟结果。原有明确标注的合成日志继续保留，不能作为真机实验记录。

## SmolVLA与历史/失败感知适配器

```python
from so101_lab.smolvla_live import SmolVLAStream, SensorSnapshot, async_runner
from so101_lab.runtime import StateAlignedBuffer

stream = SmolVLAStream.from_pretrained(checkpoint, device="cuda:0", adapter_path=adapter_checkpoint)
stream.reset()
# batch来自同步观测，包含该checkpoint所需图像、状态和语言张量。
snapshot = SensorSnapshot(monotonic_timestamp, physical_joint_state, batch)
actions = stream(snapshot)
```

按LeRobot 0.6.1接口加载SmolVLA与checkpoint内的原始前后处理器。输出残差适配器作用于模型处理空间；每个动作经过原动作后处理器，恢复真实硬件动作单位。现有适配器契约要求处理空间中的绝对关节目标，带相对动作配置的模型不自动兼容。适配器训练时的基座、预处理统计与部署checkpoint必须一致，当前声明检查不能代替权重来源核实。

传感器采集方负责形成检查点要求的batch：相机键、RGB布局、语言及状态顺序都需与训练一致。本模块不猜测SO-101的弧度、角度或电机归一化值，也不打开串口。`SensorSnapshot`复制整个图像/状态batch，防止后台推理读到下一帧。`async_runner`与原`StateAlignedBuffer`配合，保留带图像快照并确保至多一个在途推理。

```python
buffer = StateAlignedBuffer(initial_action, lower, upper, max_step)
with async_runner(stream, buffer, sample_dt=sample_dt, clock=monotonic_clock) as runner:
    control_step = runner.advance(snapshot)
    # 仅将control_step.action交给经过校验的硬件驱动。
```

Python和C++调度约定第k个动作在`observation_time + (k+1)*sample_dt`执行。模型训练若把第0个目标解释为当前时刻，调用方必须显式偏移；不得把两种时间语义直接混用。本次没有加载真实SmolVLA权重；模型包装通过模拟policy/processor测试。原有适配器在合成缓存上的CPU训练属于实际软件训练，不等于真机策略有效。

## C++核心与ROS2

内部动作专家条件模块、真实基座损失训练入口及其验证边界，另见[内部条件实现](internal-conditioning.md)。部署时使用 `internal_adapter_path`，与原 `adapter_path` 输出残差互斥。

`ros2/so101_control/include/so101_control/buffer.hpp`提供有界动作缓冲：乱序、陈旧观测、状态漂移拒绝，按时间丢弃前缀与替换未来动作，执行时再次检查年龄、限位及速度；低水位请求重规划。核心不依赖ROS，可独立编译运行测试：

```bash
g++ -std=c++17 -Wall -Wextra -Werror -I ros2/so101_control/include \
  ros2/so101_control/test/buffer_test.cpp -o /tmp/so101-buffer-test
/tmp/so101-buffer-test
```

在ROS2 Jazzy工作区构建：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2/so101_control --packages-select so101_control
source install/setup.bash
ros2 run so101_control control_node --ros-args --params-file /path/to/calibrated-robot.yaml
```

参数文件必须提供`joint_names`、`lower`、`upper`和`max_speed`数组。上限、下限、状态、模型输出和速度单位由接线方统一；不提供可误用于真机的虚构标定值。

| Topic | 类型 | 语义 |
|---|---|---|
| `joint_states` | sensor_msgs/JointState | 带ROS时间戳的真实状态；按配置名称重排 |
| `action_chunks` | so101_control/ActionChunk | 观测ROS时间、请求序号、采样周期、参考状态和绝对动作块 |
| `joint_targets` | trajectory_msgs/JointTrajectory | 约30Hz定时器发布的单点目标；需另接真实驱动 |
| `request_replan` | std_msgs/Bool | 缓冲低水位信号 |

Python端`smolvla_live.ros_chunk`可将原`runtime.ActionChunk`转为新消息；观测的单调时间必须由调用方显式转为与JointState一致的ROS时钟。ROS时间回退会丢弃缓冲并重新等待状态。状态过期时停止发新目标并重新建缓冲；下游驱动仍需要独立看门狗，停止发布不等于已实现安全停机。约30Hz仅指配置的定时器目标，没有端到端硬实时或175→58ms实测结论。

本次已在WSL Ubuntu 24.04 / ROS2 Jazzy完整构建节点，启用 `-Wall -Wextra -Werror`。使用 `rmw_cyclonedds_cpp` 在隔离domain 198和专用namespace联调：实际传输ActionChunk、接收限速目标、核实非有限输入被拒绝及状态过期停止发布。记录见[ROS2消息测试](../examples/ros2-verification/smoke.json)。Fast DDS在本机未完成节点发现，未将该失败记为通过。新增CI也执行完整构建与消息测试。没有真机驱动或端到端延迟/硬实时验证。

```bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp python3 ros2/so101_control/test/ros_smoke.py \
  install/so101_control/lib/so101_control/control_node
```

## 跨本体MuJoCo后端

```python
from so101_lab.mujoco_backend import MujocoKinematics

fk = MujocoKinematics(model_path, joint_names, eef_site=site_name)
adapter = fk.adapter()
unified = adapter.encode(current_joints, target_joints, gripper_opening)
target_joints, gripper = adapter.decode(current_joints, unified)
```

调用方提供真实MJCF/URDF与唯一关节名称，选择末端site或body；后端读取实际运动学和关节限位，运行MuJoCo FK，再使用已有数值IK。当前仅支持明确有限位的单自由度hinge/slide关节。一个后端对象按单线程使用；并行采集为每个worker创建独立实例。

统一EEF动作仍为body-frame米/弧度和绝对夹爪开度。新增固定官方模型的Panda/UR5e接入和各32次实际MuJoCo FK/IK回环，具体模型版本、末端与误差见[模型验证](model-kinematics.md)。夹爪开度仍是表示字段，没有运行夹爪抓取。SO-101模型、跨本体任务资产和专家策略仍待实现；该接口不包含8,000条候选/2,500条有效任务轨迹，也不证明31→49%或40条真机示范后43→59%。

参考：[LeRobot 0.6.1 SmolVLA](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py)、[原处理器加载工厂](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/factory.py)。

**合成结果仅用于流程演示，不能作为历史模型性能、仿真迁移或真机实验成果。**
