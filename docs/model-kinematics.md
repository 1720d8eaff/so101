# Panda / UR5e 固定模型接入

本次使用官方 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) commit `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35` 的 Panda 与 UR5e 模型实际运行运动学。模型保留上游许可，本仓库提供模型定位、接入代码与逐资产散列，不重新内嵌模型网格。

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git artifacts/menagerie
git -C artifacts/menagerie sparse-checkout set franka_emika_panda universal_robots_ur5e
git -C artifacts/menagerie checkout 62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35
python -m so101_lab.model_check --menagerie artifacts/menagerie --output artifacts/model-check.json
```

| 机器人 | 模型与关节 | 末端 |
|---|---|---|
| Panda | franka_emika_panda/panda.xml，joint1至joint7 | hand body坐标系 |
| UR5e | universal_robots_ur5e/ur5e.xml，6个命名机械臂关节 | attachment_site |

检查从各模型的home keyframe附近生成32组有限位的当前/目标关节，经真实MuJoCo FK转换为统一EEF body-frame增量，再用数值IK还原。逐条检查位置误差≤0.1mm、旋转误差≤0.001rad，记录全部误差和模型资产SHA256。Panda为冗余机械臂，因此检查末端位姿而不是强求关节解相同。

结果见[本次64组记录](../examples/model-verification/model-check.json)。这是局部可达位姿回环验证，没有接触物理任务、碰撞规划、抓取成功率、策略训练或跨本体泛化数据。夹爪值只通过统一动作字段往返，不能证明真实夹爪执行。Panda hand body也不是经过真机工具标定的TCP。
