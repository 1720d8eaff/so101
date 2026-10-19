# 原始证据核查入口

`so101-evidence`只核查调用方提供的记录，不生成示范、Rollout、任务标签或延迟数值。没有输入时不能生成通过的历史实验报告。输出记录输入文件SHA256；real_robot、simulation和synthetic必须分开。

## 采集记录：episode与frame分别计数

```bash
so101-evidence collection --input private-data/episodes.jsonl \
  --source real_robot --expected-total 360 --expected-accepted 360 \
  --output outputs/collection-audit.json
```

期望数量是要核查的声明，不是默认已存在的数量；不匹配就失败。若360是采集总量而非全部有效，应按实际情况填写。G1的174条真机声明可用同一工具核查`--expected-accepted 174`；220若只是规划，不能当成已采集总数。

每行一个episode，必须包含episode_id、source、robot、task、operator_role、started_at、ended_at、frames、accepted和artifacts。日期需带时区，结束晚于开始；帧数为正整数；accepted必须经过人工复核。排除条目须有exclusion_reason。操作者字段可用匿名角色，区分本人、队友或设备自动运行。

artifacts至少包含trajectory与calibration，可包含video；每个条目为`{"path":"相对路径","sha256":"64位小写十六进制"}`。文件需位于manifest目录内，工具逐块核对散列，公共标定文件只核查一次。示范数按唯一episode计，帧数另列，不能把174帧计为174条episode。视频覆盖量单独报告。

文件存在和散列匹配只支持材料一致性，不能证明格式、标定有效性、本人贡献或真机真实性；仍需数据读取器、现场演示与人工审查。保密材料可在本地核查后只分享脱敏统计与散列；工具不自动上传原始文件。不可读取的私有引用不能标为文件核查通过。

## 逐次配对评测

```bash
so101-evidence trials --input private-data/trials.jsonl --source real_robot \
  --output outputs/trial-audit.json
```

每行需包含so101-report字段：policy、checkpoint、task、condition、seed（训练种子）、trial_id、人工复核的success和最终失败类型failure_type；另需scene_id（字符串）、repeat、environment_seed、checkpoint_sha256、protocol_sha256、initial_condition_sha256及source。

同一训练种子、任务、条件、场景和repeat的全部policy必须齐全，并使用相同环境种子、协议和初态散列，否则拒绝。各策略可有不同checkpoint，来源需另外保留配置与训练日志。工具按policy/checkpoint/任务/条件/训练种子汇总。配对检查通过不表示统计显著；还需逐对差值、聚类区间和预先冻结的分析协议。

恢复标注使用recognized_failure和autonomous_recovery两个布尔字段。分母为全部已识别失败，分子为其中自主恢复次数；不尝试恢复也留在分母。最终任务成功与中途恢复是两个不同终点。

## 实测延迟

```bash
so101-evidence latency --input private-data/control-ticks.jsonl --source real_robot \
  --target-hz 30 --output outputs/latency-audit.json
```

每行一个真实控制tick：source、mode、trial_id、step、clock_id、control_start_ns、control_end_ns、blocked_ns，以及与执行动作对应的capture_ns、preprocess_end_ns、inference_start_ns、inference_end_ns、send_ns、execute_ns。所有ns字段为同一时钟域下的非负整数；跨进程/设备时需先转换并保留时钟校准记录。

可额外提供episode_start_ns和episode_end_ns，同一trial边界必须一致并涵盖tick。未提供时不会推断任务时长。任务时长按trial统计，不能让长episode因帧数多而权重更大。

输出分别报告每tick阻塞、tick耗时、采集到执行的动作年龄、纯推理时间的均值/p50/p95/最大值，以及超过目标周期的tick数。175ms或58ms阻塞与30Hz周期33.3ms之间的关系必须解释；异步减少阻塞不一定降低动作年龄。

时钟域、success标签、checkpoint和protocol散列由记录生产者声明。数字合法或字段齐全不等于设备执行证明；应保留录像抽样、原始时间线、代码版本、全部失败及本人实际负责范围。
