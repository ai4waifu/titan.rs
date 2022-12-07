# 阶段 06：分布式训练与故障恢复

## 前置条件

- 单机训练、ExecutionPlan、stream/event、kernel 和 checkpoint 基础稳定。
- `ShardSpec`、参数 key、optimizer state 和 RNG/DataLoader 状态可序列化。
- 运行时具有取消、超时和统一错误传播。

## 代码交付物

- rendezvous、membership、rank/world、lease、topology snapshot 和 epoch。
- TCP transport 以及 NCCL/RCCL/InfiniBand 适配接口。
- Ring AllReduce、ReduceScatter、AllGather、send/recv 和 collective 序列校验。
- Tensor Parallel、Pipeline Parallel、GPipe、1F1B、ZeroBubble、FSDP 和 ZeRO 1/2/3。
- gradient bucket、反向通信重叠、rank skew 监测和资源计划。
- checkpoint manifest、tensor/optimizer shard、RNG/DataLoader 状态、两阶段提交和恢复状态机。

## API 交付物

- `#[distributed(...)]` 声明并行布局和策略。
- `DistributedContext` 提供 run、rank、world、epoch、topology 和 collective 接口。
- 策略配置可表达 shard 轴、pipeline stage、micro-batch、ZeRO stage 和容错预算。

## 测试交付物

- 单机多进程和跨机 TCP collective 已知答案测试。
- collective 乱序、rank 丢失、frame 损坏、超时、重连和 epoch 切换测试。
- 各并行策略与单设备训练的数值等价和参数覆盖测试。
- checkpoint 部分写入、manifest 损坏、world size 改变和重复恢复测试。
- 通信与反向重叠的 event 依赖和内存生命周期测试。

## 性能交付物

- collective 延迟/带宽曲线、scaling efficiency、rank skew 和重叠率。
- FSDP/ZeRO 的峰值显存、额外通信和 step time 对比。
- checkpoint 写入、恢复时间和存储放大率。

## 文档交付物

- 完成控制面、数据面、collective、并行策略、checkpoint 和故障模型文档。
- 为每个协议字段和状态转移定义版本和错误。

## 失败条件

- collective 顺序依赖调用时序且无法在执行前校验。
- 任一 rank 失败后其他 rank 可以继续更新参数。
- checkpoint 在 manifest 提交前可见。
- 恢复后参数、optimizer、RNG 或 DataLoader 摘要不一致。

## 完成验收

多节点训练可执行、保存、注入 peer failure、从最近 checkpoint 建立新 epoch 并继续；恢复后的下一步与无故障基线满足确定性约束。

## 解锁条件

run/rank/step/collective/checkpoint 标识和故障事件稳定，允许统一遥测、诊断和运维工具消费。
