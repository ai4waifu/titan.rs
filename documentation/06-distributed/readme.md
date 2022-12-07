# 分布式系统

本目录定义 Titan.rs 在多进程、多节点和多加速器上的一致执行模型。分布式运行由控制面建立成员关系和拓扑，由数据面承载张量与 collective 通信；训练状态、检查点和故障恢复均以显式协议推进。

## 文档索引

- [控制面](control-plane.md)：rendezvous、成员关系、rank/world、拓扑和会话生命周期。
- [数据面](data-plane.md)：传输层、连接、缓冲区、超时和错误传播。
- [Collective 原语](collectives.md)：Ring AllReduce、ReduceScatter、AllGather 和序列一致性。
- [并行策略](parallelism.md)：Tensor Parallel、Pipeline Parallel、1F1B、GPipe、ZeroBubble、FSDP 和 ZeRO。
- [检查点与恢复](checkpoint-recovery.md)：manifest、分片、提交、恢复和校验。
- [故障模型](failure-model.md)：故障分类、检测、隔离、重试和恢复状态机。

## 设计不变量

1. 同一 `RunId` 内的 rank 必须看到相同的 collective 序列、图版本和参数版本。
2. 控制面只管理身份、拓扑、租约和状态，不传输大张量。
3. 数据面操作必须携带 `run_id`、`step`、`collective_seq` 和 `tensor_id`，禁止使用隐式全局上下文。
4. collective 超时必须让所有参与者得到同一个失败结论，不能只让发起方返回错误。
5. checkpoint 只有在 manifest 原子提交后才可被恢复；未提交分片一律不可见。
