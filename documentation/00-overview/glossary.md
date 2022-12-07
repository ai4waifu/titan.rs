# 核心术语

- **Tensor**：已物化的数据对象，具有 storage、dtype、shape、stride、layout 和 device。
- **TensorView**：不拥有 storage 的借用视图，只改变逻辑 shape、stride 或偏移。
- **Term**：不可变的延迟数学项，记录操作、输入、约束、source span 和 effect，不直接暴露可变 storage。
- **Graph**：Term 经过约束求解后的 typed DAG，包含 value、operator、effect token 和设备分区。
- **Value**：Graph 中的数据边，拥有 dtype、shape、layout、device、alias set 和 lifetime。
- **Operator**：Graph 中的语义节点，例如 MatMul、Reduce、Collective 或状态写。
- **VJP**：vector-Jacobian product，定义输出梯度到输入梯度的反向规则。
- **Parameter**：具有稳定 ID、名称路径、训练状态、梯度槽和 checkpoint 映射的 Tensor。
- **Storage**：由 HAL 管理的设备字节区域和生命周期。
- **Kernel**：满足统一 ABI、可由设备执行的编译单元。
- **Execution Plan**：图分区、buffer、kernel、stream、event 和 collective 的完整调度结果。
- **Capability Fingerprint**：设备型号、架构、驱动、指令、内存和队列能力的稳定摘要。
- **Tune Key**：确定一个调优结果适用范围的完整键。
- **World**：参与同一分布式作业的 rank 集合及其 epoch。
- **Collective Sequence**：所有 rank 必须一致遵循的 collective 顺序编号。
- **Checkpoint Commit**：全部 shard 持久化并通过校验后，对 manifest 的原子发布。
- **Run**：一次具有稳定 RunId、配置、模型、数据、拓扑和遥测上下文的训练或推理执行。
