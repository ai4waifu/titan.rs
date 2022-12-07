# 并行策略

并行策略由模型布局、图编译器和运行时共同决定。每个参数、激活和梯度都带有 `ShardSpec`，任何隐式 reshape 或跨策略复制都必须在图中出现。

## Tensor Parallel

Tensor Parallel 沿权重维度切分线性层和注意力投影。列并行层在输出侧使用 AllGather 或延迟到下一个算子；行并行层在输入侧使用 ReduceScatter/AllReduce。布局转换必须由 `Shard` 节点表达，编译器据此安排通信与计算重叠。

## Pipeline Parallel

Pipeline Parallel 将连续层分配到 stage。micro-batch 由调度器编号并携带 forward/backward 生命周期。支持：

- GPipe：先完成全部 forward，再反向，内存需求较高但协议简单。
- 1F1B：warmup 后交替 forward/backward，平衡吞吐和激活内存。
- ZeroBubble：把可独立执行的权重梯度通信填入 pipeline bubble，要求图标注依赖和优先级。

stage 边界使用异步 send/recv，激活和梯度必须带 micro-batch id，乱序到达时由接收队列按 id 配对。

## FSDP 与 ZeRO

- FSDP：参数在 forward 前 all-gather，计算后释放完整参数；backward 前重新 gather，梯度以 ReduceScatter 形式落回 shard。
- ZeRO Stage 1：只切分 optimizer state。
- ZeRO Stage 2：切分 optimizer state 和 gradient。
- ZeRO Stage 3：进一步切分 parameter，按计算窗口 gather/release。

每种策略都必须定义 shard owner、生命周期、重建点和 checkpoint 映射。显存峰值报告按完整 tensor、shard、通信 buffer 分开统计。

## Bucket 与调度

gradient bucket 按参数拓扑顺序和目标字节数构建，不能跨 dtype 或不同通信域合并。调度器优先发起已就绪且能释放内存的 bucket；通信失败时保留 bucket 未规约状态，供恢复状态机重新提交。
