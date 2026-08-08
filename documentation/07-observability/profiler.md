# 性能分析器

性能分析器由轻量采样器和显式 instrumentation 组成。默认只启用低开销指标，详细 kernel 和图分析必须通过会话配置打开。

## 采集范围

- CPU：线程状态、run queue、上下文切换、DataLoader 等待和 host-to-device copy。
- GPU：kernel 时间、stream、occupancy 摘要、显存分配、copy engine 和同步等待。
- Allocator：已用、保留、峰值、碎片率、分配失败和 buffer 生命周期。
- Compiler：每个 graph pass 的输入节点数、输出节点数、耗时和内存变化。
- Kernel：kernel id、变体、launch 参数、编译缓存命中、调优候选和最终选择。
- Distributed：collective 类型、字节数、耗时、有效带宽、队列等待和 rank skew。
- DataLoader：读取、解码、batch 组装、预取队列和丢弃样本计数。
- 分层执行：L1/L2/L3 命中、专家 block 读取/传输、预取准确率、KV page、pinned bytes、碎片率和回退原因。

## 时间基准

区间时间使用单调时钟；跨进程显示时同时记录 wall-clock 校准点和 clock uncertainty。设备时间和 host 时间通过 event 对齐，不能直接相减而不注明校准误差。

本地 `Collector` 在事件入队时分配严格递增的 `sequence`，包括容量耗尽而丢弃的事件。drain 不会重置该计数器，因此同一 collector 的序号不会回退或复用。

## 分析产物

分析器生成三种产物：

1. 时间线：按 run、rank、线程和 stream 展开 trace。
2. 聚合表：按 operator、kernel、collective 和 pass 聚合计数、p50/p95/p99、峰值和失败率。
3. 建议：根据规则识别通信未重叠、allocator 碎片、kernel 退化、DataLoader 饥饿和 rank skew。

## 开销预算

默认 instrumentation 的 CPU 开销目标小于 1%，GPU 事件记录不应引入全局同步。超过预算时采集器自动降低 normal/debug 采样，并在报告中标注降级区间。
