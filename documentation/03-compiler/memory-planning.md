# 内存规划

## 生命周期分析

规划器依据每个 Value 的 first use、last use、alias set、device、stream 和 checkpoint 保存需求计算生命周期。两个值只有在：生命周期不重叠、设备相同、layout 可复用、alignment 满足、无活跃 alias 时才可共享 buffer。

## Buffer 类型

- Persistent：Parameter、optimizer state、长期缓存。
- Activation：forward 中间值，默认按 backward 需要决定保存或重算。
- Workspace：kernel 临时空间，可在执行计划中复用。
- Communication：gradient bucket、send/receive staging。
- PinnedHost：异步 CPU/GPU 搬运。
- Offload：checkpoint/recompute 的层级存储。

## Allocator

每个 Device 拥有 allocator。allocator 提供 size class、alignment、stream-safe lease、统计和碎片率。释放不是立即归还操作系统，而是进入带 event 的缓存池；event 完成后才允许重用。

规划器必须输出峰值显存、每类 buffer 统计、复用次数、碎片率和最危险的长生命周期值。OOM 错误必须包含 requested、reserved、active、fragmented 和可选的重算建议。

## Checkpoint 与内存

Checkpoint 保存值必须在 planner 中标记为 persistent，不能被临时 buffer 覆盖。重算区域必须标记 RNG、effect 和通信依赖。分布式 shard 的 host staging 也必须纳入峰值内存报告。
