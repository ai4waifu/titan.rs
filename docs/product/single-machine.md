# 单机弱配置承诺

Titan 的产品承诺是：只要运行所需状态能够持久化到可用磁盘，系统必须继续运行，而不是因为 RAM 或 VRAM 不能完整容纳模型而直接拒绝。该承诺允许吞吐下降，不承诺固定性能。

兑现路径：

```text
磁盘权重/状态 -> mmap 或分块读取 -> 量化/压缩 -> Host/Device 分层 Storage
  -> workspace admission -> paged KV cache -> backpressure -> generated execution
```

验收规则：

- 每项 allocation 必须先经过 RAM、VRAM、workspace 和磁盘预算 admission。
- 内存不足时必须将可换出权重、激活或 KV page 写入可用磁盘，并记录 bytes、原因和耗时。
- 磁盘空间耗尽、I/O 错误或不可换出 workspace 失败必须产生结构化诊断，不能死锁或静默丢失数据。
- 每个模型发布最低磁盘、建议 RAM/VRAM、最大上下文、吞吐和降级策略。
- 未实现 mmap、分页和磁盘 spill 前，任何版本都不得自称已兑现该承诺。
