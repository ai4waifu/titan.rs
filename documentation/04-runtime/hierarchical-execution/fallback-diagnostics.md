# 回退、限额与诊断

## 回退顺序

资源不足时按策略依次选择：降低 admission 并发、缩短新会话最大上下文、收缩可驱逐专家缓存、降低预取深度、切换更紧凑且已验证的权重/KV 格式、把冷 block 留在 L2/L3、排队或拒绝请求。已开始会话的语义不能被静默改变。

CPU fallback 只适用于算子明确支持且传输成本未超过 deadline 的场景。无法保证正确权重版本、数值策略或资源上限时立即失败。

## 失败类型

- `BudgetUnsatisfied`：静态预算不可行，包含最小所需资源。
- `AdmissionRejected`：当前并发和动态占用无法接收请求。
- `WeightBlockUnavailable`：block 缺失、损坏、格式或 capability 不支持。
- `PrefetchDeadlineMissed`：确定性权重未在执行 deadline 前就绪。
- `PinnedMemoryLimit`：页锁定分配超过上限。
- `FragmentationLimit`：总空闲足够但没有满足对齐的连续区域。

## 观测指标

Profiler 记录 L1/L2/L3 命中、block 读取/传输字节与延迟、预取准确率、取消率、expert wait、KV page、pinned bytes、fragmentation、fallback、首 token 延迟和 token 吞吐。

`tt debug` 输出资源预算与实际峰值差异，`tt cluster` 输出 GPU P2P/NUMA/transport 能力，Web UI 显示内存区域、在途传输、缓存命中和降级原因。

## 验收

基准必须记录硬件、互连、存储、模型/量化摘要、batch、context、concurrency、冷热状态和统计方法。分层执行只有在正确性、硬资源上限、稳定运行和可复现性能均通过时才视为可支持配置。
