# MoE 路由与专家缓存

## 执行序列

```text
输入激活
 -> Router
 -> token 的 Top-K 专家与权重
 -> 合并本层需求
 -> 查询专家位置与状态
 -> 命中复用 / 缺失加载
 -> 专家 FFN
 -> 按路由权重合并
 -> 更新统计与后续预取
```

Router 和关键归一化属于常驻路径。专家权重的放置由运行配置和执行计划决定，模型结构宏不编码设备层数或缓存策略。

## 专家状态

每个专家块处于 `Indexed`、`ReadingL3`、`ResidentL2`、`TransferringL1`、`ReadyL1`、`InUse`、`Evictable` 或 `Unavailable`。状态变化由 token 和 event 驱动；同一 block 的并发请求共享 load future，不能重复读取和分配。

## 缓存评分

缓存决策综合最近访问、窗口频率、读取/传输成本、专家大小、后续层预测、请求优先级、跨会话共享度和 L1/L2 压力。热、温、冷是评分区间，不是写死的专家类别。

驱逐只能选择 `Evictable` 且没有 stream lease 的 block。优先驱逐低收益大块，并保留能减少下一 token 关键路径的专家。命中率按模型、层、batch、context、并发和路由分布分别统计。

## 多设备放置

HAL 检测 P2P、NVLink/PCIe 互连、NUMA、IOMMU 和跨进程访问能力。Planner 在复制、专家分片、远程设备读取和显式传输间选择，并结合数据并行、专家并行、Tensor Parallel 与 Pipeline Parallel。

不存在通用的单副本共享假设。没有 P2P 的设备必须使用副本或显式通信；最终放置和成本写入 ExecutionPlan 与 profiler。
