# 阶段 05：自动调优与分层执行

## 前置条件

- kernel 变体、launch 参数、capability 和性能事件可枚举。
- Kernel cache key 与二进制校验稳定。
- profiler 能获得可靠的设备时间且不引入全局同步。
- 模型包支持按 tensor、layer、expert 和 block 索引读取。

## 代码交付物

- 候选生成、约束过滤、warmup、重复测量、异常值处理和选择器。
- `.tune` 版本化文件、原子提交、checksum、锁和损坏恢复。
- 设备、驱动、kernel、shape bucket、dtype/layout 和编译 flag 完整 key。
- 确定性、workspace、编译成本和数值误差约束。
- 离线导入导出、过期策略和后台重新调优。
- L0-L4 存储位置、显存区域、页锁定内存和持久化存储后端。
- `ResourcePlanner`、admission、KV Cache/专家缓存预算和带置信度的延迟/吞吐估计。
- MoE 路由需求、专家状态机、热度评分、多设备放置和驱逐 lease。
- 确定性/预测性预取、双缓冲、块/tile 页化和 deadline 回退。

## API 交付物

- `TunePolicy::{Off, ReadOnly, OnMiss, Refresh}`。
- 编译选项暴露调优预算、候选上限、确定性和 workspace 上限。
- 查询接口返回选中候选、测量分布、缓存来源和拒绝原因。
- `RuntimeProfile`、`MemoryBudget`、`WeightPlacement`、`QuantizationPolicy` 和 `ResourceBudgetReport`。
- 请求 admission 返回接受、排队、降级或拒绝及结构化原因。

## 测试交付物

- cache key 完整性和稳定性属性测试。
- `.tune` 截断、checksum、并发写、旧 schema 和只读目录测试。
- 计时噪声、异常值、候选失败、数值不一致和超预算测试。
- 命中缓存时不得重复编译或测量的集成测试。
- 模型块部分读取、checksum、L1/L2/L3 命中和异步迁移生命周期测试。
- 专家并发请求合并、预测失误、驱逐、KV 扩张、fragmentation 和 OOM 前拒绝测试。
- P2P/NUMA 能力差异、pinned limit 和 CPU fallback 测试。

## 性能交付物

- 调优开销、缓存命中延迟和候选收益报告。
- 代表 shape 分布上的加权收益，不以单个极端 shape 决策。
- 调优后的性能不得低于稳定基线超过 3%。
- 报告冷/热启动、首 token、稳态吞吐、专家等待、I/O 带宽、命中率和峰值 L1/L2。
- 每个受支持模型配置必须给出资源预算估计与实际值偏差。

## 文档交付物

- 完成调优协议、`.tune` schema、缓存 key 和生命周期文档。
- 写明用户如何迁移、清理和共享调优数据。
- 完成模型包、资源规划、MoE 调度、预取页化、量化、回退和诊断文档。

## 失败条件

- 缓存 key 漏掉设备、驱动、layout 或编译 flag。
- 测量包含首次编译或未完成的异步工作。
- 并发写入可以产生半个有效文件。
- 数值不合格候选仍可能被选择。
- Planner 在硬资源不可满足时仍允许请求进入设备执行。
- 预测预取失误可以改变权重版本或模型结果。
- 页化粒度与 kernel layout、量化 group 或 DMA 对齐不一致。

## 完成验收

在固定设备和 workload 上完成候选测量、选优、`.tune` 提交和二次命中；损坏或过期缓存可诊断并安全重建。代表性的超显存 MoE 配置能够在硬预算内完成冷启动、预取、专家执行和回退，结果与完整驻留基线满足数值约束。

## 解锁条件

kernel 与分层放置在单机上稳定，性能、权重版本和 identity 可跨 rank 校验，允许分布式计划引用。
