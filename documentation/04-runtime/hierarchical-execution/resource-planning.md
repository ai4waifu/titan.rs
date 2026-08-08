# 运行前资源规划

`ResourcePlanner` 在模型加载和请求 admission 前生成 `ResourceBudgetReport`。规划使用硬上限，不依赖运行期间无限扩容。

## 最小运行时契约

运行时基础层提供后端无关的 `ResourceBudget`、`ResourceRequest` 与 `ResourceBudgetReport`。调用方先以字节数和并发数构造请求，再调用 `ResourceBudget::assess`；返回 `feasible=false` 时不得进入分配、DMA 提交或 kernel launch。报告中的剩余值使用饱和减法，便于诊断超额请求而不发生整数下溢。

完整 Planner 在此契约之上填充模型放置、带宽预测和回退决策；它不能绕过基础预算检查。

## 输入

- 模型总权重、量化后字节、dense/专家划分和最大单块；
- backend capability、设备显存、互连拓扑、DMA 和虚拟内存能力；
- 主机可用内存、页锁定上限、NUMA 和 L3/L4 带宽/延迟；
- batch、上下文长度、并发、生成 token 上限和精度策略；
- KV Cache page、activation、workspace、communication 和安全余量；
- 首 token 延迟、稳态吞吐和请求队列预算。

## 输出

报告包含可行性、L0-L4 放置图、常驻权重、专家槽位、KV Cache 上限、buffer 数量、最大并发、预取深度、预计命中路径、冷启动 I/O、首 token 延迟范围、稳态吞吐范围和主要瓶颈。

估计值必须带输入假设和置信区间。带宽未知时运行受控探测；无法探测时使用保守配置并标记 `estimate_quality=low`。

## Admission

请求进入前根据实际 KV 占用、在途传输、队列和碎片重新校验预算。无法满足硬约束时按策略排队或拒绝，不得先分配后依赖 OOM 触发回退。

## 规划 key

规划缓存 key 包含模型/权重摘要、设备与驱动、拓扑、精度、batch/context/concurrency、kernel 版本、`.tune` 版本和预算策略。任一影响内存或时延的字段变化都必须重新规划。
