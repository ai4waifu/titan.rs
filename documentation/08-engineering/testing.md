# 测试策略

测试按风险分层，每层都必须有稳定、可重复的输入和明确的失败诊断。

## 单元测试

覆盖 shape/dtype/layout 推导、Term 构造、VJP 规则、错误分类、序列化、缓存 key、拓扑排序、planner 生命周期和诊断规则。纯函数测试禁止依赖 GPU、网络和系统时钟。

## 集成测试

使用 CPU HAL 验证 Tensor、MatMul、训练循环、checkpoint 提交与恢复、AllReduce 和 `tt.exe` 输出。运行时测试覆盖 stream/event、背压、取消、超时和资源回收。

## 数值回归

每个算子提供 finite-difference 梯度检查；浮点比较按 dtype 定义绝对/相对误差。混合精度测试覆盖 loss scaling、溢出、反向 accumulate dtype 和 deterministic 模式。固定种子、线程数、拓扑和输入摘要，失败报告必须包含第一个不一致 tensor。

## 属性与模糊测试

shape 广播、layout 变换、ShardSpec 覆盖范围、manifest 合并、frame 解码和 cache key 使用 property test。frame、manifest、配置和宏输入使用 fuzz test，要求拒绝畸形数据且不 panic、不越界、不泄漏资源。

## 端到端测试

最小工作流必须贯通：构图、编译、执行、反向、优化器更新、checkpoint、故障注入、恢复、遥测采集、`tt debug` 报告和 Web UI API。分布式矩阵至少覆盖单机多进程、跨机 TCP 和可用的 GPU transport。

## 基准门槛

基准测试固定硬件、驱动、线程、输入和 warmup 次数。报告吞吐、p50/p95 延迟、峰值显存、kernel 命中率、通信带宽和编译时间。相对基线退化超过目录定义阈值时阻断发布，除非变更记录给出测量证据和批准。
