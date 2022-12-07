# 优化流水线

## Pass 顺序

1. 解析和 Term normalization。
2. dtype、rank、shape、device 和 layout 求解。
3. 常量折叠与 identity elimination。
4. dead code、公共子表达式和 view 合并。
5. broadcast materialization 与 reduce-to-shape。
6. alias analysis 与 in-place 合法性证明。
7. forward/backward elementwise fusion。
8. MatMul、Conv、Reduce、Attention pattern recognition。
9. layout、tile、workspace 和 device placement 选择。
10. buffer liveness 与内存规划。
11. kernel candidate 生成、autotune 查询和 launch 选择。
12. stream、event、collective 和 checkpoint 调度。

## 优化合法性

每个 pass 必须声明输入前提、输出不变量、数值影响、梯度影响、内存影响和失败处理。浮点重排需要通过 precision policy 授权；fast-math 不能默认打开。

融合必须证明：

- dtype 和 accumulation 规则相同。
- broadcast 后的索引等价。
- alias 不产生读写覆盖。
- NaN/Inf 行为符合当前数值模式。
- backward graph 的 VJP 与融合算子一致。
- workspace、alignment 和 launch 限制满足后端能力。

## Pass Report

每个报告包含 pass 名称、输入 graph hash、输出 graph hash、变更节点、删除节点、插入节点、source spans、合法性证明摘要、预计收益、风险和回退原因。Web UI 和 `tt debug` 只消费该报告，不猜测优化发生与否。

## 编译策略

- Eager：单次物化，低延迟优先。
- JIT：按 shape、dtype、device 延迟编译并缓存。
- AOT：发布时编译固定输入与设备集合。
- Capture：稳定形状和地址后捕获可重复执行图。
- Interpret：CPU 诊断模式，逐节点执行并保留完整 trace。
