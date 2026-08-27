# Kernel-First 架构

Titan 只允许以下执行数据流：

```text
Tensor / Graph Value
  -> OperatorSchema + OpRequest
  -> Graph normalization / fusion
  -> StrategyRegistry
  -> KernelRecipe
  -> SSA Kernel IR + KernelAbi verification
  -> TargetCompiler
  -> Artifact cache
  -> Correctness validation
  -> Device-event benchmark
  -> Tune winner
  -> ExecutionPlan
  -> HAL stream/event/launch
```

Tensor 只持有 Storage、shape、strides、layout、device 和 pending event，不依赖 Runtime。HAL 只管理资源和异步执行，不认识任何算子。backend 不得依赖 Graph 或 Runtime。

eager 单算子和 compiled graph 必须构造相同的 `OpRequest`，并使用同一 candidate/ABI/launch 管线。每个支持算子必须同时提供 CPU scalar reference 和同设备 generated baseline；handwritten kernel 只是同一注册表中的普通候选。

默认禁止跨设备和 CPU fallback。仅当 `ExplicitCpu` policy 明确启用时，计划可插入可观测 Upload/Download 节点。
