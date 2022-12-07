# 执行状态机

## Run 状态

```text
Created -> Validating -> Compiling -> Ready -> Running
Running -> Paused -> Running
Running -> Checkpointing -> Running
Running -> Recovering -> Running
Running -> Completed
Running -> Failed -> Recovering | Aborted
```

每次状态迁移写入 RunEvent，包含 RunId、Step、rank、原因、上一个状态、下一个状态、耗时和错误链。非法迁移立即返回错误。

## Step 状态

```text
FetchBatch
 -> TransferInputs
 -> Forward
 -> Loss
 -> Backward
 -> Unscale
 -> OverflowCollective
 -> GradientReduce
 -> Optimizer
 -> Scheduler
 -> CheckpointPolicy
 -> TelemetryFlush
 -> StepComplete
```

失败点必须声明是否可以重试、是否需要回滚参数、是否需要恢复 checkpoint、是否允许跳过 batch。

## 执行模式

`#[distributed]` 仅校验声明中的 `world` 与 `strategy` metadata，不会创建 runtime、world group 或通信操作。执行器在 `Validating` 阶段将该 metadata 与实际 `DistributedRuntime` 的 rank、world 和拓扑交叉校验；不匹配必须在进入 `Running` 前失败。

- 同步模式：每个 operator 完成后才继续，适合解释和故障定位。
- 异步模式：使用 stream/event 依赖执行独立节点。
- 捕获模式：地址、shape、launch 和依赖稳定后重复提交 capture graph。
- 恢复模式：从最后一个 committed checkpoint 重建 optimizer、RNG、DataLoader 和 collective epoch。

任何模式都必须共享同一 Graph 语义和数值策略。
