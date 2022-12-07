# Parameter 与 Optimizer

## Parameter

Parameter 包含稳定 ParameterId、名称路径、Tensor、requires-grad、gradient slot、group tag、shard metadata 和状态版本。`#[parameters]` 生成确定性遍历；字段重排通过显式 key 保持 checkpoint 映射。

Buffer 是模型状态但默认不求梯度，用于 normalization statistic、position cache 等。Parameter 与 Buffer 均进入 state dict，但命名空间和更新规则分开。

## 梯度状态机

```text
Absent -> AllocatedZero -> Accumulating -> Ready -> Reduced -> Consumed
```

单设备 Optimizer 消费 `Ready`，分布式 Optimizer 消费 `Reduced`。溢出、取消或 collective 失败使本 step 进入 `Discarded`，不得更新 parameter 或 scheduler。

## Optimizer 协议

Optimizer 定义 parameter groups、state 初始化、step、zero_grad、state_dict 和 load_state_dict。SGD、AdamW 的 state 以 ParameterId 为 key，sharding 独立于模型字段顺序。

## Mixed Precision

PrecisionPolicy 分别指定 forward、parameter、master weight、accumulate 和 optimizer state dtype。动态 loss scaler 记录 scale、增长窗口、连续成功、最近溢出和上下限。任一 rank 溢出都统一取消更新。

## 原子更新

一个 optimizer step 要么提交全部 parameter/state/version，要么不提交。异步 kernel 完成后发布 step commit event；checkpoint 只读取已提交版本。
