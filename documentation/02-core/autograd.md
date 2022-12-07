# 反向自动微分

Titan.rs 使用 reverse-mode Autograd，将 forward Term/Graph 展开为显式 backward graph。梯度贡献、保存值、重算和 Parameter 写入都由节点表达，不依赖隐式全局 tape。

## VJP 协议

```rust
pub trait VjpRule {
    fn save_for_backward(&self, ctx: &mut SaveContext);
    fn backward(
        &self,
        ctx: &BackwardContext,
        output_grad: ValueId,
    ) -> TitanResult<Vec<Option<ValueId>>>;
}
```

每条规则声明保存输入/输出、broadcast reduction、view 逆映射、不可微点、支持 dtype、complex 约定和 higher-order gradient 能力。

## Backward 构造

1. 从 scalar loss 或显式 seed gradient 反向遍历。
2. 按输出 value 收集所有下游贡献。
3. 调用 VJP 生成输入梯度。
4. 对 broadcast 插入 `sum_to_shape`，对 view 插入 inverse-view 或 scatter。
5. 按数值策略合并贡献并写入 Parameter gradient slot。
6. 分布式模式插入 bucket、ReduceScatter/AllReduce 和溢出 collective。

## 保存与重算

`ActivationPolicy` 支持 `SaveAll`、`Checkpoint(regions)`、`Budget(bytes)` 和 `Offload(target)`。重算区域必须保存 RNG counter；状态写、外部调用和通信只有在定义幂等语义时才能重放。

## 梯度模式

`no_grad` 禁止记录新梯度关系，`detach` 生成共享数据但断开的 Term/Tensor，inference mode 进一步禁止训练状态和版本计数。嵌套上下文使用显式 guard，跨 async task 传播由 runtime context 控制。

## 数值验证

每个 VJP 覆盖 finite-difference、broadcast、非连续 view、空维、极值、NaN/Inf、累积和融合前后等价。容差按输入 dtype、accumulate dtype 和算子条件数定义。
