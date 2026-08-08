# 连续 eager CPU 算子

`titan-tensor` 提供连续 row-major `f32` storage 的可复用 eager CPU 参考实现。它们是数值基线，不创建 Stable Diffusion 或其他模型 pipeline。

## 已支持契约

- `reshape` 保持元素顺序并要求元素数量一致；`permute` 验证每个轴恰好出现一次，并 materialize 连续输出。
- `add`、`mul` 支持同 rank 的逐轴广播（每个维度相等或一侧为 1）。`silu` 逐元素计算 `x / (1 + exp(-x))`。
- `concat(axis, tensors)` 仅允许选定轴以外的维度相同；`softmax(axis)` 使用减去行最大值的稳定实现。
- `layer_norm(epsilon, weight, bias)` 沿最后一维归一化；可选 affine 张量必须为该维长度。
- `conv2d` 接受 NCHW 输入、OIHW weight、可选 output-channel bias，以及 stride/padding/dilation/groups。语义是 cross-correlation。
- `group_norm` 在每个 NCHW 样本的 channel group 与空间位置上归一化；`resize_nearest2d` 使用 `floor(output_index * input_size / output_size)` 的 nearest source index。

每个失败路径返回 `TensorError`，包含操作名与必要的 axis、rank、期望/实际 shape 或参数详情；调用者不得依赖字符串匹配错误。

## 边界与演进

当前 Tensor API materialize 连续输出，尚未暴露 stride/offset view。GPU lowering、自动调优和非连续 view 必须先保持与此数值/shape 契约一致。

SDPA 后续建议公开为：

```rust
pub fn scaled_dot_product_attention(
    query: &Tensor<B, f32, 4>, key: &Tensor<B, f32, 4>, value: &Tensor<B, f32, 4>,
    options: SdpaOptions,
) -> Result<Tensor<B, f32, 4>, TensorError>;
```

shape 使用 `[batch, heads, sequence, head_dim]`；`SdpaOptions` 应明确 `scale`、可选 additive mask、causal 标志和数值精度策略。实现前必须定义 mask 广播规则、head/sequence 的完整 shape 错误以及 CPU reference test vectors。
