# 量化与精度策略

量化策略同时决定模型质量、传输字节、缓存容量、kernel 能力和回退路径。策略通过模型加载配置传入，不由 `#[neural]` 或 `#[kernel]` 隐式决定。

```rust
let policy = QuantizationPolicy::balanced();
let model = ModelLoader::new()
    .quantization(policy)
    .load(path)?;
```

## 分组件策略

- Router、关键 normalization：BF16/FP16，统计和 reduction 可使用 FP32 accumulate。
- Attention 与高复用 dense：选择设备验证过的 BF16、FP16、FP8 或低比特格式。
- 热专家：优先设备高吞吐格式，避免每次执行转换。
- 温/冷专家：允许更紧凑存储格式，但转换成本进入预取计划。
- KV Cache：根据上下文、并发、质量和 backend kernel 能力独立配置。

## 模型准备流程

识别张量语义，选择目标格式，读取或执行校准，生成 scale/zero-point/group metadata，校验 tensor 完整性，运行质量评估，再写入模型 manifest。蒸馏、再训练和校准属于模型准备流程，不在推理请求中执行。

## 运行时回退

运行时只能选择 manifest 声明且通过质量/能力验证的格式。目标 kernel 不可用时可以切到已验证的高精度副本或 CPU 路径；每次回退记录实际 dtype、额外字节、预计延迟和原因。
