# 部署与互操作 API

## DeploymentManifest

```text
schema: titan.deployment
version: major.minor
model_id
model_schema
target: native | wasm
backend: cpu | cuda | rocm | metal | wgpu
required_capabilities
weights_manifest
graph_artifact
kernel_artifacts
quantization
runtime_version
checksum
```

Native 是大模型默认目标，可使用 CUDA/ROCm 多卡和流式服务。WASM 必须声明模型大小、算子集合、内存上限和浏览器能力，只允许轻量模型与演示。

## ONNX

导入流程：读取 opset、graph、value info、initializer、operator attributes，映射到 Term，再执行 shape/dtype/device 校验。未覆盖算子必须生成带节点路径的错误，不得静默替换。

导出流程：从冻结 Graph 生成稳定 value name、initializer、operator、shape 和 dtype，并写入版本、校验和与算子覆盖报告。ONNX 文件是标准二进制协议，不使用文本伪格式。

## 权重

权重 manifest 记录 ParameterId、名称路径、dtype、shape、layout、分片、offset、length、checksum 和压缩方式。加载支持严格匹配、显式迁移和只读 memory map。

## 分层加载配置

超显存模型通过加载配置和运行时 profile 指定放置策略，模型结构宏不携带设备策略：

```rust
let profile = RuntimeProfile::builder()
    .device(Device::cuda(0))
    .memory_budget(MemoryBudget::from_gib(20))
    .weight_placement(WeightPlacement::tiered())
    .max_context(8192)
    .max_concurrency(2)
    .build()?;

let model = ModelLoader::new()
    .quantization(QuantizationPolicy::balanced())
    .profile(profile)
    .load(path)?;
```

加载阶段必须生成资源预算报告，列出 L1/L2/L3/L4 放置、KV Cache 上限、预取深度、预计延迟/吞吐和无法满足的约束。执行过程中只允许在报告声明的回退策略内降低并发、收缩可驱逐缓存、延长等待或拒绝新请求。

详细协议见 [超显存模型与 MoE 分层执行](../04-runtime/hierarchical-execution/readme.md)。
