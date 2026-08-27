# Titan.rs

Titan.rs 是面向生成式 kernel、运行时设备抽象和自动调优的 Rust 深度学习基础设施。

当前版本定位为 **CPU protocol/execution prototype**。仓库已经冻结了 HAL、Tensor、Graph、Kernel ABI、Schema 和本地分布式协议的方向，但还没有完成生产级 kernel dispatch、GPU 真实执行、网络集群或端到端模型运行。

## 当前可验证能力

- Rust workspace 可构建，协议 crate 依赖方向已经固定。
- CPU buffer 的 allocate、upload、download、copy 和 event contract 测试。
- 运行时设备绑定的 typed Tensor metadata、显式 Storage 和显式下载。
- 不包含算子知识的 HAL contract。
- typed Graph、OpRequest 和基础 Kernel ABI。
- `titan-schema` 的算子协议边界。
- 进程内、无网络 I/O 的 distributed protocol tests。

## 当前不可用能力

- Runtime 的真实 kernel dispatch 和可执行 `ExecutionPlan`。
- generated elementwise、reduction、MatMul kernel 的端到端执行。
- CPU AVX2/FMA executable codegen 和 W^X loader。
- CUDA Driver API module load/launch。
- ROCm gfx1100 code-object load/launch。
- 真实网络集群、多进程 transport 和 NCCL/RCCL collective。
- 生产级 autotune correctness gate、设备事件 benchmark 和 winner 晋升。
- mmap、分块权重、device offload、paged KV cache 和 quantized model execution。

## 产品目标

### 单机弱配置

Titan 的硬产品承诺是：**只要模型、KV cache、临时 artifact 和输入数据在可用磁盘上可持久化，系统就必须能够运行，不得因为 RAM/VRAM 不足直接拒绝。** 性能可以下降，但必须通过分层存储、分页、分块执行和可观测的 backpressure 继续取得进展。

为兑现该承诺，目标执行链路为：

```text
磁盘模型文件 -> mmap/分块读取 -> 量化权重 -> 分层 Storage
  -> workspace admission -> paged KV cache -> generated baseline
```

该承诺不等于任意性能或任意延迟保证。每个模型仍必须发布 RAM、VRAM、磁盘、上下文、吞吐和降级策略边界。

### GPU

目标是不依赖 CUDA Toolkit、NVRTC、CUDART、hipRTC、COMGR、LLVM 或 clang，通过动态 Driver API 加载由 Titan 生成的 artifact。

### 集群

目标是真实多节点执行、拓扑感知 collective、通信/计算重叠、可观测调优和 checkpoint/recovery。Titan 的长期目标是通过一体化优化，在定义明确的 workload 上超越 PyTorch、vLLM 和 SGLang；任何达成宣称都必须由固定 benchmark suite 的公开实测报告证明。

### Kernel

每个支持算子必须有 CPU scalar reference 和同设备 generated baseline。handwritten kernel 只能作为普通 candidate，不能拥有 Runtime/HAL 特殊入口。

## 承诺边界

Titan.rs 当前不承诺：

- 当前原型已经兑现“硬盘放得下就一定能运行”；
- 未经 benchmark 证据即已超越 PyTorch/vLLM/SGLang；
- 仅凭 CUDA/ROCm 枚举或 placeholder 即支持 GPU；
- 仅凭 FSDP/ZeRO 枚举或 checkpoint 文本即支持分布式训练；
- 仅凭 `.tune` 文件 round-trip 测试即完成 autotune。

## 开发入口

```shell
cargo check --workspace
cargo test --workspace
cargo run -p titan-example
```

example 当前只验证 CPU/协议路径，不代表端到端模型推理或集群执行。

详细边界见：

- `docs/architecture/kernel-first.md`
- `docs/architecture/protocols.md`
- `docs/architecture/backend-contract.md`
- `docs/architecture/autotune-v2.md`
- `docs/product/capability-matrix.md`
- `docs/product/single-machine.md`
- `docs/product/cluster.md`
- `docs/benchmarks/benchmark-policy.md`
