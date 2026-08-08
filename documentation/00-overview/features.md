# 最终功能规格

## 用户编程接口

Titan 对外提供统一的 `titan` facade crate。普通 Rust 用户只需依赖该 crate，无需了解内部 workspace 拆分，也不需要安装 `tt.exe` 或 Web UI。

用户可以通过 `tensor!` 编写接近数学公式的表达式：

```rust
let output = tensor! {
    weight[0]
        * pow(difficulty, -weight[1])
        * (pow(state.col(0) + 1.0, weight[2]) - 1.0)
        * exp((1.0 - rating) * weight[3])
}
.eval()?;
```

表达式支持：

- Tensor 与 scalar 双向 `+ - * /` 运算。
- `pow(Tensor, scalar)`、`pow(Tensor, Tensor)`、`exp`、`log` 和常用逐元素函数。
- 广播、reshape、view、slice、row、col、select、gather 和 reduce。
- `f16`、`bf16`、`f32`、`f64`、`i32`、`i64` 及明确的 dtype promotion 规则。
- `Tensor`、`TensorView`、`Term`、`Parameter` 和参数 scalar 的自然组合。
- 在 `.eval()`、`.compile()` 或 `.run()` 边界集中报告 shape、dtype、device 和布局错误。

## 张量与类型系统

张量提供三档 shape 模式：

```rust
Tensor<B, F32, 2>
Tensor<B, F32, Shape<(Const<128>, Const<256>)>>
Tensor<B, F32, DynShape>
```

- 默认模式使用静态 rank 和动态 shape。
- 固定模型、协议张量和部署输入可以使用静态 shape。
- 动态控制流、研究模型和不规则数据可以使用 `DynShape`。
- owned tensor 与 view tensor 拥有不同类型和生命周期。
- contiguous、strided、blocked 和 backend-native layout 均有显式描述。

## 自动微分与训练

- 通用 reverse-mode autograd tape 和 backward graph。
- 每个基础算子注册 VJP/backward rule。
- `requires_grad`、`backward()`、`grad()`、`zero_grad()` 和梯度累积。
- no-grad、inference mode 和可嵌套的训练上下文。
- `Parameter`、参数树、参数分组、冻结与解冻。
- SGD、AdamW、学习率调度器、梯度裁剪和 optimizer state。
- mixed precision、主权重、loss scaling、溢出检测。
- activation checkpoint、recompute 和选择性保存。
- DataLoader、prefetch、pinned memory、批处理和可审计 shuffle。

DataLoader 使用显式策略：

```rust
ShufflePolicy::EveryEpoch { seed: 42 }
ShufflePolicy::Once { seed: 42 }
ShufflePolicy::Sequential
```

每个 epoch 记录 effective seed、permutation hash、数据分片和 worker 信息。

## 模型与宏

- `#[neural]`：神经网络结构、子模型遍历和 forward 入口。
- `#[parameters]`：参数注册、状态遍历、梯度清理和 checkpoint 映射。
- `#[kernel]`：内核 DSL、launch 配置和 lowering 元数据。
- `#[distributed]`：数据并行、张量并行、流水线并行、FSDP 和 ZeRO 元数据。
- `tensor!`：数学表达式到 `Term` 的构造。

## 编译器与执行引擎

- `Term` 到 typed graph 的捕获。
- forward graph 与 backward graph 的统一表示。
- symbolic shape、dtype、device、layout 和 alias 推导。
- 常量折叠、死代码消除、公共子表达式消除、逐元素融合和算子融合。
- buffer liveness、内存复用、in-place 合法性分析和 workspace 规划。
- eager、JIT、AOT、capture 和 replay 执行模式。
- CPU 队列、GPU stream、event、future、依赖栅栏和取消。
- 每个优化 pass 输出来源 span、变换原因和执行计划。

## 内核生成与设备后端

- CPU scalar、SIMD 和多线程后端。
- CUDA Driver API、ROCm、Metal 和 WGPU 后端。
- Rust 内核 DSL lowering 到 CPU SIMD、PTX、HIP、Metal IR 和 WGSL。
- 手工精调内核与自动生成内核共享统一 ABI、注册表和测试协议。
- 设备发现、能力指纹、驱动版本、stream/queue 和事件管理。
- owned allocation、内存池、对齐、pinned memory、统一内存和 RAII 释放。
- kernel binary cache、编译日志、source map 和错误定位。

## 自动调优

- MatMul、Conv2d、Reduce、Attention 和 fused graph 的策略注册表。
- block shape、vector width、pipeline depth、shared-memory padding 和 workspace 调优。
- Tensor Core、CMMA、TMA 等能力开关。
- 真实硬件重复采样、预热、异常值处理和置信阈值。
- `.tune` 文件、Redis 和对象存储后端。
- 设备、驱动、dtype、shape、layout、kernel 版本和编译选项组成完整 cache key。
- 生产遥测可以提交候选观测，只有统计显著更优的结果才能晋升。

## 分布式训练

- 数据并行、张量并行、流水线并行、FSDP 和 ZeRO Stage 1/2/3。
- Ring AllReduce、ReduceScatter、AllGather、Broadcast 和 point-to-point。
- 纯 Rust TCP、NCCL、RCCL 和 InfiniBand transport。
- rendezvous、membership、collective sequence、timeout 和故障传播。
- gradient bucket 与 backward/communication overlap。
- 1F1B、GPipe 和 ZeroBubble 调度。
- 弹性成员变化、故障注入、自动 checkpoint 和恢复。
- checkpoint manifest、tensor shard、optimizer shard 和 RNG/DataLoader state。

## 推理与部署

- 单二进制 Native 部署。
- 模型冻结、常量折叠、量化和静态内存计划。
- ONNX 导入导出、版本兼容和算子覆盖报告。
- 版本化权重格式、分片、校验和与按需映射。
- batch、streaming、KV cache、prefix cache 和服务调度接口。
- WASM 轻量推理和浏览器演示。

## 可观测性与工具链

- CPU/GPU 利用率、显存、allocation、kernel、graph pass、DataLoader 和 collective 指标。
- run、model、graph、operator、kernel、rank 和 checkpoint 的统一关联 ID。
- trace、metric、event 和诊断建议。
- 本地 collector、批处理、采样、背压、脱敏、鉴权和离线缓存。
- `tt.exe debug`：运行时、DataLoader、调优、checkpoint、遥测和部署诊断。
- `tt.exe cluster`：拓扑、rank、endpoint、通信能力、启动与故障诊断。
- Vue + Vite Web UI：训练曲线、设备、显存、kernel、调优、collective、checkpoint 和部署状态。
