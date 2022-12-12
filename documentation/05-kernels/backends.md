# 设备后端

## HAL 能力

Backend trait 分为四组：

- Allocation：allocate、deallocate、reallocate、pin、map、unmap。
- Execution：queue、stream、event、submit、wait、poll。
- Kernel：compile、load、launch、query capability。
- Transfer：copy、fill、host staging、peer copy。

每组使用关联类型，CPU 后端可以同步实现，GPU 后端提供异步实现。Backend 不暴露厂商裸指针到上层。

## 后端实现顺序

1. CPU scalar 与 SIMD：作为所有数值和 autograd 回归基线。
2. WGPU：验证跨平台 queue、buffer、shader 和浏览器兼容边界。
3. CUDA Driver：实现高性能 Native GPU 主路径。
4. ROCm：复用 graph/ABI/autotune 语义并接入 HIP/RCCL。
5. Metal：实现 Apple 设备能力和 Metal shader lowering。

## CUDA PTX lowering

CUDA 后端的 PTX 只允许通过原子、带类型的 `PtxInstruction` emitter 生成：每条指令的 `Display` 输出一行 PTX。Lowering 不得再使用多行字符串模板拼装内核体；宏算子（GEMM、Conv2d、归一化等）必须展开为原子指令序列后再 stringify。

## eager CPU 基线

`titan-tensor` 的连续 `f32` eager 算子是后端实现的数值基线。CPU 版本 materialize 连续 row-major 输出，并对 Conv2d、GroupNorm、softmax、layer norm、nearest resize、concat 与广播给出结构化 shape 错误。任一新后端的 kernel 必须在相同输入、参数及 NaN/Inf 策略下匹配这些契约；性能优化不得改变其索引、group 或归一化范围。

未来 SDPA backend ABI 应使用显式 Q/K/V shape、可选 additive mask buffer、scale 和 causal flag，而不是由模型层拼接专用 kernel。CPU reference 应首先验证该 ABI，再为 SIMD/GPU target 添加 lowering 与 capability gating。

## 设备生命周期

DeviceHandle 创建 runtime session。Session 管理 allocator、queue、event pool、kernel cache 和 telemetry context。Session 关闭前必须 drain 所有 stream；关闭失败要报告未完成任务和资源 lease。

## 能力声明

每个 backend 返回 CapabilitySet：dtype、layout、原子操作、矩阵指令、最大 rank、最大 workspace、异步能力、peer access 和 deterministic support。编译器只能使用 CapabilitySet 允许的路径。
