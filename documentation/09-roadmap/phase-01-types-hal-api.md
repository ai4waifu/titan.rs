# 阶段 01：共享类型、HAL 与公共 API

## 前置条件

- 固定 nightly toolchain、Cargo workspace 和 crate 命名。
- 明确 CPU、CUDA、ROCm、WASM 的目标边界。
- 确定 `titan-types` 作为共享底层库，`titan` 作为 facade。

## 代码交付物

- `DType`、`Shape`、`Stride`、`Layout`、`DeviceId`、`TensorId`、`RunId` 等稳定类型。
- `Storage`、`Tensor`、`TensorView` 的所有权和 alias 规则。
- HAL 的 device、allocator、stream、event、copy、launch 接口。
- CPU HAL，支持分配、拷贝、逐元素、reduction 和 MatMul。
- 分层错误类型，包含设备、shape、dtype、编译、执行和资源错误。

## API 交付物

- `titan` 统一导出 Tensor 构造、设备选择、基础算子和错误类型。
- `Device::cpu/cuda/rocm`、`Tensor::zeros/from_slice/to_device` 和显式同步 API。
- 公共 API 不暴露具体 allocator、backend handle 和内部 graph 节点。

## 测试交付物

- shape/dtype/layout 的单元和属性测试。
- view、slice、transpose、contiguous、跨设备 copy 的生命周期测试。
- CPU MatMul 已知答案、空维、非连续输入和溢出错误测试。
- allocator 重用、双重释放保护、event 顺序和多线程测试。

## 性能交付物

- CPU 分配、copy、elementwise 和 MatMul 基线。
- 小 tensor 调用开销、峰值内存和 allocator 命中率报告。

## 文档交付物

- 完成类型系统、Tensor/Storage、HAL 和公共 facade 规范。
- 列出每个 dtype、layout 和设备能力的支持矩阵。

## 失败条件

- 同一概念在多个 crate 有不兼容定义。
- view 可以越界或在 storage 释放后继续访问。
- HAL 需要上层模型或编译器类型才能使用。
- CPU 基线无法完整执行最小算子链。

## 完成验收

在 CPU 上通过公共 `titan` API 构造 tensor、执行 MatMul、读取结果并可靠释放资源；全 workspace 格式、lint 和测试通过。

## 解锁条件

Tensor 标识、shape/layout 语义和 HAL stream/event 契约稳定，允许 Term 和 Autograd 依赖。
