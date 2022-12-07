# 阶段 04：Kernel ABI 与真实设备后端

## 前置条件

- ExecutionPlan、Tensor layout、stream/event 和内存计划稳定。
- operator schema 可以 lowering 到 kernel 描述。
- 设备 capability 查询和错误模型完备。

## 代码交付物

- `#[kernel(...)]` 宏、Kernel IR、参数 ABI、launch geometry 和 workspace 描述。
- CPU、CUDA 和 ROCm backend 的编译、加载、launch、event 和错误映射。
- elementwise、reduction、MatMul、normalization、attention 基础 kernel 集合。
- kernel cache、二进制校验、目标架构匹配和 backend fallback。
- Native backend 的异步执行和多 stream 依赖。

## API 交付物

- 后端注册、设备枚举、capability 查询和 kernel launch 接口。
- kernel 作者可使用受限 DSL 定义索引、共享内存、同步和参数约束。
- 不支持的 dtype/layout/capability 返回结构化错误。

## 测试交付物

- ABI 对齐、参数布局、launch 边界、空 tensor 和大索引测试。
- CPU/CUDA/ROCm 的已知答案与交叉后端数值测试。
- 非连续 layout、尾块、共享内存上限、设备重置和加载失败测试。
- kernel 二进制损坏、架构不匹配和签名错误测试。

## 性能交付物

- 代表 shape 的 kernel 吞吐、带宽、occupancy 摘要和 launch 开销。
- MatMul、reduction 和 attention 的 roofline 位置与瓶颈说明。
- Native 大模型路径的显存与吞吐基线；WASM 只测轻量场景。

## 文档交付物

- 完成 Kernel DSL/ABI、backend capability 和编译缓存文档。
- 列出每个 backend 的算子、dtype、layout 和确定性支持矩阵。

## 失败条件

- ABI 依赖 Rust 内存布局而未固定版本。
- kernel launch 隐式同步整个设备。
- backend 错误丢失 kernel id、设备或 launch 参数。
- Native 执行必须经过浏览器或 WASM 路径。

## 完成验收

同一 ExecutionPlan 在 CPU 和至少一个 Native GPU 后端执行，基础训练算子结果满足误差约束，异步 event 和内存生命周期无泄漏。

## 解锁条件

kernel 变体、性能测量和缓存 identity 稳定，允许自动调优系统选择实现。
