# Kernel DSL 与 ABI

## Kernel 声明

```rust
#[kernel(
    backend = Auto,
    block_size = 256,
    vector_width = 4,
    pipeline_depth = 2,
    shared_memory_padding = 1,
)]
fn matmul(a: TensorRef<F16>, b: TensorRef<F16>, out: TensorMut<F32>, spec: MatmulSpec) { ... }
```

过程宏会在编译期拒绝未知属性、`block_size`/`vector_width`/`pipeline_depth` 的零值、非法 backend，以及把 `#[kernel]` 用在非函数上的声明。`shared_memory_padding = 0` 表示不补齐，是有效的默认值。它只生成稳定 metadata；kernel lowering 仍由编译器完成。

Kernel DSL 描述数学语义和访问约束，不直接写厂商指令。手工 backend kernel 可以请求寄存器分块、tensor instruction、异步 copy 和 shared-memory layout，但必须同时提供能力要求和 CPU reference。

## ABI

Kernel ABI 包含入口名、参数顺序、参数类型、shape/stride 参数、workspace、alignment、launch、同步和错误返回。所有参数使用序列化的 ABI schema，运行时依据 schema 检查，禁止按 Rust 内存布局猜测。

## Source Map

生成代码映射回 kernel 声明、Term source span、Graph operator 和 launch config。编译错误必须能从设备编译器位置回到用户 Rust 文件。

## 正确性

每个 kernel 至少有 scalar CPU reference、随机 shape、边界 shape、非 contiguous layout、不同 dtype、NaN/Inf 和空输入测试。优化 kernel 失败时只允许回退到已验证实现。
