# 类型系统与数据模型

## 基础类型

`titan-types` 定义跨 crate 共用的纯数据契约，不能持有设备句柄或执行逻辑。

```rust
pub trait DType: Copy + Send + Sync + 'static {
    const ID: DTypeId;
    const SIZE: usize;
    const ALIGN: usize;
}

pub trait ShapeSpec {
    const RANK: Option<usize>;
    fn constrain(&self, env: &mut ShapeEnv) -> TitanResult<()>;
}
```

稳定标识包括：`DeviceId`、`StorageId`、`TensorId`、`TermId`、`GraphId`、`ValueId`、`OperatorId`、`KernelId`、`ParameterId`、`RunId`、`CheckpointId`。标识在进程内不可复用；跨进程工件使用带 namespace 的可序列化 ID。

## Shape 表达

shape 维度有三种：

- `Const<N>`：编译期常量。
- `Symbol<S>`：同一图中的符号维度，例如 batch、sequence。
- `RuntimeDim`：只有执行时才能确定的维度。

约束系统支持相等、广播、乘积、整除、上下界和布局对齐。约束求解失败必须指出冲突的两个 source span 和推导路径。

```rust
Tensor<B, F32, Shape<(Symbol<Batch>, Const<128>)>>
```

Rank 必须静态可知时使用 rank 参数；控制流输出 rank 不确定时使用 `DynShape`。不得用静态 shape 牺牲动态 batch、动态序列长度和模型易用性。

## DType 与提升规则

每个算子声明：允许输入 dtype、输出 dtype、accumulation dtype 和 backend capability。提升规则是全局版本化协议，不由各后端自行决定。

- 整数与浮点混合时提升到可表示的浮点类型。
- `f16`、`bf16` 的 reduce 和 MatMul 默认使用至少 `f32` accumulation。
- loss、norm、softmax 和统计量允许声明更高精度 accumulation。
- 有损 cast 必须显式或由 `PrecisionPolicy` 授权。
- 量化 dtype 携带 scale、zero point、axis 和校准版本。

## Device 与 Layout

`Device` 描述逻辑设备，`DeviceCapabilities` 描述能力，`DeviceHandle` 属于 HAL 私有实现。公共 device 类型不暴露厂商裸句柄。

Layout 包含：维度顺序、stride、blocking、alignment、contiguous 范围和 backend tag。通用图 pass 只能依据公共 layout 属性优化；backend-specific pass 可以读取 backend tag。

## Storage、Tensor 与 View

```text
Storage: 拥有字节区域和设备资源
Tensor:  拥有或共享 Storage 的逻辑张量描述
View:    借用 Storage 的 shape/stride/layout 投影
Term:    不持有可变数据的延迟计算节点
```

Storage 的释放由 RAII 完成。异步执行时，runtime event 持有资源 lease，直到设备确认操作完成。不得仅依赖 Rust lexical lifetime 释放仍被 GPU 使用的内存。

View 不得延长 mutable alias。in-place 写入需要 alias analysis 证明：没有活跃读取者、layout 兼容、autograd 保存值不被覆盖、分布式 bucket 未引用该 storage。

## 版本与 Schema

跨进程或持久化类型包含：

- `schema_name`
- `major_version`
- `minor_version`
- `producer_version`
- `required_features`
- `payload_checksum`

Major 不兼容必须拒绝；minor 字段只能向后兼容增加；未知 required feature 必须拒绝；未知 optional 字段可以忽略。
