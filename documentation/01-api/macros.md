# 宏框架

## 宏职责

宏名称按职责拆分，禁止使用一个带大量子模式的总宏：

```rust
#[neural]
struct Decoder<B: Backend> { ... }

#[parameters]
struct DecoderParameters<B: Backend> { ... }

#[kernel(block_size = 256, backend = Auto)]
fn attention_kernel<B: Backend>(...) { ... }

#[distributed(strategy = "fsdp")]
struct ShardedDecoder<B: Backend> { ... }

tensor! { exp(q @ k.transpose(0, 1)) }
```

## `#[neural]`

当前稳定展开契约：宏保留原结构体或函数，并生成同名规则的隐藏元数据常量 `__TITAN_<名称>_META`，值为 `neural:<名称>`。该常量供模型注册、文档生成和诊断索引使用；宏不隐式改变字段布局或执行设备。`#[neural]` 不接受属性参数。

输入可以是结构体或函数。模型层可据此元数据建立 `Forward`、`Module`、子模块遍历、设备迁移和图捕获；这些运行时行为不由属性宏隐式生成。宏不得生成隐藏的 storage、线程、设备句柄或全局可变状态。

## `#[parameters]`

当前稳定展开契约：宏只接受结构体，且不接受属性参数；它保留参数结构体并生成 `__TITAN_<名称>_META`，值为 `parameters:<名称>`。参数遍历、梯度槽和 checkpoint 映射由模型层显式实现，不能依赖字段声明顺序。

模型层利用元数据生成稳定的 ParameterId、名称路径、参数树遍历、梯度槽遍历和 checkpoint 映射。字段顺序和名称变更必须导致 schema 差异，加载器可以要求显式迁移。普通 Tensor 不能被隐式猜测为 Parameter。

## `#[kernel]`

当前稳定展开契约：宏只接受函数，并生成 `__TITAN_<名称>_META`，值为 `kernel:<名称>`。允许的属性为 `backend = Auto|CpuSimd|Ptx|Hip|Metal|Wgsl`，以及值大于零的 `block_size`、`vector_width`、`pipeline_depth` 和非负的 `shared_memory_padding`。宏只校验声明并保留源位置；后续 lowering 使用该位置建立 ABI/source map。

宏解析 kernel 参数、内存访问、launch 配置、同步声明和目标后端。输出 kernel IR、source map、ABI schema、候选 config 和 capability requirement。

编译诊断必须包含：宏调用位置、kernel 函数位置、非法访问、shape 使用、未满足 capability、配置冲突和 lowering 建议。

## `#[distributed]`

当前稳定展开契约：宏接受结构体或函数，并生成 `__TITAN_<名称>_META`，值为 `distributed:<名称>`。允许 `strategy = "..."` 与值大于零的 `world`；宏生成分布式 metadata，不隐藏通信调用。rank、拓扑和通信序列由 `DistributedRuntime` 注入。

## `tensor!`

`tensor!` 解析数学语法为 `Term`。`Term` 是唯一的延迟表达式 IR 名称；不得再引入 `Expr`、`TensorExpr` 等同义公共类型。解析器必须保留 token span、运算符优先级、显式 cast、广播方向和变量路径。宏只负责语法树构造；shape/dtype/device 约束由 Term compiler 统一求解。

## 宏版本与错误

宏 crate 单独版本化。错误使用 `syn::Error` 聚合多个问题后一次返回。宏不得读取文件、访问网络、执行外部命令或依赖 `clap`。
