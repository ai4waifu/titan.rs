# 公共 API 设计

## Facade 与 Prelude

普通用户依赖 `titan`，由该 crate 统一导出稳定接口：

```rust
use titan::prelude::*;
```

`prelude` 只导出模型编写必需项：`Tensor`、`Term`、`Parameter`、`Backend`、dtype、shape、模型层、optimizer、DataLoader、训练上下文和五类宏。设备厂商 API、编译器内部节点、遥测传输和工具链类型不得进入 prelude。

公共 API 分为四层：

1. 数学层：Tensor、Term、算子、广播、索引和 reduce。
2. 训练层：Model、Parameter、autograd、optimizer、DataLoader 和 checkpoint。
3. 执行层：Device、CompileOptions、ExecutionMode、StreamPolicy 和 DistributedConfig。
4. 运维层：RunConfig、TelemetryConfig、ArtifactStore 和诊断快照。

## 错误边界

用户数学表达式不逐算子返回 `Result`。以下边界可以失败：

- `Term::eval()`：eager materialization。
- `Term::compile()`：约束求解、图优化和内核生成。
- `Executable::run()`：设备执行与同步。
- `Trainer::step()`：完整训练 step。
- `Checkpoint::save/load()`：持久化。
- `DistributedRuntime::join()`：集群加入。

统一错误类型包含：稳定错误码、层级、source span、operator/graph/device/rank 上下文、因果链、可恢复性和修复提示。库代码不得依赖错误字符串判断分支。

## Tensor API

```rust
let x = Tensor::<Cuda, F32, 2>::from_slice([batch, hidden], &values)?;
let y = x
    .view([batch, heads, head_dim])?
    .transpose(1, 2)
    .contiguous()?;
```

Tensor 方法分组：

- 构造：`empty`、`zeros`、`ones`、`full`、`arange`、`from_slice`、`from_iter`。
- 查询：`shape`、`strides`、`dtype`、`device`、`layout`、`numel`、`is_contiguous`。
- view：`reshape`、`view`、`transpose`、`permute`、`squeeze`、`unsqueeze`、`flatten`。
- 索引：`slice`、`select`、`narrow`、`row`、`col`、`gather`、`scatter`。
- 设备与类型：`to_device`、`cast`、`contiguous`、`detach`。
- 训练：`requires_grad`、`grad`、`backward`、`zero_grad`。

同步读取设备 scalar 必须显式调用 `to_scalar()` 或 `read()`；普通算子不能隐式把 GPU 值拉回 CPU。

## Term API

`Term` 是延迟、不可变、可共享的数学项：

```rust
let logits: Term<Cuda, F32, 2> = tensor! {
    input @ weight.transpose(0, 1) + bias
};

let output = logits.eval()?;
let executable = logits.compile(CompileOptions::training())?;
```

Term 允许构图、检查、编译和解释，但不直接暴露可变 storage。Term 的 clone 只复制图引用，不复制 Tensor 数据。Term 的 identity 与 source span 独立，便于公共子表达式和诊断。

## 模型与训练 API

```rust
#[neural]
struct Classifier<B: Backend> {
    encoder: Linear<768, 3072, B>,
    output: Linear<3072, 10, B>,
}

impl<B: Backend> Forward<B> for Classifier<B> {
    type Input = Tensor<B, F32, 2>;
    type Output = Term<B, F32, 2>;

    fn forward(&self, x: Self::Input) -> Self::Output {
        tensor! { gelu(self.encoder(x)) |> self.output }
    }
}
```

训练入口：

```rust
let mut trainer = Trainer::builder(model, AdamW::default())
    .device(Cuda::new(0)?)
    .precision(PrecisionPolicy::MixedBf16)
    .checkpoint(store)
    .build()?;

for batch in loader.epochs(epochs) {
    let report = trainer.step(batch?)?;
}
```

`Trainer::step` 的顺序固定为：取批次、forward、loss、backward、unscale、溢出检测、梯度裁剪、collective、optimizer、scheduler、zero-grad、遥测、checkpoint policy。

## 稳定性分级

- Stable：用户模型、Tensor、训练和部署所需 API，遵循语义版本。
- Advanced：编译选项、设备能力、分布式策略和扩展 trait，可按次版本扩展。
- Internal：IR、pass 节点、厂商句柄和缓存内部格式，不承诺源码兼容。

所有公开类型必须说明线程安全、同步点、内存所有权、确定性和失败语义。
