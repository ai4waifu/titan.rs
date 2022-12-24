# Typed Graph IR

编译器的用户可见输入 IR 始终称为 `Term`。Term DAG 经过下列阶段生成内部 typed graph；内部节点和值是实现细节，不得作为第二个公共表达式 IR 暴露。

## Node 与 Value

Graph 由不可变 Node 和 Value 组成。Node 记录：`OperatorId`、输入 ValueId、输出 ValueId、属性、effect token、source span、dtype rule、shape rule、gradient rule 和 backend capability。

Value 记录：dtype、shape expression、stride expression、layout、device placement、alias set、producer、consumer count、lifetime、gradient relation 和 storage requirement。

## Core Graph IR 合同（`titan-graph`，schema = 1）

第一版不可推翻的字段（实现：`projects/titan-graph`）：

| 字段 | 说明 |
|------|------|
| `schema` | `GRAPH_SCHEMA_VERSION`；不匹配 → `DXO_IR_SCHEMA_INVALID` |
| `inputs` / `outputs` | ValueId 列表；至少一输出 |
| `values` | `ValueId → TensorSpec`（dtype/shape/strides/layout/alias） |
| `nodes` | `NodeId` + operator + I/O + attrs + effects + source |
| `constraints` | `SameDtype` / `SameShape` / `Custom` |
| `debug` | 不进入 `semantic_hash` |
| `semantic_hash` | FNV-1a64 over canonical JSON（排除 debug） |
| 序列化 | debug JSON roundtrip（调试用，非生产 ExecutablePlan） |

验证失败产出 Living `15` JSON（`IrDiagnostic`，码如 `DXO_IR_SHAPE_CONSTRAINT_UNSAT`）；Titan 不翻译。

Pass 合同见 `PassDecl` / `PassRegistry`（name · stage · invariants · failure behavior）；`builtin_pass_registry()` 为骨架，不含执行器。

## Effect Token

纯计算节点没有 effect。随机数、状态读写、通信、外部调用和同步节点携带 effect token。两个共享 effect token 的节点不能被重排或跨 stream 执行。

## Shape Constraint

编译器构建约束环境：维度相等、广播兼容、矩阵乘积、中间维度、对齐、workspace 上限和设备限制。约束求解输出 substitution、runtime assertion 或不可满足错误。

## Graph 生命周期

```text
Term DAG
 -> normalized graph
 -> constrained graph
 -> differentiated graph
 -> optimized graph
 -> partitioned graph
 -> scheduled graph
 -> executable plan
```

每个阶段拥有不可变输入和版本化输出。缓存只允许复用具有相同 schema、输入约束、编译选项和 capability fingerprint 的阶段结果。

## Graph 序列化

调试序列化包含节点、value、约束、pass report 和 source span；生产 artifact 只包含经过校验的 executable plan 和必要元数据。调试内容不得被生产 loader 当作可执行代码。
