# 阶段 03：图编译器与内存规划

## 前置条件

- Term DAG、算子 schema、shape/dtype/layout 和 effect 语义稳定。
- Autograd 可以生成显式反向图。
- HAL 能表达 stream/event 和内存资源。

## 代码交付物

- SSA 风格 Graph IR、Block、Value、Operator、Region 和 effect token。
- shape 约束求解、dtype promotion、layout propagation 和 capability 检查。
- canonicalization、常量折叠、DCE、CSE、fusion、autograd lowering 和 communication lowering。
- alias/liveness 分析、静态内存计划、workspace 复用和重算标记。
- ExecutionPlan、编译缓存 key、图版本和可序列化诊断。

## API 交付物

- compile/capture 接口，输入模型、示例输入和编译选项，返回可执行计划。
- 图检查 API，可导出稳定的文本/JSON 表示和 pass 统计。
- `CompileError` 包含 operator、shape 约束、设备能力和源码位置。

## 测试交付物

- 每个 pass 的 before/after golden 测试和幂等性测试。
- 动态 shape、广播冲突、effect 排序、alias 和控制流测试。
- planner 的生命周期交叠、对齐、view、workspace、重算和峰值测试。
- eager 与 compiled 的数值、错误和副作用等价测试。

## 性能交付物

- 编译总时长和各 pass 时长基线。
- fusion 前后 launch 数、峰值内存和执行时间对比。
- planner 相对不复用分配的内存节省报告。

## 文档交付物

- 完成 Graph IR、pass 顺序、memory planner 和 compilation 文档。
- 每个 pass 写明前置条件、保持的不变量和失败诊断。

## 失败条件

- pass 依赖未声明的执行顺序或改变 effect 语义。
- planner 复用仍存活或被 alias 的 buffer。
- 动态 shape 失败只能在设备执行后发现。
- 编译缓存 key 漏掉影响语义的配置。

## 完成验收

代表模型的 forward/backward/optimizer 子图可编译为稳定 ExecutionPlan，结果与 eager 一致，并生成可审计的 pass 与内存报告。

## 解锁条件

ExecutionPlan 与 kernel 调用边界稳定，允许真实后端按统一 ABI 实现。
