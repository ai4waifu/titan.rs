# Term 表达式 DAG

`Term` 表示尚未物化的计算关系。它不拥有可变数据，也不等同于 Graph IR；Term 面向用户组合和自动微分，Graph IR 面向编译优化和执行计划。

## 节点字段

每个节点包含 TermId、opcode、输入 TermId、版本化 attributes、dtype、shape 约束、device/layout 约束、source span、effect、requires-grad 和可选常量摘要。

## Effect

effect 分为纯计算、随机数、状态读、状态写、通信和外部调用。纯节点可以 hash-cons 和公共子项消除；其他节点读写显式 effect token，编译器必须保留 happens-before。

随机 Term 记录 RNG stream 和 counter 范围，不使用无法恢复的线程局部随机状态。

## 构造与求值

`tensor!` 解析数学语法并调用普通 Term builder。Builder 立即检查 opcode/attribute，收集 shape/dtype 约束，在 `.eval()`、`.compile()` 或 `.run()` 边界完成求解并生成可定位错误。

## 共享与 identity

TermId 在 arena 内稳定且不复用。结构哈希只包含语义字段，不包含指针和日志信息；有 effect 的节点即使字段相同也不能合并。跨进程序列化使用 GraphId namespace。

## 降低到 Graph IR

lowering 先拓扑排序，物化输入/常量，建立 SSA Value，再转换 effect token、shape guard 和 source map。一个 Term 可以生成多个 IR 节点，但 mapping 必须可用于诊断和 profiler。
