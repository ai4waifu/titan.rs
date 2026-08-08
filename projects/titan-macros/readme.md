# titan-macros

Titan.rs 的过程宏 crate，提供 `#[neural]`、`#[parameters]`、`#[kernel]` 和 `#[distributed]` 的编译期声明校验与稳定 metadata 展开。

宏保留原始函数或结构体，并生成稳定的隐藏元数据常量；不创建设备句柄、线程、存储或全局可变状态。`#[neural]` 接受函数或结构体，`#[parameters]` 仅接受结构体，`#[kernel]` 仅接受函数，`#[distributed]` 接受函数或结构体。kernel 和 distributed 的允许属性与诊断规则见 API 宏文档。

模型遍历、参数字段生成、通信执行、kernel lowering 和 `tensor!` 的 `Term` 构造均属于各自的运行时或编译器组件，不由本 crate 隐式实现。
