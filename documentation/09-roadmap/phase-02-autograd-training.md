# 阶段 02：Term、Autograd 与训练组件

## 前置条件

- 阶段 01 的 Tensor、设备、错误和标识稳定。
- 算子 schema 可以描述输入、属性、shape、dtype 和 effect。
- 明确 eager 值与符号 `Term` 的转换规则。

## 代码交付物

- hash-consed Term DAG、常量、输入、算子调用和控制依赖。
- reverse-mode Autograd、拓扑排序、梯度累加和 VJP 注册表。
- Parameter、Buffer、ParameterTree、Optimizer 和状态字典。
- SGD/AdamW、mixed precision scaler、梯度裁剪和数值检查。
- DataLoader 的 sampler、batch、prefetch、worker、RNG 和恢复游标。
- `#[neural]`、`#[parameters]` 和 `tensor!` 宏的解析与展开。

## API 交付物

- `titan-model::Model` 的 forward、parameters、train/eval 和 state 接口。
- `Tensor::backward`、`grad`、`no_grad`、`detach` 和 retain graph 语义。
- Optimizer 的 step、zero_grad、state_dict/load_state_dict。
- DataLoader 的迭代、暂停、恢复和确定性配置。

## 测试交付物

- 每个可微算子的 VJP 已知答案和 finite-difference 测试。
- 分支 DAG、共享子图、多输出、stop-gradient 和重复 backward 测试。
- Parameter 遍历顺序、别名、状态保存和 optimizer 恢复测试。
- DataLoader 多 worker、尾 batch、shuffle、故障和恢复顺序测试。
- 宏的 compile-pass、compile-fail、错误 span 和泛型展开测试。

## 性能交付物

- Term 构建、拓扑排序和 backward 调度开销基线。
- 训练一步的 eager 吞吐、内存峰值和 DataLoader 等待比例。

## 文档交付物

- 完成 Tensor/Term/Autograd、模型训练 API 和宏职责文档。
- 写明梯度、mixed precision 和状态恢复的不变量。

## 失败条件

- VJP 规则依赖运行顺序或隐式全局 tape。
- Parameter 遍历不稳定，导致 checkpoint key 变化。
- mixed precision 溢出后仍更新参数。
- DataLoader 无法准确恢复下一个样本。

## 完成验收

使用 `#[neural]` 定义模型，经 DataLoader 完成多步 forward、backward 和 AdamW 更新，保存并恢复后得到相同的下一步 loss 和参数摘要。

## 解锁条件

Term、effect、参数 key 和训练状态 schema 稳定，允许图编译器捕获完整训练子图。
