# 模型与训练入口

## 模型生命周期

模型有五个上下文状态：`Constructed`、`Training`、`Evaluation`、`Frozen`、`Disposed`。状态转换只能通过公开方法完成；已 disposed 的 Parameter、Tensor 和 Runtime 不得再次使用。

## 发现与加载

抽象层以 `ModelFamilyId`、`ModelVariantId`、`ModelDescriptor`、`ModelSchema`、`ModelCapabilities`、`ModelRegistry` 和 `ModelLoader` 表示具体模型发现与加载。注册表由应用显式组合，重复 family/variant 视为错误；库不依赖全局构造器或目录扫描。

`ModelLoader::load_manifest` 在模型进入运行时前验证模型族、变体、schema、参数 key、权重摘要与目标 capability。模型 API 只表达结构、输入输出和模型状态；量化、设备放置、分层缓存、预取与并行策略属于部署配置和运行时计划。

详细模型包协议见[模型注册与包契约](../10-ecosystem/registry-contract.md)。Web UI 仅消费描述符与诊断 API，契约见[模型与 Web UI 契约](../10-ecosystem/webui-model-contract.md)。

```text
Constructed -> Training <-> Evaluation
Training/Evaluation -> Frozen
Frozen -> Training
所有状态 -> Disposed
```

`train()` 和 `eval()` 递归传播到子模块。`Frozen` 禁止参数更新但允许 forward；切换设备或 dtype 不能隐式改变训练状态。

## Forward 契约

Forward 接口声明输入 shape、dtype、device、layout 约束和输出 schema。模型可以返回一个或多个带名字的 Term，不得以无标签 tuple 隐藏多输出语义。

模型可以接受运行时配置，但配置必须在 RunId 中记录。随机模型必须接收 RNG context，不能读全局随机源。

## Trainer Step

```text
DataLoader::next
  -> 输入校验与设备搬运
  -> forward Term
  -> loss Term
  -> compile forward/backward graph
  -> execute forward
  -> execute backward
  -> unscale 与 overflow all-reduce
  -> gradient clip
  -> gradient bucket collective
  -> optimizer update
  -> scheduler update
  -> zero_grad / retain policy
  -> telemetry
  -> checkpoint policy
```

每个 step 生成 StepReport：loss、有效样本数、梯度范数、overflow、学习率、kernel、通信、显存、耗时、数据顺序摘要和 checkpoint 状态。

## Optimizer

Optimizer 只接受 ParameterTree 和 GradientTree，不接受任意 Tensor。每个 optimizer state 绑定 ParameterId 和 dtype。更新前验证：shape、dtype、device、finite、step 和 world size。

## DataLoader

DataLoader 的分片、shuffle、seed、worker、prefetch 和 drop_last 配置全部写入 RunManifest。每个 epoch 输出 permutation hash。恢复 checkpoint 时必须从保存的 epoch、batch、worker cursor 和 seed 恢复，禁止重新随机生成顺序。
