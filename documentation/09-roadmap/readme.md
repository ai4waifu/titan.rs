# 实施路线图

路线图按依赖关系组织，不按日历日期承诺。每个阶段只有通过代码、API、测试、性能、文档和失败处理的全部验收，才能解锁下一阶段。

## 阶段索引

1. [共享类型、HAL 与公共 API](phase-01-types-hal-api.md)
2. [Term、Autograd 与训练组件](phase-02-autograd-training.md)
3. [图编译器与内存规划](phase-03-graph-compiler.md)
4. [Kernel ABI 与真实设备后端](phase-04-kernel-backend.md)
5. [自动调优与分层执行](phase-05-autotune.md)
6. [分布式训练与故障恢复](phase-06-distributed.md)
7. [可观测性、tt.exe 与 Web UI](phase-07-observability-tooling.md)
8. [具体模型与领域生态](phase-08-model-ecosystem.md)
9. [生产化与发布](phase-09-production.md)

## 横向规范

- [依赖图与工作包](dependency-map.md)
- [统一验收门槛](acceptance-gates.md)

## 总体顺序

```text
共享类型
  -> HAL 与公共 facade
  -> Tensor / Term
  -> 通用 Autograd
  -> Parameter / Optimizer / DataLoader
  -> Graph IR
  -> Memory Planner
  -> Kernel ABI
  -> CUDA / ROCm / CPU 后端
  -> Autotune 与 .tune
  -> 分层权重、资源预算与 MoE 调度
  -> Distributed
  -> Checkpoint Recovery
  -> Telemetry
  -> tt.exe
  -> titan-webui
  -> titan-models 与七个领域闭环
  -> Production
```
