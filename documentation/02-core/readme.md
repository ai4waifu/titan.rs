# 核心计算语义

- [类型系统与数据模型](type-system.md)
- [Tensor、Storage 与 View](tensor-storage.md)
- [Term 表达式 DAG](term.md)
- [反向自动微分](autograd.md)
- [Parameter 与 Optimizer](parameters-optimizer.md)
- [DataLoader 与数据状态](dataloader.md)
- [连续 eager CPU 算子](eager-cpu-ops.md)

核心层只定义可跨后端保持一致的计算、梯度和状态语义；设备执行、图优化、分布式与工具行为分别由后续目录定义。
