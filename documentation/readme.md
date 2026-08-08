# Titan.rs 开发者文档

本文档是 Titan.rs 的唯一开发者规范，定义最终功能、公共接口、内部架构、运行协议、工程约束、模型生态和实施路线。

## 文档结构

```text
documentation/
├── 00-overview/       # 产品边界、最终功能、总体架构、术语
├── 01-api/            # facade、Tensor/Term、模型、宏和训练 API
├── 02-core/           # 类型系统、Tensor、Term、Autograd、参数、DataLoader
├── 03-compiler/       # 图 IR、约束求解、优化 pass、内存规划
├── 04-runtime/        # 执行状态机、并发调度、资源与工件
├── 05-kernels/        # Kernel DSL、ABI、后端、编译缓存、自动调优
├── 06-distributed/    # 控制面、数据面、collective、并行与 checkpoint
├── 07-observability/  # 遥测协议、Profiler、tt.exe、Web UI
├── 08-engineering/    # workspace、测试、安全、兼容性和发布
├── 09-roadmap/        # 实施顺序、工作包、验收门槛
└── 10-ecosystem/      # 抽象模型、具体模型和七个领域交付规范
```

## 阅读顺序

1. [产品定位](00-overview/vision.md)
2. [最终功能规格](00-overview/features.md)
3. [总体架构](00-overview/architecture.md)
4. [核心术语](00-overview/glossary.md)
5. [公共 API](01-api/readme.md)
6. [核心计算语义](02-core/readme.md)
7. [图编译器](03-compiler/readme.md)
8. [运行时](04-runtime/readme.md)
9. [内核、后端与调优](05-kernels/readme.md)
10. [分布式系统](06-distributed/readme.md)
11. [可观测性与工具链](07-observability/readme.md)
12. [工程治理](08-engineering/readme.md)
13. [实施路线图](09-roadmap/readme.md)
14. [模型与领域生态](10-ecosystem/readme.md)

## 规范用语

- “必须”：实现和评审不能偏离。
- “不得”：架构禁止项。
- “应当”：默认选择；偏离时需要设计记录和测试依据。
- “可以”：兼容设计的可选实现。
- “完成”：通过对应目录定义的全部验收门槛，不以类型存在或示例可编译为判断依据。
