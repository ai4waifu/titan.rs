# 模型与领域生态

Titan.rs 的领域生态共享同一套类型、Tensor/Term、图编译器、kernel、运行时、训练、分布式、checkpoint 和观测基础设施。领域模块用于验证并扩展公共能力，不创建私有运行时或不兼容的数据/模型接口。

## 文档索引

- [生态边界](architecture.md)：抽象模型、具体模型、领域包和依赖方向。
- [模型库组织](model-library.md)：`titan-model` 与 `titan-models` 的版本和目录契约。
- [模型注册与包契约](registry-contract.md)：显式发现、加载校验、模型包和具体模型目录。
- [模型与 Web UI 契约](webui-model-contract.md)：Vue/Vite 读取模型、运行与资源诊断的 API 边界。
- [统一交付标准](delivery-standard.md)：领域模型必须具备的数据、训练、推理、部署和验证产物。
- [领域目录](domains/readme.md)
  - [视觉](domains/vision.md)
  - [语言与生成式模型](domains/language.md)
  - [推荐系统](domains/recommendation.md)
  - [强化学习](domains/reinforcement.md)
  - [时间序列](domains/forecasting.md)
  - [音频与语音](domains/audio.md)
  - [图学习](domains/graph.md)

## 建设原则

1. 公共缺口先在基础 crate 形成通用接口，再由领域模型消费。
2. 每个模型必须覆盖加载、训练/推理、checkpoint、部署、观测和兼容性。
3. Native 是服务器、工作站和边缘部署的主要路径；WASM 只覆盖轻量推理和交互演示。
4. 性能结论必须来自可复现基准，不写固定提升比例。
