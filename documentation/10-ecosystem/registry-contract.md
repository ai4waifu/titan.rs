# 模型注册与包契约

`titan-model` 定义抽象模型生命周期和稳定的发现协议；`titan-models` 只承载具体模型族、配置、权重映射与模型级验证。二者通过显式注册连接，不使用链接期构造器、全局可变单例或文件系统扫描发现模型。

## 稳定标识

每个可加载模型必须具有 `ModelFamilyId`、`ModelVariantId` 和 schema version。family 表示跨版本的模型族，例如 `language`；variant 表示架构变体，例如 `transformer`。参数名称、输入输出 schema、权重格式和预处理语义共同构成模型兼容性边界。

`ModelDescriptor` 是注册表返回的最小只读描述，其中必须包含：

- family 与 variant；
- training、generation、streaming、native、wasm capability；
- 版本化 input/output schema；
- 模型包 manifest 的 schema 版本。

对 HTTP 目录的投影使用 `ModelCatalogEntry`，并明确包含 `ModelManifestSummary`：manifest `schema_version`、`ready|missing|invalid` 状态和允许的 `native|wasm` deployment targets。注册表不声明运行健康；运行健康只由受观测 API 返回的 `RunStatus` 表示，防止静态包元数据被误当成即时运行状态。

应用在加载权重或调度请求前先查询 descriptor。未知 family/variant、能力不满足、schema 版本不兼容或 manifest 校验失败必须在执行前返回结构化错误；不得以静默降级改变模型结果。

## 注册方式

每个具体模型 crate 暴露 `registry()` 或接收调用方传入的 `&mut ModelRegistry`。注册顺序可见且可测试；同一 `(family, variant)` 重复注册是错误。应用仅链接需要的模型 crate，因此不产生不需要模型族的后端、权重格式或数据集依赖。

`titan-models` 的初始注册表覆盖 vision、language、recommendation、reinforcement、forecasting、audio 与 graph 七个族。它只提供描述符和扩展点；每个具体实现必须在所属子 crate 中交付配置、loader、权重转换、预处理、评估和基准，不把真实权重放入源代码包。

## 加载边界

`ModelLoader` 是外部模型包进入 Titan 的唯一抽象边界。loader 必须校验 manifest、family、variant、schema、参数 key、张量摘要、量化元数据和目标后端 capability，随后生成可审计的加载报告。模型定义不携带 GPU 层数、缓存页数、设备放置或预取深度；这些值由部署配置和运行时资源预算决定。

模型包按 manifest、config、权重块索引、权重块和可选 tokenizer/feature asset 组织。manifest 必须支持部分读取、文件 checksum 和迁移器选择。权重文件损坏、缺失或版本不兼容时，loader 只能从已验证副本恢复，不能构造部分有效模型。

## 目录约定

```text
titan-models/
  src/                 # 显式注册和共享模型描述符
  vision/<model>/      # 具体模型 crate 或模块
  language/<model>/
  recommendation/<model>/
  reinforcement/<model>/
  forecasting/<model>/
  audio/<model>/
  graph/<model>/
```

每个 `<model>` 目录必须包含 `config`、`model`、`weights`、`preprocess`、`evaluate`、`deployment`、`tests` 与 `bench` 职责边界。源码结构可以因语言或构建系统调整，但不能省略相应交付物。
