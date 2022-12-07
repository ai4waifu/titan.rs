# 模型库组织

## `titan-model`

抽象层提供 `Model`、`Module`、`ParameterTree`、`TrainState`、`InferenceState`、`ModelLoader`、`ModelExporter` 和 `Generation/Serving` 接口。`#[neural]` 生成模块结构元数据，`#[parameters]` 生成稳定参数遍历和 checkpoint key。

## `titan-models`

具体模型按模型族组织：

```text
titan-models/
├── vision/
├── language/
├── recommendation/
├── reinforcement/
├── forecasting/
├── audio/
└── graph/
```

每个模型目录包含 config schema、模型结构、权重映射、预处理契约、示例配置、已知答案、checkpoint/导出迁移和基准清单。权重文件不直接进入源码包，通过版本化 manifest 获取。

## 版本契约

模型版本由 architecture schema、parameter key schema、pre/post-processing schema 和 weight format 共同决定。配置增加 optional 字段可提升 minor；参数布局、tokenizer/feature 语义或输出契约变化必须提升 major 并提供迁移器。

## 注册和发现

模型注册表以稳定 `ModelFamilyId` 和 `ModelVariantId` 查找 loader、capability、输入输出 schema 和权重转换器。注册是显式链接，不使用全局构造器副作用；应用可以只编译所需模型族。

注册、模型包校验和具体目录交付规则见[模型注册与包契约](registry-contract.md)。
