# 生态边界

## 分层

```text
titan-types / titan-hal
        -> titan-tensor / titan-graph / titan-kernel / titan-runtime
        -> titan-autotune / titan-distributed / titan-profiler
        -> titan-model
        -> titan-models
        -> 领域数据、训练配方、评估与部署模板
```

`titan-model` 只定义抽象模型、ParameterTree、Layer、Optimizer、DataLoader、state 和推理生命周期。`titan-models` 存放具体架构、配置、权重映射和模型级测试。领域目录只组合具体模型、数据管线、评估指标和部署模板。

## 公共能力归属

某能力满足两个以上领域时必须进入共享层。例如 tokenizer runtime、图采样、音频变换可以先位于领域模块；通用 streaming、ragged tensor、sparse layout、embedding storage、online state 和 serving scheduler 必须进入明确的基础 crate。

领域包不得直接调用 CUDA/ROCm API，不得定义私有 Tensor/Term，不得绕开 ArtifactStore 保存状态，也不得向核心 crate 引入数据集 SDK、Web UI 或命令行依赖。

## 验证反馈

领域模型通过 capability request 描述缺失算子、layout、稀疏格式、动态 shape、streaming 和 distributed 需求。新增基础能力必须先有跨领域语义、错误、测试和性能门槛，再供具体模型启用。
