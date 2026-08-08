# 阶段 08：具体模型与领域生态

## 前置条件

- 抽象模型、训练、Native 推理、分层执行、分布式和观测接口稳定。
- `titan-models` 具有配置、参数 key、权重 manifest 和模型注册协议。
- 领域能力缺口可以通过 capability request 进入共享层评审。

## 代码交付物

- 视觉：数据预处理、Conv/ViT、分类以及至少一个检测或分割闭环。
- 语言：tokenizer、Transformer、生成、KV Cache、MoE 与分层模型加载。
- 推荐：sparse/ragged、Embedding、召回/排序和原子在线版本。
- 强化学习：环境、向量采样、Replay、至少一个 value-based 和一个 actor-critic 算法。
- 时序：统计模型、神经预测、backtest、在线 state 和预测区间。
- 音频：解码/重采样、STFT/Mel、流式识别或分类闭环。
- 图学习：CSR/CSC、邻居采样、消息传递和至少两个 GNN 模型。

模型代码按 `titan-models/{vision,language,recommendation,reinforcement,forecasting,audio,graph}/<model>` 组织。顶层 crate 只显式注册稳定 descriptor；每个 `<model>` 独立交付 config、model、weights、preprocess、evaluate、deployment、tests 与 bench，且不将权重二进制提交到源码包。

## API 交付物

- 统一 `ModelConfig`、`ModelLoader`、`TrainRecipe`、`Evaluator`、`Predictor` 和 `ModelExporter`。
- 每个领域定义版本化 input/output schema、pre/post-processing 和 metric 接口。
- 模型注册表按 family/variant 返回 capability、loader、权重转换器和部署契约。

`ModelLoader` 在加载前校验 manifest、schema、参数 key、量化元数据、checksum 和后端 capability。任何不兼容必须返回可诊断错误，不允许用未声明回退替换模型或权重。

## 测试交付物

- 每个领域至少一个从数据到训练/加载、评估、checkpoint、恢复和 Native 部署的端到端测试。
- 预处理 golden、模型已知答案、梯度、分布式/分层配置和导出等价测试。
- 模型配置、参数 key、权重迁移、未知字段和上一兼容版本测试。

## 性能交付物

- 每个领域至少一个代表工作负载，报告吞吐、p95、峰值内存、冷启动和领域瓶颈。
- 语言额外报告 prefill/decode 与 KV/专家指标；推荐报告 lookup/更新；强化学习报告 samples/s；图学习报告 sampling/edges/s；音频报告 real-time factor。
- 基准必须记录输入、精度、硬件、后端、数据和模型版本。

## 文档交付物

- 完成生态边界、模型库组织、统一交付标准和七个领域规范。
- 每个模型提供 config、数据契约、训练、评估、部署、兼容和故障处理说明。

## 失败条件

- 领域项目复制 Tensor、runtime、checkpoint 或 profiler。
- 模型只有结构或算子示例，没有完整数据与部署闭环。
- 领域依赖反向进入 `titan-types`、HAL 或 kernel 基础层。
- 性能结论缺少硬件、输入、统计和版本记录。

## 完成验收

七个领域各有一个使用统一 Titan API 的可运行闭环，模型和工件可版本化加载、训练/推理、恢复、观测与 Native 部署；公共能力没有领域私有分叉。

验收时对每个模型族执行 descriptor 注册唯一性、manifest 拒绝、配置兼容、预处理 golden、已知答案、checkpoint 恢复、Native 部署和受支持分层/分布式配置测试；基准报告必须绑定 model、weight、schema、backend 与输入版本。

## 解锁条件

基础设施经过多领域验证，API、artifact 和性能指标具备生产发布所需的稳定性与覆盖面。
