# 遥测数据模型

## 统一标识

所有记录使用二进制或 JSON 编码均必须具备以下上下文：

| 标识 | 作用 |
| --- | --- |
| `run_id` | 一次训练、推理或编译会话 |
| `model_id` | 模型结构和参数版本 |
| `graph_id` | 可执行图版本 |
| `operator_id` | 图中的算子实例 |
| `kernel_id` | 选中的 kernel 和变体 |
| `rank` | 分布式成员编号；单机为 0 |
| `step` | 全局优化或推理步 |
| `checkpoint_id` | 已提交状态版本 |

## 三类记录

### Trace

Trace 是有开始和结束时间的区间，适合表达 operator、kernel、graph pass、通信和 DataLoader 阶段。字段包括 parent span、线程/stream、设备、状态和结束原因。

### Event

Event 是瞬时事实，例如 checkpoint commit、OOM、成员加入、collective timeout、数值检查失败和配置变更。事件必须包含单调序列号，允许在时钟不一致时排序。

### Metric

Metric 是带标签的数值样本，例如吞吐、延迟、显存、allocator 碎片、通信带宽、queue depth、loss 和梯度范数。标签集合必须有上限，禁止把 tensor 内容、用户输入或路径作为标签。

## 采样和优先级

四级优先级固定为 `critical`、`important`、`normal`、`debug`。critical 记录不得采样，important 按会话配置采样，normal 使用固定比例，debug 仅在显式开启时收集。随机采样使用会话种子，保证问题复现。

## 脱敏与大小限制

遥测只记录形状、dtype、设备和摘要，不记录输入文本、参数原值、token 内容或密钥。单条记录最大 64 KiB，批次最大 1 MiB；超过限制的 stack、配置和诊断字段必须截断并记录 `truncated=true`。

## 版本化

每种 record 有 `schema_name` 和 `schema_version`。新增字段必须向后兼容，删除字段需要经过一个完整发布周期；读取器遇到未知字段必须忽略，遇到未知主版本必须拒绝解析并给出升级提示。
