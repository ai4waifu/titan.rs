# 自动调优

## 搜索对象

每个 OperatorStrategy 定义输入约束、候选参数、合法性检查、预热次数、采样次数、停止规则、失败回退和结果合并。

MatMul 候选维度包括 tile M/N/K、block size、vector width、pipeline depth、shared-memory padding、workspace、矩阵指令和 split-K。Conv、Reduce、Attention 使用各自策略，不共用无意义的参数集合。

## 测量协议

1. 读取 capability 和硬件 fingerprint。
2. 生成并过滤候选。
3. 预热设备和 allocator。
4. 对候选执行多轮测量，丢弃异常值。
5. 检查 correctness 和资源上限。
6. 使用中位数、尾延迟和置信区间选择结果。
7. 写入版本化 `.tune` 并发送 telemetry。

## TuneEntry

`.tune` 是唯一持久化格式。文件首行固定为 `# titan.tune version=1`；以 `#` 开头的后续行是注释，空行忽略。解析器只接受已知 schema 的记录，不能把旧的 `.cache` 文件作为输出目标。调用方传入其他扩展名时，运行时规范化为同路径的 `.tune` 文件，以防重新产生旧格式。

```text
schema
operator
backend
device_fingerprint
driver
graph_shape
dtype
layout
candidate
measurement_count
median_ns
p95_ns
correctness_hash
created_at
runtime_version
```

生产反馈必须带 RunId、KernelId、TuneKey 和采样上下文。新结果只有在样本量、置信度和 correctness hash 均满足条件时才能替换 incumbent。
