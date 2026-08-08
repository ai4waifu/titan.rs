# DataLoader 与数据状态

DataLoader 把数据源、sampler、变换、batch、worker 和 prefetch 组成可恢复流水线。训练正确性依赖确定性的 sample identity，而不是线程完成顺序。

## 数据源协议

Dataset 提供稳定 dataset id、schema version、长度或流式游标、sample id 和 checksum/版本。Map-style 数据集按 index 读取；streaming 数据集通过 partition、offset 和 watermark 定位。

## 采样

`Sequential`、`ShuffleEveryEpoch`、`ShuffleOnce` 和分布式 shard sampler 都记录 seed、epoch、permutation hash、rank/world 和已消费游标。world 变化时恢复器按 sample id 重分片，避免重复或遗漏未提交 batch。

## Worker 与 Prefetch

worker 负责 I/O 和纯变换，结果带 batch sequence 进入有界队列。主训练线程按 sequence 消费；慢 worker 可以阻塞对应位置但不能改变顺序。pinned buffer、decode workspace 和预取深度受 ResourceBudget 约束。

## Batch 提交

batch 在 optimizer step 提交后才标记 consumed。step 失败并回滚时 DataLoader 可以重放同一 batch 及 RNG；显式跳过坏样本必须记录 sample id、原因和策略。

## Checkpoint 状态

保存 dataset/version、sampler、epoch、partition、offset、permutation hash、worker RNG、变换 state、已提交 batch 和下一 sequence。恢复先验证数据版本，再恢复队列边界。

## 观测与测试

记录读取、解码、变换、组 batch、queue wait、H2D 和空队列时间。测试覆盖多 worker、尾 batch、shuffle、流式 watermark、worker 崩溃、坏样本和 checkpoint 恢复。
