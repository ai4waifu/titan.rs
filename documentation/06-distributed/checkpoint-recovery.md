# 检查点与恢复

检查点是可验证的版本化目录，由 manifest 作为唯一入口。目录中至少包括模型参数分片、optimizer 分片、训练标量、RNG 状态、DataLoader 游标、并行策略、拓扑 epoch、代码和协议版本。

## Manifest

manifest 固定记录 `checkpoint_id`、父 checkpoint、`run_id`、全局 step、模型和图版本、world size、`ShardSpec`、每个文件的大小与 checksum、提交时间和生成器版本。字段新增必须递增 schema version，并保留向后读取规则。

## 两阶段提交

1. 每个 rank 将分片写入带 nonce 的 staging 目录，分块计算 checksum 并执行 flush。
2. rank 向控制面发送 `Prepare`，包含文件清单和本地校验结果。
3. 控制面收齐并验证所有清单后生成 commit token。
4. rank 原子写入本地完成标记，控制面原子替换根 manifest 并广播 `Commit`。

只有根 manifest 指向的分片可被读取。提交失败时 staging 目录进入可回收状态，不得覆盖上一个有效版本。

## 恢复协议

当前本地恢复器在解码 checkpoint 前调用 `CheckpointManifest::validate_recovery`。它要求 manifest 已提交、run id 与 step 精确匹配、shard 数非零，并要求 payload 的确定性 checksum 与 manifest 一致。验证失败不会开始恢复。

本地契约不创建 rendezvous、不重分布参数，也不访问远端 checkpoint 存储。

## 增量与保留

增量 checkpoint 记录相对父版本的 changed shard，读取时沿父链合并并 materialize 为可独立恢复的逻辑视图。保留策略按最近 N 个、时间窗口和显式标签执行，删除前必须确认没有活动恢复任务引用该版本。
