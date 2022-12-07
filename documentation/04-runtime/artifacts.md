# 运行工件与状态

所有工件由 ArtifactStore 管理，不允许业务代码直接拼路径。Store 提供临时写、checksum、fsync、原子 rename、版本检查、读取锁和垃圾回收。

## 工件集合

- `run.manifest`：RunId、模型、数据、设备、配置、代码版本和启动参数。
- `graph.plan`：typed graph、优化报告、执行计划和 kernel 引用。
- `autotune.tune`：版本化 TuneEntry 集合。
- `checkpoint/manifest`：checkpoint 版本、step、shard、optimizer、RNG 和 DataLoader state。
- `weights/`：带 checksum 的参数 shard。
- `deployment.manifest`：部署目标、后端、能力、权重和 graph artifact。
- `telemetry/`：结构化 trace、metrics 和事件批次。
- `diagnostics.json`：`tt debug` 生成的只读诊断快照。

## 提交协议

1. 写入临时目录并生成每个文件 checksum。
2. 对所有文件执行长度、schema、dtype、shape 和 checksum 校验。
3. 写入临时 manifest，执行 flush/sync。
4. 原子切换版本指针。
5. 写入 committed 事件。
6. 清理旧版本，保留用户指定数量。

进程崩溃只能留下临时目录，读取器不得把未 commit 文件当作有效状态。
