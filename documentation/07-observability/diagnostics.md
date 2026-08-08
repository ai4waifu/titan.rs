# 诊断规则

诊断引擎读取 telemetry、manifest、执行计划和集群快照，输出稳定的 issue code、级别、证据范围、影响、建议和可重复的检测版本。

## 规则分级

- `fatal`：状态不可恢复或数据一致性已破坏，例如 checkpoint checksum 错误、collective 序列不一致。
- `error`：本次运行无法继续，但可以从有效 checkpoint 恢复，例如设备丢失、全局超时。
- `warning`：运行可继续但有明显风险，例如 rank skew、显存碎片、通信未重叠。
- `info`：配置和性能事实，不代表异常。

## 必备规则

`COLLECTIVE_SEQUENCE_GAP` 检查 collective 序列；`CHECKPOINT_INTEGRITY` 检查 manifest 和 shard；`RANK_SKEW` 比较 rank step 和通信等待；`MEMORY_FRAGMENTATION` 比较 reserved/allocated；`KERNEL_CACHE_MISS` 统计编译缓存；`DATALOADER_STARVATION` 检测预取队列为空；`NUMERIC_OVERFLOW` 聚合 NaN/Inf 和 scaler 状态；`TOPOLOGY_MISMATCH` 比较拓扑摘要与执行计划。

## 报告格式

报告包含 `report_id`、生成时间、工具版本、输入快照摘要、issue 列表和退出码。证据使用 record id、时间范围和字段路径引用，不复制敏感 payload。相同 issue 在相同输入摘要下必须产生稳定排序。
