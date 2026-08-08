# 并发调度与资源

## Stream 类别

- Compute：前向、反向和 kernel 计算。
- Communication：AllReduce、ReduceScatter、AllGather 和点对点。
- Transfer：Host/device、device/device 和 offload。
- Checkpoint：压缩、校验和、写入和 manifest commit。

调度器为每个任务建立依赖 DAG。任务只有在输入 event 完成、资源 lease 有效、collective sequence 正确时才能提交。

## Future 与取消

Future 必须携带 output lease 和取消 token。取消流程先阻止新提交，再等待可安全停止的设备边界，最后释放 lease 和写入失败事件。不能在设备仍访问时强行 Drop storage。

## 背压

队列长度、待处理 bytes、collector backlog、checkpoint backlog 和通信 credit 都有上限。达到上限时优先暂停生产者、降低 telemetry 采样或延迟非关键 checkpoint，不得无限增长内存。

## CPU 回退

后端能力不足时只有在算子声明允许 fallback 且数据迁移成本在策略阈值内才回退 CPU。回退事件必须进入 trace，不能静默改变设备路径。
