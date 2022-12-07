# 遥测采集器

当前 `Collector` 是进程内有界事件队列，只负责接收记录、分配单调 sequence、保留容量内事件并统计丢弃数。

## 队列与批处理

事件写入无锁或低锁环形队列，消费者线程按最大条数、最大字节数和最大等待时间触发 flush。critical 记录使用独立保留队列，normal/debug 共享可丢弃队列。

## 背压策略

队列达到 70% 时降低 debug 采样，达到 85% 时停止 debug，达到 95% 时只保留 critical 和重要故障事件。每次降级发出一个 `telemetry_backpressure` 事件；恢复后记录丢弃计数。

## 本地离线缓存

默认写入 `.titan/telemetry/<run_id>/`，文件按小时滚动，采用长度前缀记录和 checksum。进程异常退出时读取器可恢复到最后一个完整记录。磁盘配额达到上限时按优先级和时间淘汰，critical 文件保留到配额保护阈值。

## 本地边界

当前 `Collector` 不发送遥测、不建立远端连接、不使用远端存储或身份凭据。容量耗尽时事件被计数为 dropped，仍消耗一个单调 sequence，便于本地诊断发现缺口。

## 生命周期

Collector 状态为 `Created -> Running -> Draining -> Closed`。关闭时先停止新记录，再 flush critical/important，最后写入关闭摘要。强制终止只保证已落盘批次，不承诺内存队列内容。
