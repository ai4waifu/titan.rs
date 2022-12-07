# 可观测性与工具链

本目录定义 Titan.rs 从运行时事件到开发者界面的完整观测链路。数据先在进程内标准化，再由采集器批处理写入本地或远端；`tt.exe` 面向 debug 和集群运维，`titan-webui` 提供 Vue + Vite 的浏览和交互界面。

## 文档索引

- [遥测数据模型](telemetry-schema.md)：标识、事件、trace、metric、采样和脱敏。
- [性能分析器](profiler.md)：CPU/GPU、内存、kernel、图 pass 和通信分析。
- [采集器](collector.md)：批处理、背压、离线缓存、远端发送和生命周期。
- [tt 工具链](tt-toolchain.md)：`tt.exe debug` 与 `tt.exe cluster` 的输入输出契约。
- [Web UI](webui.md)：Vue/Vite 前端边界、API、权限和大模型部署边界。
- [诊断规则](diagnostics.md)：错误分级、规则引擎、报告和退出码。

## 数据流

```text
运行时/编译器/后端
        -> 进程内事件缓冲
        -> Collector（批处理、采样、脱敏）
        -> 本地 .titan/telemetry 或远端 Collector
        -> tt.exe / titan-webui
```

遥测不能改变训练语义。采集器发生拥塞时优先丢弃低优先级采样，不得阻塞 device stream 或 collective。
