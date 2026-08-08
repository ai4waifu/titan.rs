# 阶段 07：可观测性、tt.exe 与 Web UI

## 前置条件

- runtime、compiler、kernel、distributed 和 checkpoint 具有稳定标识及事件点。
- 聚合服务可提供只读 HTTP/WebSocket API。
- pnpm workspace 和 Vue/Vite 构建链固定。

## 代码交付物

- trace/event/metric schema、采样、批处理、背压、脱敏和本地离线缓存。
- CPU/GPU、allocator、kernel、graph pass、DataLoader 和 collective profiler。
- 远端 Collector、批次鉴权、重试和 schema 迁移。
- `titan-tools` 与 `tt.exe debug/cluster`，稳定 JSON 和退出码。
- `titan-webui` 的运行、时间线、图、集群、checkpoint 和诊断页面。
- WebSocket 增量更新、断线重连、分页和权限处理。

## API 交付物

- 统一 `TelemetryRecord` 和 context 标识。
- run、timeline、metrics、diagnostics、checkpoint、topology 的版本化 Web API。
- `tt.exe` 的命令、参数、issue code 和输出 schema。

## 测试交付物

- schema 编解码、未知字段、大小限制、脱敏和采样确定性测试。
- collector 队列满、磁盘配额、断网、重试和异常退出恢复测试。
- profiler 时间对齐、异步 kernel 完成和开销预算测试。
- `tt.exe` golden 输出和退出码测试。
- Web UI 单元、API contract、WebSocket、桌面/移动布局和生产构建测试。

## 性能交付物

- 默认遥测 CPU 开销、记录丢弃率和磁盘/网络带宽。
- 大 run 的 timeline 查询、分页和 Web UI 首屏/交互延迟。
- 采集关闭、默认和详细三档对训练吞吐的影响。

## 文档交付物

- 完成 schema、profiler、collector、tt、Web UI 和 diagnostics 文档。
- 每个 issue code 写明检测条件、证据和处理建议。

## 失败条件

- 遥测背压可以阻塞训练或 collective。
- 记录包含参数原值、输入、token 或密钥。
- `tt.exe` 承担模型训练或成为 Rust 用户必需依赖。
- 浏览器被要求部署 Native 大模型执行路径。

## 完成验收

端到端训练的编译、执行、通信、checkpoint 和故障可在 `tt debug` 与 Web UI 中用同一标识串联；关闭远端网络后本地缓存仍可完整诊断关键故障。

## 解锁条件

运维数据、诊断接口和性能指标稳定，允许生产发布流程据此设置门槛和告警。
