# titan-webui

`titan-webui` 是由 pnpm workspace 管理的 Vue + Vite 前端。它显示运行、图、kernel、分布式拓扑、checkpoint 和诊断结果；浏览器不承担大模型训练或主要推理。

## 前端边界

Vue 应用只通过 HTTP/WebSocket API 读取聚合后的遥测和控制面只读视图。大模型 Native CUDA/ROCm 进程运行在服务器或集群，Web UI 只提交任务、查看状态和下载报告。WASM 仅承载轻量 tensor 检查、配置校验和演示级推理。

## API 契约

- `GET /api/runs`：分页列出 run 摘要。
- `GET /api/runs/{run_id}`：返回模型、图、step、rank 和健康状态。
- `GET /api/runs/{run_id}/timeline`：按时间范围和 rank 查询 trace。
- `GET /api/runs/{run_id}/metrics`：查询指标聚合。
- `GET /api/runs/{run_id}/diagnostics`：返回 issue code、级别、证据和建议。
- `GET /api/checkpoints`：列出已提交 checkpoint。
- `GET /api/cluster/topology`：返回脱敏拓扑和能力。
- `WS /api/stream`：推送状态变化、故障和 checkpoint 提交事件。

所有响应带 `schema_version`、请求 id 和生成时间；分页使用 opaque cursor，时间线查询必须限制范围和最大记录数。

模型目录与运行详情使用 `titan-model` 的只读 schema v1：`GET /api/models` 和 `GET /api/runs/{run_id}` 成功时均为 `{schema_version, request_id, generated_at, data}`。前者的 data 是包含 manifest schema version、deployment targets 和 manifest validation state 的目录数组；后者包含 `RunHealth` 与脱敏 `health_summary`。错误为结构化 `{schema_version, request_id, code, message, retryable}`，而不是 HTML 或未分类字符串。完整字段和 JSON Schema 见[模型与 Web UI 契约](../10-ecosystem/webui-model-contract.md)。

Web UI 对 v1 响应做运行时 shape 检查：网络失败显示 offline，非 2xx 显示 HTTP 和 request id，无法验证的 JSON 显示 protocol，取消的 fetch 显示 cancelled。客户端不会以缓存或猜测字段伪造当前模型目录或 run health。

## 页面职责

运行总览显示吞吐、延迟、显存、loss、step 和告警；时间线页面按 rank/stream 展示 trace；图页面显示 operator、fusion、内存和 kernel 选择；集群页面显示拓扑、租约和通信；checkpoint 页面显示 manifest、校验和恢复状态；诊断页面按 issue code 聚合证据。

## 权限与数据

默认只读。修改运行配置、停止任务和触发恢复必须由服务端鉴权并记录审计事件。前端不接触密钥、参数原值、输入文本和未经脱敏的路径。
