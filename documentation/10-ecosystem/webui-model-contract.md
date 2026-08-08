# 模型与 Web UI 契约

`titan-webui` 是 pnpm workspace 中的 Vue + Vite 应用。它读取 Native 运行时、artifact store 与控制面提供的 API，不直接持有训练 Tensor、模型权重或设备句柄。前沿大模型在 Native CUDA/ROCm 进程中加载和执行；WASM 仅可用于配置校验、tensor 检查和明确标记为轻量演示的模型。

## 资源模型

Web UI 中的模型列表来自 `GET /api/models`。每个条目至少包含 `family`、`variant`、`schema_version`、capabilities、manifest 状态、可用 targets 和最近诊断摘要。该端点只返回模型描述与审计信息，不返回权重字节、密钥、原始样本或可识别遥测字段。

模型详情由 `GET /api/models/{family}/{variant}` 返回，包含 input/output schema、预处理版本、权重 manifest 摘要、兼容 runtime/后端、资源预算结论和已知回退策略。UI 使用 schema version 渲染字段；未知字段原样保留在诊断视图，不能猜测语义。

### 只读 JSON v1

`titan-model::ApiJsonSchema::documents()` 是 Draft 2020-12 的可导出 schema 文档。服务端在该 schema 的范围内实现如下 wire format；版本变化必须增加新的 schema version，不能改写 v1 字段语义。

```json
{
  "schema_version": 1,
  "request_id": "req_01H...",
  "generated_at": "2026-08-08T12:00:00Z",
  "data": []
}
```

`GET /api/models` 的 `data` 为 `ModelCatalogEntry[]`。每项包含 `family`、`variant`、`schema` (`input`、`output`、`version`)、五个布尔 `capabilities`、`manifest` (`schema_version`、`state`、`deployment_targets`) 与脱敏的 `diagnostic_summary`。`deployment_targets` 为 `native` 或 `wasm`；`wasm` 只表示轻量验证/演示兼容性，不表示训练或生产大模型推理能力。

`GET /api/runs/{id}` 的 `data` 为 `RunStatus`：`run_id`、完整的只读 `model` 条目、可空的 `step`/`rank`/`graph_version`、`health` (`healthy`、`degraded`、`failed` 或 `unknown`) 以及 `health_summary`。这些字段描述已发生的运行状态，端点不接受控制、训练或调度参数。

失败响应不使用成功 envelope，而返回 `{ "schema_version", "request_id", "code", "message", "retryable" }`。v1 code 仅为 `schema_unsupported`、`model_not_found`、`run_not_found`、`manifest_invalid`、`invalid_request`、`internal`。`message` 不得带路径、权重、输入样本或密钥；客户端可以显示 `request_id` 供关联诊断。

## 运行与诊断

`GET /api/runs/{run_id}` 返回模型标识、step、rank、图版本、设备状态和健康摘要。`GET /api/runs/{run_id}/budget` 返回分层存储预算、KV cache、专家缓存、L1/L2/L3 命中、预计和实测延迟；只有运行时生成且经权限过滤的数据才能显示。

`WS /api/events` 推送已聚合的 run、kernel、collective、checkpoint 与诊断事件。事件必须包含 `run_id`、时间、严重级别和 schema version；客户端按 event id 去重，重连后从最后确认 id 续传。服务端背压或权限变化时发送明确状态事件，UI 不以轮询掩盖数据缺口。

## 操作边界

UI 可以创建受控任务请求、取消自身有权访问的任务、触发只读诊断和下载报告；高风险的集群修复、权重迁移、缓存清理和密钥操作由 `tt.exe` 或服务端受审计 API 执行。所有写操作使用请求 id 与幂等键，响应返回状态、审计 id 与可重试语义。

前端构建只能使用 pnpm workspace。浏览器页面不得成为 Rust 库的依赖，也不得将 Node 运行时打包进 Native 模型部署物。
