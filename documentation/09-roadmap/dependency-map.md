# 依赖图与工作包

## 架构依赖

```text
titan-types
  ├─> titan-kernel ─> titan-runtime ─> titan
  ├─> titan-model ──────────────────────┘
  ├─> titan-models ─> titan-model
  └─> titan-tools ─> runtime/telemetry/cluster client

titan-webui ─> HTTP/WebSocket API
```

底层 crate 不依赖 facade；工具和前端不进入模型执行依赖链。任何新 crate 必须标明所属层和允许依赖集合。

## 工作包依赖

| 工作包 | 依赖 | 解锁内容 |
| --- | --- | --- |
| 类型系统 | 无 | HAL、Tensor、序列化 |
| HAL | 类型系统 | CPU/设备执行和 stream/event |
| Tensor/Term | 类型系统、HAL | eager 和表达式构建 |
| Autograd | Term、VJP 注册 | 训练和梯度校验 |
| 训练组件 | Autograd、Parameter | optimizer、DataLoader、checkpoint |
| Graph IR | Term、shape/effect | pass 和执行计划 |
| Memory Planner | Graph IR、alias | 静态 buffer 复用 |
| Kernel ABI | 类型系统、执行计划 | 后端编译和 launch |
| 真实后端 | HAL、Kernel ABI | Native GPU 执行 |
| Autotune | kernel 变体、profiler | `.tune` 和选择策略 |
| 分层执行 | backend、model package、planner | 超显存模型和 MoE 调度 |
| Distributed | stream/event、ShardSpec | collective 和并行策略 |
| Recovery | artifact、distributed | 弹性恢复 |
| Telemetry | 稳定标识、runtime event | profiler 和诊断 |
| tt.exe | telemetry、cluster API | debug 和 cluster 工具 |
| Web UI | 聚合 API | 运行和集群界面 |
| 模型生态 | 模型接口、训练/推理/部署 | 七领域垂直闭环 |
| Production | 所有工作包 | 发布、兼容、安全和运维 |

## 并行原则

同一阶段内可以并行开发不共享协议的工作包，但协议的唯一所有者必须先提交 schema 和契约测试。例如 Web UI 可以在 API schema 固定后与服务端并行，不能在标识和分页协议未定时自行定义接口。
