# 控制面

本文保留控制面术语作为本地确定性契约的背景；当前 crate 未实现控制面、进程编排、成员租约或跨进程协调。

## 会话对象

每次启动生成不可复用的 `RunId`。会话包含以下不可变字段：

| 字段 | 语义 |
| --- | --- |
| `run_id` | 训练或推理会话的全局标识 |
| `world_size` | 预期参与进程数 |
| `min_world_size` | 弹性恢复允许的最小规模 |
| `graph_id` | 已编译图版本 |
| `model_id` | 模型结构和参数布局版本 |
| `topology_epoch` | 拓扑变更单调递增序号 |
| `lease_ttl_ms` | 成员心跳租约时长 |

## 本地边界

`LocalTransport` 只接受调用方给定的 `run_id` 和 epoch，并对每个本地 frame 验证这两个值。它不发现成员、分配 rank、建立连接或发布拓扑。

本地 API 不定义节点、地址、启动令牌或跨进程排序。

## 成员租约

本地实现没有心跳、租约、故障广播或重连。epoch 不匹配会立即作为本地帧错误返回。

## 拓扑快照

本地实现不维护 `TopologySnapshot` 或设备路径；调用方仅以 `run_id`、epoch 与 sequence 描述可重复的单进程顺序。

## 未实现的控制面消息

- `Register`：成员注册和能力声明。
- `AssignRank`：发布 rank/world 和 epoch。
- `TopologyPrepare`、`TopologyCommit`：两阶段拓扑切换。
- `StepAdvance`：提交全局 step 和图版本。
- `CheckpointCommit`：发布已提交 checkpoint。
- `FailureNotice`：传播成员、transport 或 collective 故障。
- `Shutdown`：有序结束会话。

上列消息类型是未实现的设计记录，不代表可用的跨进程协议。
