# 总体架构

## 系统分层

```text
用户模型与 tensor! 表达式
        │
        ▼
titan facade / titan-macros / titan-model
        │
        ▼
Term / Tensor / Parameter / Autograd
        │
        ▼
Typed Graph / Optimizer / Memory Planner
        │
        ▼
Kernel Registry / Lowering / Autotune
        │
        ▼
HAL / CPU / CUDA / ROCm / Metal / WGPU
        │
        ├────────► Distributed Runtime
        └────────► Profiler / Telemetry / tt / Web UI
```

## Crate 职责

| Crate | 最终职责 |
|---|---|
| `titan` | 对外 facade，统一导出 Tensor、模型、训练、部署 API 和宏 |
| `titan-types` | shape、dtype、device、ID、版本、错误和跨层协议 |
| `titan-macros` | `neural`、`parameters`、`kernel`、`distributed` 过程宏 |
| `titan-hal` | 设备、storage、allocation、queue/stream、event 和同步抽象 |
| `titan-tensor` | Tensor、TensorView、Term、算子、广播、索引和自动微分 |
| `titan-model` | 模型、参数树、层、optimizer、DataLoader 和模型状态 |
| `titan-graph` | typed graph、优化 pass、shape/layout 推导和内存规划 |
| `titan-kernel` | 内核 DSL、lowering、binary cache、ABI 和 kernel 注册表 |
| `titan-autotune` | 搜索空间、基准、缓存、反馈和策略晋升 |
| `titan-runtime` | eager/JIT/AOT 执行、调度、future、stream 和错误边界 |
| `titan-distributed` | transport、collective、并行策略、checkpoint 和恢复 |
| `titan-profiler` | trace、metric、event、collector 和遥测协议 |
| `titan-tools` | `tt.exe` 调试与集群工具链，唯一允许使用 `clap` 的 crate |
| `titan-webui` | pnpm 管理的 Vue + Vite 运维界面 |
| `titan-models` | 具体模型实现、权重映射、配置和模型级测试 |

## 依赖方向

依赖只能从上层指向下层：

```text
titan
  -> model / runtime / distributed
  -> tensor / graph / autotune / profiler
  -> kernel / hal
  -> types
```

强制约束：

- `titan-types` 不依赖解析器、CLI、GPU SDK、网络和 Web UI。
- `titan-hal` 不依赖 Tensor、Graph、Model、Distributed 和 Tooling。
- `titan-kernel` 只通过 HAL 能力接口接触设备。
- `titan-model` 不依赖 `titan-webui` 或 `titan-tools`。
- `titan-tools` 可以组合诊断接口，但不得成为训练和推理的运行时依赖。
- `titan-webui` 只属于 pnpm workspace，不进入 Cargo workspace。

## 表达式与自动微分

`tensor!` 将数学语法构造为不可变 `Term`。每个 `Term` 保存操作、输入、dtype、device、rank、shape 约束、布局约束、source span 和梯度需求。

```text
tensor! syntax
  -> Term DAG
  -> constraint solving
  -> typed forward graph
  -> VJP expansion
  -> typed backward graph
```

Tensor 是已物化数据，Term 是延迟计算。TensorView 借用 Tensor storage，不拥有设备内存。Parameter 是具有稳定 ID、训练状态和 checkpoint 映射的 Tensor 包装。

自动微分以通用 reverse-mode tape/backward graph 为基础。过程宏只生成结构遍历和静态元数据，不能替代 VJP 系统。

## 图编译器

图节点与值均使用稳定 ID。每个 value 具有 dtype、shape、device、layout、alias set 和 lifetime。编译流水线按以下顺序执行：

1. 约束求解与类型检查。
2. 常量折叠和无效节点消除。
3. layout 选择、broadcast materialization 和 view 合法性分析。
4. forward/backward 融合与设备分区。
5. buffer liveness、内存复用和 workspace 规划。
6. kernel 候选生成与 autotune 查询。
7. stream/queue 调度、event 依赖和 command buffer 生成。

每个 pass 产生可序列化报告，包含输入图版本、输出图版本、变换节点、source span、合法性依据和性能估计。

## 内核与后端

Kernel ABI 描述参数、shape、stride、layout、alignment、workspace、launch config 和同步语义。生成内核与手工内核使用同一 ABI。

设备 capability fingerprint 至少包含：设备型号、架构版本、驱动版本、warp/wave 大小、共享内存、寄存器限制、向量宽度、矩阵指令能力和队列能力。该指纹参与 kernel binary 与 `.tune` cache key。

HAL 负责资源生命周期，不负责图优化。Runtime 负责执行计划，不直接操作厂商 API。Kernel 层负责 lowering 和 launch 描述，不负责模型语义。

## 分布式架构

分布式系统分为控制面和数据面：

- 控制面：rendezvous、membership、rank、world、拓扑、健康检查、弹性变更和恢复协调。
- 数据面：transport、collective、point-to-point、bucket、stream overlap 和错误传播。

每个 collective 具有 sequence ID、参与者集合、dtype、元素数、timeout 和 trace context。梯度 bucket 在 backward graph 中显式表示，使通信可以与剩余反向计算重叠。

Checkpoint 由 manifest、模型 tensor shard、optimizer shard、RNG state、DataLoader state、拓扑和版本信息组成。提交过程使用临时版本、校验和和原子 manifest 切换。

## 可观测性架构

所有层通过结构化事件写入无阻塞 telemetry channel。collector 负责批处理、采样、背压、脱敏、持久化和远程发送。训练关键路径在 collector 不可用时仍可继续运行。

统一关联键包括 `run_id`、`model_id`、`graph_id`、`operator_id`、`kernel_id`、`rank`、`step` 和 `checkpoint_id`。Web UI 和 `tt.exe` 只读取版本化诊断 API，不直接解析内部内存或未提交文件。
