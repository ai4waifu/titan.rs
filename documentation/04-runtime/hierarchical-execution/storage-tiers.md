# 存储层级与内存区域

## 逻辑层级

- L0：GPU 寄存器、共享内存和单次 kernel workspace，生命周期不超过 launch 或已声明的 pipeline。
- L1：GPU/加速器显存，承载常驻权重、专家缓存、KV Cache、激活、工作区和传输缓冲。
- L2：主机内存，包含普通页、受限的页锁定缓冲、内存映射窗口和格式转换空间。
- L3：本地持久化存储，支持按张量、专家、线性层、块或 tile 读取。
- L4：远程对象存储或网络文件系统，只用于显式启用的冷启动、恢复和预热。

每个 `StorageLocation` 包含层级、设备/NUMA 节点、可访问后端、对齐、容量、带宽、延迟等级、可迁移性、可写性和一致性版本。

## L1 显存预算

显存按执行计划分配，不使用固定百分比。预算至少拆为：

| 区域 | 内容 | 约束 |
| --- | --- | --- |
| 常驻权重 | embedding、attention、normalization、router 等高复用权重 | 默认不可驱逐，变更必须重新规划 |
| 专家缓存 | 当前或预测会访问的专家 | 可收缩，有严格字节上限 |
| KV Cache | 每会话的 key/value page | 按上下文和配额增长，可回收 |
| 激活与 workspace | 中间值、reduction、转置、通信 | 由 liveness 和 kernel ABI 决定 |
| 传输缓冲 | host/device 和 device/device staging | 双/多缓冲受总预算约束 |
| 安全余量 | 驱动、第三方库和不可预见分配 | 不能被业务区域占用 |

KV Cache 增长只能在会话配额内压缩专家缓存，不得覆盖常驻权重、存活激活或未完成传输。

## L2 主机内存

普通页内存用于冷权重和元数据，页锁定内存只用于确有异步 DMA 收益的 buffer。页锁定预算同时受进程上限和整机上限约束，分配失败时回退到普通页和同步/分块 copy，并记录性能降级。

NUMA 感知分配应优先靠近目标 GPU。跨 NUMA 迁移必须计入成本模型和 trace。

## 分配与碎片

临时对象按尺寸类别复用；KV Cache 使用固定页/块；专家缓存使用固定槽位或可合并大块；后端具备虚拟内存映射时可以重映射物理页。任何 compact 都不得移动仍被 stream/event 引用的区域。

allocator 记录 allocated、reserved、largest free block、fragmentation ratio、failed allocations 和 eviction latency。碎片超过阈值时先降低并发和缓存预算，再安排安全整理，禁止触发未声明的设备级同步。
