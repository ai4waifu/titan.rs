# Tensor、Storage 与 View

## 对象边界

`Storage` 拥有设备字节区域和释放责任；`Tensor` 在 Storage 上叠加 dtype、shape、stride、layout、offset、device 和逻辑 id；`TensorView` 是受生命周期与 alias 约束的投影，不拥有底层内存。

## Storage 状态

Storage 状态为 `Allocated -> Ready -> InUse -> Retired -> Released`。异步命令提交后，stream event 持有资源 lease；Rust 对象离开词法作用域只会进入 `Retired`，设备确认完成后才能真正释放。

每个 storage 记录字节数、对齐、设备、allocator pool、可写性、版本、活动 reader/writer、最后写 event 和分布式引用。

## View 规则

reshape 只有在 stride 可表达时返回 view，否则显式 materialize。slice、transpose、select、broadcast 和 as_strided 必须验证 offset 与可达范围。零 stride broadcast view 不允许原地写。

## 原地操作

in-place 写入需要同时证明：没有活动读取者；输出 layout 与 kernel 兼容；Autograd 未保存旧值；不存在重叠 view；通信 bucket 和 checkpoint writer 未引用；写入 event 能成为后续消费者依赖。

证明失败时返回结构化 alias 错误或生成 out-of-place 节点，不允许静默覆盖。

## 跨设备迁移

`to_device` 是显式操作并生成新的 Storage。传输包含 source/destination lease、transfer stream、完成 event 和错误；跨 NUMA、host staging、P2P 和 dtype/layout 转换均进入执行计划与 trace。

## 测试不变量

覆盖越界 view、空 tensor、非连续 layout、重叠 alias、多线程 clone、异步 drop、设备 copy 和 allocator 重用。Miri/解释器可覆盖 host unsafe，GPU 生命周期由事件注入和故障测试覆盖。
