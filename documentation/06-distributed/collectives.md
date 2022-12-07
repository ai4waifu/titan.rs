# Collective 原语

collective 是图中的显式通信节点。编译器为每个节点分配单调的 `collective_seq`，运行时以序号检查所有 rank 的调用顺序。

## 原语语义

### Ring AllReduce

输入为每个 rank 的同形状 tensor，输出为所有输入的规约结果。执行分为 ReduceScatter 和 AllGather 两阶段，每阶段按 ring 邻居传递固定大小 chunk。默认规约为求和，accumulate dtype 可高于输入 dtype。

### ReduceScatter

先对所有 rank 的输入按元素规约，再将结果切分给各 rank。输出 shard 的布局由 `ShardSpec` 指定，未对齐的最后一个 shard 必须进行显式 padding 并在 metadata 中记录有效长度。

### AllGather

每个 rank 提供一个 shard，所有 rank 收集完整 tensor。结果顺序由 `ShardSpec` 的全局索引确定，不依赖 rank 的物理位置。

## Collective 序列

每个 collective 节点包含操作类型、输入输出 tensor id、dtype、shape、规约算子、`collective_seq` 和超时预算。执行前交换序列摘要；摘要不一致时在设备写入前终止，以避免部分 rank 修改状态。

## 通信与计算重叠

编译器把 gradient bucket 切成可独立规约的 chunk，在反向图产生 chunk 后立即提交 ReduceScatter。计算 stream 和通信 stream 通过 event 建立依赖，禁止 busy-wait。planner 必须同时保留尚未通信的梯度和正在发送的 shard。

## 数值规则

规约顺序在同一拓扑 epoch 内固定，确定性模式使用固定 ring 顺序和高精度 accumulate。非确定性模式可以使用树形算法，但执行计划和遥测必须标记该选择。
