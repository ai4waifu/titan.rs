# 图学习领域

## 范围

覆盖同构/异构图、CSR/CSC/边列表、邻居采样、消息传递、GCN、GAT、GraphSAGE、GIN、MPNN、分子图、知识图谱和大图分区。

## 数据模型

Graph schema 固定 node/edge type、id space、feature schema、direction、multi-edge/self-loop 和 partition version。CSR/CSC 的 offsets、indices 和 feature shard 必须经过范围、排序和 checksum 验证。

## 基础能力

- sparse layout、segment/scatter/gather、sampled subgraph 和 ragged batch。
- 可组合 message、aggregate、update 和 attention。
- neighbor/fanout/random-walk/negative sampling，显式 RNG 与 epoch。
- 图分区、远程 feature fetch、Embedding 和 sampler checkpoint。

## 分布式执行

采样和计算拓扑分离描述。Planner 根据 partition、feature locality、网络和 GPU 内存放置 sampler、cache 和 model shard；跨 rank 请求带 graph version，避免读取不一致更新。

## 验收

测试覆盖 CSR/CSC 转换、孤立节点、重复边、采样分布、消息传递已知答案、分区覆盖和恢复。基准报告 edges/s、sampling p95、feature cache hit、通信字节、显存和扩展效率。
