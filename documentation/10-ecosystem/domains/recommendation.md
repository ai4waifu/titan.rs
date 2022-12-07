# 推荐系统领域

## 范围

覆盖稀疏特征、大规模 Embedding、双塔召回、排序、CTR、向量检索、增量更新和低延迟服务。

## 数据与一致性

Feature schema 固定字段 id、类型、缺失策略、词表/哈希版本、归一化和时间有效性。训练样本、离线评估和线上请求携带 schema version，禁止按字段位置隐式匹配。

## 基础能力

- sparse/ragged tensor、EmbeddingBag、segment reduce 和稀疏 optimizer。
- 分片 Embedding、冷热缓存、一致性版本和增量 checkpoint。
- 双塔向量导出、ANN 检索接口、负样本和排序特征融合。
- 离线训练与在线模型服务的原子版本切换。

## 在线更新

更新流按 model/feature version 排序，先写 staging shard，经 checksum 和一致性验证后提交。服务请求固定读取一个模型 epoch，不能在一次请求中混用新旧 Embedding。

## 验收

测试覆盖 sparse index 边界、重复 id、分片覆盖、增量恢复、离线线上特征一致和原子切换。基准报告训练吞吐、lookup p95/p99、cache hit、更新延迟、内存/存储占用和服务失败率。
