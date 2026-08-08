# 模型包与块索引

模型包必须允许部分读取，不能要求先把全部权重物化到内存。根 manifest 描述模型、张量、量化、分片和完整性信息。

## Manifest 字段

- schema、producer、model id、模型配置和 tokenizer/config 引用；
- 每个 tensor 的 ParameterId、名称路径、shape、dtype、layout、语义角色；
- shard、文件、offset、length、alignment、checksum 和压缩方式；
- 专家 id、layer id、线性层 id、block/tile 索引；
- 量化格式、scale、zero point、group size、校准版本和质量报告摘要；
- 可执行 backend、kernel ABI、最低 capability 和 fallback 格式。

## 索引粒度

索引支持 `Tensor -> Layer -> Expert -> Projection -> Block/Tile` 多级定位。模型准备工具根据设备 I/O 粒度和 kernel layout 生成块；运行时只消费索引，不在请求关键路径重新切分大文件。

## 读取后端

本地后端支持普通异步文件读取、memory map 和平台特定异步 I/O。Direct I/O 只在对齐、buffer 和基准均满足时启用，不作为默认路径。L4 后端先下载到有 checksum 的本地 staging，再原子发布为 L3 block。

## 完整性

读取前校验范围和对齐，读取后校验长度与 checksum，解压/解量化后校验逻辑 tensor 摘要。损坏 block 只允许从已验证副本恢复；不存在副本时中止相关请求并标记模型版本不可用。
