# 兼容性策略

## 版本层次

语义版本分别作用于公共 Rust API、artifact schema、遥测 schema、kernel ABI 和控制面协议。不同层次独立递增，不能用一个版本号掩盖不兼容变更。

## API 规则

同一主版本内保留公共 trait、错误类型和序列化字段的向后兼容。废弃 API 至少保留一个次版本周期，并在编译器警告和迁移文档中给出替代接口。内部模块不承诺稳定，但必须通过 workspace 编译。

## Artifact 与 schema

artifact manifest、`.tune` 自动调优缓存和遥测 record 都带 schema version。读取器支持当前版本和上一主版本的迁移；迁移只生成新副本，不覆盖原文件。无法验证的旧工件必须明确报错而不是静默忽略。

## 设备矩阵

后端能力以 capability table 声明：设备类型、compute capability、dtype、layout、atomic、通信 transport 和确定性支持。编译器在选择 kernel 前检查 capability，缺失能力返回可定位的 `UnsupportedCapability`。

## 二进制与工具

`tt.exe` 的子命令、选项、JSON 字段和退出码属于工具契约。新增字段向后兼容，删除选项需提供迁移周期。Web UI API 以 `schema_version` 和 cursor 保持兼容，前端不能假定服务端只返回固定字段。
