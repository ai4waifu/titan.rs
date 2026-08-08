# 发布流程

## 发布输入

发布候选必须包含源代码、Cargo.lock、pnpm-lock.yaml、nightly toolchain、crate 和 Web UI 版本、artifact/schema 变更清单、许可证清单、基准报告和安全审计结果。

## 可复现构建

构建环境固定 Rust nightly、Node/pnpm、target、编译器 flags 和 native SDK 版本。产物记录源码 revision、依赖 checksum、编译环境摘要和 SBOM；重复构建应产生相同或可解释差异的 checksum。

## 验收顺序

1. 格式、lint、单元和集成测试。
2. 数值回归、属性/fuzz 样本和端到端工作流。
3. CPU、GPU、TCP 和可用高速 transport 基准。
4. 多小时 soak test，覆盖 checkpoint、telemetry 背压和故障恢复。
5. `tt cluster validate` 与 Web UI 生产构建。
6. 产物签名、清单生成、发布说明和升级/回滚演练。

## 回滚

发布包保留上一稳定版本、artifact schema 迁移器和 kernel cache 兼容信息。发现数据一致性、安全或严重性能问题时，先停止新任务，再切回上一版本；已有 checkpoint 必须在回滚版本上完成读取验证。回滚动作写入审计记录并发布影响范围。
