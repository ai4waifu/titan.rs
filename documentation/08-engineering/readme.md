# 工程治理

本目录是贡献者和发布维护者的执行规范。它定义 workspace 边界、测试矩阵、安全约束、兼容性策略、性能门槛和发布流程。

## 文档索引

- [开发环境](development.md)：nightly Rust、Cargo workspace、pnpm workspace 和依赖规则。
- [测试策略](testing.md)：单元、集成、数值、属性、模糊、端到端和基准测试。
- [安全规范](security.md)：unsafe、设备代码、集群端点、checkpoint 和遥测安全。
- [兼容性](compatibility.md)：API、二进制、artifact、schema 和设备支持矩阵。
- [性能工程](performance.md)：基准、回归阈值、内存和分布式指标。
- [发布流程](release.md)：版本、可复现构建、soak、签名、回滚和公告。

## 强制边界

1. `titan-types` 是整个生态的共享底层库，不包含网络、文件系统、进程执行和命令行依赖。
2. `clap` 只允许出现在 `titan-tools`；其他 crate 不得通过传递依赖获得命令行解析能力。
3. Web UI 只由 pnpm workspace 管理，包管理器固定为 pnpm，不使用 npm lockfile。
4. Rust workspace 使用 nightly，并在根目录锁定 toolchain、组件和 target。
5. `titan-webui` 使用 Vue + Vite；Rust/WASM 只提供轻量浏览器能力，不承担大模型部署。
