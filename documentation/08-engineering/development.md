# 开发环境

## Rust 工具链

根目录 `rust-toolchain.toml` 固定 nightly channel、`rustfmt` 和 `clippy` 组件。提交前必须在干净依赖缓存和指定 target 上执行格式化、检查和测试。需要 nightly feature 的模块在 crate 根部显式声明，并在 API 文档中说明原因。

## Cargo workspace

workspace 成员按职责分层：

- `titan-types`：共享标识、shape、dtype、device、layout、错误和序列化类型。
- `titan`：面向 Rust 用户的公共 facade。
- `titan-model`：抽象模型接口和模型生命周期。
- `titan-models`：具体模型集合，按模型独立模块维护。
- `titan-kernel`：kernel 描述、ABI、缓存和后端适配。
- `titan-runtime`：执行计划、调度、资源和工件。
- `titan-tools`：`tt.exe` 命令实现，唯一允许 `clap`。

依赖方向只能从上层指向下层，底层不得反向依赖 facade。公共类型优先放入 `titan-types`，避免 crate 间复制定义。

## pnpm workspace

`titan-webui` 与前端包通过根 `pnpm-workspace.yaml` 管理。提交必须包含 `pnpm-lock.yaml`，使用 `pnpm install --frozen-lockfile` 验证，生产构建使用 `pnpm -r build`。Rust 构建和前端构建分别产出工件，发布脚本在集成阶段组合，不把 Node 运行时打进 Rust 库。

## 依赖审查

新增依赖必须记录用途、许可证、最小版本、是否引入 native code、构建时间和安全公告。依赖不得绕过 crate 边界；编译期宏、运行时、工具链和 Web UI 的依赖集合分别锁定并定期审计。

## 提交检查

每个变更至少通过 `cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets --all-features -- -D warnings`、`cargo test --workspace`、`cargo cry` 和受影响的 `pnpm` 检查。提交信息应包含影响 crate、协议变更和验证命令。
