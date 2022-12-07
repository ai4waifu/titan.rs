# 编译缓存与执行计划

## CompileKey

CompileKey 包含：Term/Graph schema、graph hash、输入 symbolic shape substitution、dtype、layout、device capability fingerprint、backend、编译选项、precision policy、determinism policy、kernel version 和 runtime ABI version。

任何字段变化都必须产生新 key。缓存命中后仍检查 artifact checksum、签名、设备能力和版本兼容。

## ExecutablePlan

计划由以下数组组成：values、buffers、operators、kernels、streams、events、collectives、checkpoint points 和 telemetry points。引用使用整数 ID，避免运行时字符串查找。

执行计划必须能输出：

- 编译输入与最终约束。
- 每个 operator 的 kernel、tile、workspace 和 stream。
- buffer 分配、复用和释放点。
- event/collective 依赖。
- 预计和实际耗时。
- 失败时的 fallback 或回滚计划。

计划不包含任意代码指针或不可校验的外部路径。Native binary、PTX、WGSL 和计划均通过 artifact manifest 管理。
