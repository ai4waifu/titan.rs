# titan-backend-wgpu

Titan backend-neutral `DeviceSession` / HAL 合同的 WebGPU backend。

本 crate 暴露 `WgpuDriver`、`WgpuCompiler`，并在首个垂直切片（`gemm.f32`、`elementwise.add.f32`）中使用与 `titan-backend-cuda` 相同的固定 launch ABI。
