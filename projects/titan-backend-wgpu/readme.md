# titan-backend-wgpu

WebGPU backend for Titan's backend-neutral `DeviceSession` / HAL contract.

The crate exposes `WgpuDriver`, `WgpuCompiler`, and the same fixed launch ABIs used by
`titan-backend-cuda` for the initial vertical slice (`gemm.f32`, `elementwise.add.f32`).
