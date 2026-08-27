# Backend Contract

backend 状态只允许为：

- `verified`：真实硬件测试通过，包括 generated kernel、artifact load、异步 launch、event timing 和 correctness。
- `discoverable`：可发现驱动或设备，但未完成 kernel launch 验收。
- `skipped_no_driver`：硬件 suite 因本机没有可用驱动而跳过。
- `unsupported`：该 target 不在支持范围。

当前 CPU 只验证了资源 contract；AVX2 generated codegen 尚未验收。CUDA 与 ROCm 当前不得声明执行支持。

HAL 只能包含 enumerate/open、allocation、copy、stream/event、artifact load、launch、poll 和 wait。禁止算子方法、厂商裸句柄、PTX/HIP 算子常量、host-mediated fake execution 和全设备 synchronize。

CUDA 的目标实现只允许动态 CUDA Driver API，不允许 CUDA Toolkit、NVRTC 或 CUDART。ROCm 的目标实现只允许动态 HSA/HIP Driver API，不允许 hipRTC、COMGR、LLVM 或 clang。
