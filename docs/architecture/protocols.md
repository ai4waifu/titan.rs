# 稳定协议

`titan-types` 是跨 crate 的唯一稳定协议来源。`BackendId`、`DeviceId`、`DeviceFingerprint`、`DType`、`Shape`、`Strides`、`Layout`、`OperatorId`、`CandidateId`、`KernelId`、`AbiHash`、`ArtifactKey`，以及 precision、determinism、workspace、fallback、effect、alias contract 均不得在其他 crate 重定义。

所有 identity 必须由 canonical serialization 计算。字符串拼接、后端裸指针、host address 和未排序 attrs 不得参与稳定 identity。

Kernel ABI 的 buffer 参数使用 opaque slot。slot 只能由 Runtime/DeviceSession 的 launch context 绑定，不能编码 CUDA pointer、CPU address 或 ROCm native handle。
