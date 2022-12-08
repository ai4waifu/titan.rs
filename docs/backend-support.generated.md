# Backend Support Matrix

> Generated-artifact contract: this document is intended to be regenerated from backend/schema registries, hardware CI and benchmark artifacts. It is not evidence of hardware support by itself.

| Backend | Current status | Evidence required before `verified` |
| --- | --- | --- |
| CPU | discoverable | AVX2 generated kernel execution, correctness, artifact load and event tests |
| CUDA | unsupported | compute_80+ Driver API discovery, PTX load/launch, correctness and hardware event suite |
| ROCm | unsupported | gfx1100 code-object load/launch, correctness and hardware event suite |

No row may be promoted based on an enum, crate presence, placeholder source, mock driver or host-mediated execution.
