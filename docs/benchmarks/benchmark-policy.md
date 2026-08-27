# Benchmark Policy

Titan 只有在固定 workload、模型、shape、batch、context、dtype/quantization、硬件、driver、比较对象版本和 commit 都公开时，才可发布性能结论。

报告必须同时包含 startup、compile/tune cost、cache-hit latency、throughput、kernel median/p95、端到端 p50/p95/p99、peak RAM/VRAM/disk、workspace 和 correctness tolerance。

没有对应 benchmark artifact 时，不得声称 Titan 超越 PyTorch、vLLM 或 SGLang。
