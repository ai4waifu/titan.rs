---
name: titan-autotune-debug
description: Diagnoses Titan.rs MatMul autotuning and .tune cache behavior. Use when users mention autotune, tuning cache, tile selection, telemetry feedback, or autotune.tune.
disable-model-invocation: true
---

# Titan Autotune Debug

## Workflow

1. Run the example once to create or refresh `target/titan/autotune.tune`.
2. Read the cache. Each current row is `backend,m,n,k,tile`.
3. Run the example again to verify cached selection is reused for the same backend and shape.
4. Use `cargo run -p titan-tools -- debug` to check the artifact set.

## Interpretation

- The current search evaluates CPU tile candidates `8`, `16`, `32`, and `64`.
- A feedback promotion means a supplied production observation beat the incumbent duration.
- Cache text is an MVP format. Do not hand-edit it during a concurrent training run.

## Guardrails

- Never compare microsecond timings across different machines as a regression threshold.
- A `.tune` hit is keyed by backend and MatMul shape; it is not a universal kernel-performance guarantee.
