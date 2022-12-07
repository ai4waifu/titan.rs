---
name: titan-cluster-debug
description: Validates Titan.rs cluster topology parameters and inspects distributed training artifacts through tt.exe. Use when users mention tt cluster, rank, nodes, endpoint, all-reduce, checkpoint, ZeRO, FSDP, or distributed debugging.
disable-model-invocation: true
---

# Titan Cluster Debug

## Workflow

1. Validate topology:

   ```powershell
   cargo run -p titan-tools -- cluster --nodes 2 --rank 1 --endpoint tcp://127.0.0.1:29500
   ```

2. Inspect artifacts:

   ```powershell
   cargo run -p titan-tools -- debug
   ```

3. For the CPU smoke test, verify `all_reduce=[4.0, 6.0]` and inspect `target/titan/checkpoint.titan`.

## Current Scope

`LocalRing` is a local semantic implementation. `tt cluster` validates topology only; it does not start processes, establish a TCP rendezvous, invoke NCCL/RCCL, or provide fault tolerance.

## Guardrails

- Reject `rank >= nodes` before attempting any launch workflow.
- Treat checkpoint text as development state until it has a versioned, atomic production protocol.
