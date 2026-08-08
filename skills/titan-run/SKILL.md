---
name: titan-run
description: Runs the Titan.rs CPU end-to-end training smoke test and verifies generated tuning, checkpoint, deployment, and interchange artifacts. Use when a user asks to run, validate, or demonstrate the Titan training workflow.
disable-model-invocation: true
---

# Titan Run

## Workflow

1. From the repository root, run:

   ```powershell
   cargo run -p titan-example
   cargo run -p titan-tools -- debug
   ```

2. Confirm the training loss decreases and `tt debug` reports these as `ready`:

   - `autotune.tune`
   - `checkpoint.titan`
   - `deployment.manifest`
   - `model.onnx.txt`

3. Report the output paths under `target/titan` and distinguish MVP artifact envelopes from production formats.

## Guardrails

- Do not claim PTX IR means a GPU kernel was executed; the current example runs on CPU.
- Do not use `tt` as a requirement for importing Titan Rust crates.
