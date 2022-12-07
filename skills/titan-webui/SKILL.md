---
name: titan-webui
description: Builds, runs, and verifies the Titan.rs Vue and Vite operations dashboard with pnpm. Use when users mention titan-webui, dashboard, Vue, Vite, training UI, telemetry UI, or Web UI.
disable-model-invocation: true
---

# Titan Web UI

## Workflow

1. Use pnpm only:

   ```powershell
   pnpm install
   pnpm --dir projects/titan-webui build
   pnpm --dir projects/titan-webui dev --host 127.0.0.1
   ```

2. Open the Vite URL and check the dashboard shows training loss, graph fusion, autotune feedback, and four runtime artifacts.

3. Keep Cargo and pnpm boundaries intact. `projects/titan-webui` is excluded from Cargo workspace membership.

## Guardrails

- Do not use npm to install or update frontend dependencies.
- The current UI renders MVP state. Do not present it as a live production telemetry system without a versioned authenticated API.
- Native CUDA/ROCm deployment is the target for large models. Keep WebAssembly limited to lightweight demos and small inference.
