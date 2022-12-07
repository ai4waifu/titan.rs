#![warn(missing_docs)]
//! Coordinates tuning, execution and telemetry at the operator boundary.

use std::{path::Path, time::Duration};
use titan_autotune::{Autotuner, MatmulKey, measure};
use titan_hal::Backend;
use titan_profiler::Profiler;
use titan_tensor::{Tensor, TensorError};

/// Hard resource limits used before admitting a model or request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceBudget {
    /// Maximum device memory in bytes available to Titan allocations.
    pub device_bytes: u64,
    /// Maximum host staging memory in bytes.
    pub host_bytes: u64,
    /// Maximum concurrent requests.
    pub concurrency: u32,
}

/// Input to the deterministic resource planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceRequest {
    /// Resident device bytes required by the plan.
    pub device_bytes: u64,
    /// Host staging bytes required by the plan.
    pub host_bytes: u64,
    /// Number of requests to admit.
    pub concurrency: u32,
}

/// Result of checking a request against a hard budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudgetReport {
    /// Whether all hard limits are satisfied.
    pub feasible: bool,
    /// Device bytes still available (zero when over budget).
    pub device_available: u64,
    /// Host bytes still available (zero when over budget).
    pub host_available: u64,
    /// Concurrency slots still available (zero when over budget).
    pub concurrency_available: u32,
}

impl ResourceBudget {
    /// Checks a request without allocating or relying on OOM behavior.
    pub fn assess(self, request: ResourceRequest) -> ResourceBudgetReport {
        let device_available = self.device_bytes.saturating_sub(request.device_bytes);
        let host_available = self.host_bytes.saturating_sub(request.host_bytes);
        let concurrency_available = self.concurrency.saturating_sub(request.concurrency);
        ResourceBudgetReport {
            feasible: request.device_bytes <= self.device_bytes
                && request.host_bytes <= self.host_bytes
                && request.concurrency <= self.concurrency,
            device_available,
            host_available,
            concurrency_available,
        }
    }
}

#[derive(Debug)]
pub struct Runtime {
    tuner: Autotuner,
    profiler: Profiler,
    last_matmul: Option<MatmulKey>,
}

impl Runtime {
    pub fn open(cache_path: impl AsRef<Path>) -> Self {
        Self { tuner: Autotuner::open(cache_path.as_ref()), profiler: Profiler::default(), last_matmul: None }
    }
    pub fn matmul<B: Backend>(
        &mut self,
        left: &Tensor<B, f32, 2>,
        right: &Tensor<B, f32, 2>,
        backend: B,
    ) -> Result<Tensor<B, f32, 2>, TensorError> {
        let [m, k] = left.shape();
        let [_, n] = right.shape();
        let key = MatmulKey { backend: B::NAME.into(), m, n, k };
        let tile = self.tuner.choose(key.clone(), |tile| {
            measure(|| {
                let _ = left.matmul(right, backend.clone(), tile);
            })
        });
        self.last_matmul = Some(key);
        self.profiler.measure(format!("matmul/{}/tile-{tile}", B::NAME), || left.matmul(right, backend, tile))
    }
    pub fn telemetry(&self) -> &Profiler {
        &self.profiler
    }

    /// Applies a production timing observation to the most recently tuned MatMul configuration.
    pub fn record_autotune_feedback(&mut self, tile: usize, observed: Duration, incumbent: Duration) -> bool {
        self.last_matmul.clone().is_some_and(|key| self.tuner.record_feedback(key, tile, observed, incumbent))
    }
}
