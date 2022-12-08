#![warn(missing_docs)]
//! ROCm/HSA backend boundary. AMDGPU code-object loading belongs here, not in HAL.
use std::sync::Arc;
use titan_hal::{BackendDriver, DeviceSession, HalError};
use titan_types::{BackendId, DeviceFingerprint, DeviceId};
/// ROCm backend placeholder until the gfx1100 code-object adapter is installed.
#[derive(Debug, Default)]
pub struct RocmDriver;
impl BackendDriver for RocmDriver {
    fn id(&self) -> BackendId {
        BackendId::Rocm
    }
    fn enumerate(&self) -> Result<Vec<DeviceFingerprint>, HalError> {
        Err(HalError { operation: "rocm.devices", detail: "ROCm code-object adapter is not enabled".into() })
    }
    fn open(&self, _device: DeviceId) -> Result<Arc<dyn DeviceSession>, HalError> {
        Err(HalError { operation: "rocm.open", detail: "ROCm code-object adapter is not enabled".into() })
    }
}
