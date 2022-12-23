#![warn(missing_docs)]
//! Backend-neutral device, storage and asynchronous launch contracts.

use std::{any::Any, borrow::Borrow, collections::BTreeMap, fmt, sync::Arc};
use titan_types::{AbiHash, BackendId, DeviceFingerprint, DeviceId, KernelId, KernelLaunchMetadata};

/// Backend errors are intentionally opaque to the upper layers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HalError {
    /// Operation that failed.
    pub operation: &'static str,
    /// Human-readable backend detail.
    pub detail: String,
}

impl std::fmt::Display for HalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.operation, self.detail)
    }
}
impl std::error::Error for HalError {}

impl HalError {
    /// Map this HAL failure into a language-agnostic [`titan_types::TitanError`].
    pub fn to_titan_error(&self) -> titan_types::TitanError {
        let kind = titan_types::TitanErrorKind::from_hal_operation(self.operation);
        titan_types::TitanError::with_detail(kind, format!("{}: {}", self.operation, self.detail))
    }
}

impl From<HalError> for titan_types::TitanError {
    fn from(value: HalError) -> Self {
        value.to_titan_error()
    }
}

/// Opaque device allocation. It carries no host-slice access contract.
pub trait Buffer: fmt::Debug + Send + Sync {
    fn device(&self) -> DeviceId;
    fn byte_len(&self) -> usize;
    /// 后端私有的稳定 identity；上层不得将其解释为原生地址。
    fn identity(&self) -> u64;
}

/// A concrete slot-to-buffer launch binding.
#[derive(Clone)]
pub struct BufferBinding {
    /// ABI slot occupied by the buffer.
    pub slot: u32,
    /// Opaque device allocation retained for the launch.
    pub buffer: Arc<dyn Buffer>,
    /// Device/session identity that owns the allocation.
    pub device_id: DeviceId,
}

impl BufferBinding {
    /// Creates a binding and records the owning device/session explicitly.
    pub fn new(slot: u32, buffer: Arc<dyn Buffer>, device_id: DeviceId) -> Self {
        Self { slot, buffer, device_id }
    }
}

impl fmt::Debug for BufferBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferBinding")
            .field("slot", &self.slot)
            .field("device_id", &self.device_id)
            .field("buffer_identity", &self.buffer.identity())
            .finish()
    }
}
/// Opaque command stream.
pub trait Stream: fmt::Debug + Send + Sync {
    fn device(&self) -> DeviceId;
}
/// Opaque completion event.
pub trait Event: fmt::Debug + Send + Sync {
    fn device(&self) -> DeviceId;
}
/// Loaded kernel artifact.
pub trait LoadedKernel: fmt::Debug + Send + Sync + Any {
    fn device(&self) -> DeviceId;
    fn abi_hash(&self) -> &AbiHash;
    fn kernel_id(&self) -> &KernelId;
    /// Returns the validated byte-level launch contract.
    fn launch_metadata(&self) -> &KernelLaunchMetadata;
    /// Returns the concrete backend kernel for backend-local launch handling.
    fn as_any(&self) -> &dyn Any;
}

/// Opaque canonical launch arguments encoded by the kernel ABI.
///
/// The first field is the backend-facing payload.  The second field keeps
/// every buffer slot alive and explicit for the duration of a launch.  The
/// third field is the canonical ABI identity used to validate the artifact.
pub struct EncodedLaunchArgs(pub Vec<u8>, pub BTreeMap<u32, BufferBinding>, pub Vec<u8>);

impl Clone for EncodedLaunchArgs {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone(), self.2.clone())
    }
}

impl fmt::Debug for EncodedLaunchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodedLaunchArgs")
            .field("payload_len", &self.0.len())
            .field("bindings", &self.1.keys().collect::<Vec<_>>())
            .field("canonical_abi", &self.2)
            .finish()
    }
}

impl EncodedLaunchArgs {
    /// Returns the backend-facing launch payload.
    pub fn payload(&self) -> &[u8] {
        &self.0
    }

    /// Returns the canonical ABI bytes associated with this payload.
    pub fn canonical_abi(&self) -> &[u8] {
        &self.2
    }

    /// Alias for [`Self::canonical_abi`] with the full ABI terminology.
    pub fn canonical_abi_bytes(&self) -> &[u8] {
        self.canonical_abi()
    }

    /// Returns the explicit slot-to-buffer bindings.
    pub fn bindings(&self) -> &BTreeMap<u32, BufferBinding> {
        &self.1
    }

    /// Alias for [`Self::bindings`].
    pub fn slot_bindings(&self) -> &BTreeMap<u32, BufferBinding> {
        self.bindings()
    }

    /// Constructs encoded arguments while rejecting duplicate slots.
    pub fn try_new(
        payload: Vec<u8>,
        canonical_abi: Vec<u8>,
        bindings: impl IntoIterator<Item = BufferBinding>,
    ) -> Result<Self, HalError> {
        let mut indexed = BTreeMap::new();
        for binding in bindings {
            if indexed.insert(binding.slot, binding).is_some() {
                return Err(HalError { operation: "encode", detail: "duplicate buffer binding slot".into() });
            }
        }
        Ok(Self(payload, indexed, canonical_abi))
    }

    /// Verifies that all required slots belong to the launch device/session.
    pub fn validate_for<I, S>(&self, device_id: DeviceId, required_slots: I) -> Result<(), HalError>
    where
        I: IntoIterator<Item = S>,
        S: Borrow<u32>,
    {
        for slot in required_slots {
            let slot = slot.borrow();
            let binding = self.1.get(slot).ok_or_else(|| HalError {
                operation: "validate_bindings",
                detail: format!("missing buffer binding for slot {slot}"),
            })?;
            if binding.device_id != device_id || binding.buffer.device() != device_id {
                return Err(HalError {
                    operation: "validate_bindings",
                    detail: format!("cross-session buffer binding for slot {slot}"),
                });
            }
        }
        Ok(())
    }
}
/// Device launch geometry selected by a strategy.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LaunchGeometry {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
}

/// A backend session owns allocation, streams, events and kernel launches.
pub trait DeviceSession: Send + Sync {
    fn device(&self) -> DeviceId;
    fn fingerprint(&self) -> &DeviceFingerprint;
    fn allocate(&self, bytes: usize, alignment: usize) -> Result<Arc<dyn Buffer>, HalError>;
    fn upload(&self, stream: &dyn Stream, dst: &dyn Buffer, src: &[u8]) -> Result<Arc<dyn Event>, HalError>;
    fn download(&self, stream: &dyn Stream, src: &dyn Buffer, dst: &mut [u8]) -> Result<Arc<dyn Event>, HalError>;
    fn copy(&self, stream: &dyn Stream, dst: &dyn Buffer, src: &dyn Buffer, bytes: usize) -> Result<Arc<dyn Event>, HalError>;
    fn create_stream(&self) -> Result<Arc<dyn Stream>, HalError>;
    fn create_event(&self) -> Result<Arc<dyn Event>, HalError>;
    fn load(
        &self,
        artifact: &[u8],
        abi_hash: &AbiHash,
        metadata: KernelLaunchMetadata,
    ) -> Result<Arc<dyn LoadedKernel>, HalError>;
    fn launch(
        &self,
        stream: &dyn Stream,
        kernel: &dyn LoadedKernel,
        args: &EncodedLaunchArgs,
        geometry: &LaunchGeometry,
    ) -> Result<Arc<dyn Event>, HalError>;
    fn poll(&self, event: &dyn Event) -> Result<bool, HalError>;
    fn wait(&self, event: &dyn Event) -> Result<(), HalError>;
    /// Make `stream` wait until `event` completes (cross-stream dependency).
    ///
    /// Same-stream ordering is already implied by submission order; use this for
    /// upload→compute and other cross-stream edges. Host-side awaits must not
    /// substitute for this HAL primitive.
    fn wait_event(&self, stream: &dyn Stream, event: &dyn Event) -> Result<(), HalError>;
}

/// Discoverable backend driver.
pub trait BackendDriver: Send + Sync {
    fn id(&self) -> BackendId;
    fn enumerate(&self) -> Result<Vec<DeviceFingerprint>, HalError>;
    fn open(&self, device: DeviceId) -> Result<Arc<dyn DeviceSession>, HalError>;
}

/// Portable CPU backend identifier. The implementation is supplied by runtime registration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cpu;
