#![warn(missing_docs)]
//! Stable protocol types shared by the graph, kernel and runtime layers.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};

/// A registered backend implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum BackendId {
    Cpu,
    Cuda,
    Rocm,
    /// WebGPU adapter backend (Vulkan / DX12 / Metal via wgpu).
    Wgpu,
}

/// A concrete device owned by a backend session.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceId {
    /// Backend owning the device.
    pub backend: BackendId,
    /// Device ordinal.
    pub ordinal: u32,
}

/// A stable hardware identity used by compilation and tuning caches.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DeviceFingerprint {
    /// Device identity string.
    pub device: DeviceId,
    /// Reported model/ISA.
    pub model: String,
    /// Driver or runtime version.
    pub driver: String,
    /// Backend capability revision.
    pub capability_revision: String,
}

/// Runtime dtype supported by the tensor and kernel protocols.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
}

/// Physical tensor layout.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum Layout {
    Contiguous,
    Strided,
    Transposed { permutation: Vec<u8> },
}

/// Canonical dynamic shape used by graph and cache protocols.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Shape(pub Vec<u64>);
/// Canonical signed strides measured in elements.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Strides(pub Vec<i64>);
/// Stable operator identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct OperatorId(pub String);
/// Stable candidate identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CandidateId(pub String);
/// Stable compiled kernel identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelId(pub String);
/// Stable ABI digest.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AbiHash(pub String);

/// Backend-neutral type of one encoded kernel launch argument.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LaunchArgKind {
    /// Opaque device buffer identified by a runtime buffer slot.
    Buffer,
    /// Fixed-width scalar bytes.
    Scalar { byte_len: u32 },
    /// A fixed-rank sequence of unsigned 64-bit dimensions.
    Shape { rank: u8 },
}

/// Stable launch contract retained with a backend-loaded artifact.
///
/// It deliberately describes bytes and buffer slots only. Operator semantics,
/// tensor layouts and generated source remain outside the HAL boundary.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct KernelLaunchMetadata {
    /// ABI revision used to decode launch bytes.
    pub version: u32,
    /// Entry-point name expected by the artifact.
    pub entry: String,
    /// Ordered encoded arguments.
    pub arguments: Vec<LaunchArgKind>,
    /// Thread-block geometry selected by the compiler.
    pub block: [u32; 3],
    /// Dynamic shared-memory bytes.
    pub shared_bytes: u32,
}
/// Stable artifact cache identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArtifactKey(pub String);
/// Precision selection contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PrecisionPolicy {
    Strict,
    AllowReduced,
}
/// Determinism selection contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeterminismPolicy {
    Relaxed,
    Deterministic,
}
/// Explicit workspace admission policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkspacePolicy {
    pub max_bytes: u64,
}
/// Cross-device transfer policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FallbackPolicy {
    Error,
    ExplicitCpu { max_transfer_bytes: u64 },
}
/// Source location propagated into structured execution errors.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub file: String,
    pub line: u32,
    pub column: u32,
}
/// Canonical operator attribute value.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    Bool(bool),
    Int(i64),
    Float(u64),
    String(String),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}
/// Ordered attribute map used for deterministic identity.
pub type AttrMap = BTreeMap<String, AttrValue>;
/// Alias and in-place access contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum AliasContract {
    NoAlias,
    MayAlias,
    MustAlias,
}
/// Memory effect contract for graph scheduling.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum MemoryEffect {
    Pure,
    Reads,
    Writes,
    ReadWrite,
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self).map(|_| ())
    }
}

pub use crate::errors::{Result, TitanError, TitanErrorKind};
mod errors;
