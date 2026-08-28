#![warn(missing_docs)]
//! Backend-neutral Kernel IR, ABI and candidate contracts.

use std::{fmt, sync::Arc};
pub use titan_hal::EncodedLaunchArgs;
use titan_hal::{Buffer, BufferBinding};
use titan_types::{AbiHash, BackendId, DType, DeviceFingerprint, KernelId, KernelLaunchMetadata, LaunchArgKind};

/// SSA value identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueId(pub u32);
/// SSA basic block identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockId(pub u32);
/// Kernel address space.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
    Constant,
    Register,
}
/// SSA scalar and pointer types.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IrType {
    I32,
    I64,
    F32,
    F16,
    BF16,
    Pointer { address_space: AddressSpace, dtype: DType },
}
/// Structured SSA instruction subset used by generated strategies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Instruction {
    Parameter { index: u32, ty: IrType },
    ConstF32(u32),
    Load { ptr: ValueId, ty: IrType },
    Store { ptr: ValueId, value: ValueId },
    Add { lhs: ValueId, rhs: ValueId },
    Mul { lhs: ValueId, rhs: ValueId },
    Fma { a: ValueId, b: ValueId, c: ValueId },
    Barrier,
}
/// SSA block with explicit parameter and instruction values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub params: Vec<(ValueId, IrType)>,
    pub instructions: Vec<(ValueId, Instruction)>,
}
/// Verifiable structured SSA kernel module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelModule {
    pub kernel_id: KernelId,
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
    pub abi: KernelAbi,
}
impl KernelModule {
    /// Verifies SSA definitions and references before code generation.
    pub fn verify(&self) -> Result<(), KernelError> {
        if self.blocks.iter().filter(|b| b.id == self.entry).count() != 1 {
            return Err(KernelError::InvalidAbi("missing entry block".into()));
        }
        let mut defs = std::collections::BTreeSet::new();
        for block in &self.blocks {
            for (id, _) in &block.params {
                if !defs.insert(*id) {
                    return Err(KernelError::InvalidAbi("duplicate SSA value".into()));
                }
            }
            for (id, instruction) in &block.instructions {
                if !defs.insert(*id) {
                    return Err(KernelError::InvalidAbi("duplicate SSA value".into()));
                }
                let refs = match instruction {
                    Instruction::Load { ptr, .. } => vec![*ptr],
                    Instruction::Store { ptr, value } => vec![*ptr, *value],
                    Instruction::Add { lhs, rhs } | Instruction::Mul { lhs, rhs } => vec![*lhs, *rhs],
                    Instruction::Fma { a, b, c } => vec![*a, *b, *c],
                    _ => vec![],
                };
                if refs.iter().any(|v| !defs.contains(v)) {
                    return Err(KernelError::InvalidAbi("use before definition".into()));
                }
            }
        }
        Ok(())
    }
}

/// Target language understood by a backend compiler.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelTarget {
    CpuAvx2,
    CudaPtx,
    RocmGfx1100,
    /// WGSL compute shaders for WebGPU adapters.
    WgpuWgsl,
}

/// A bounded launch configuration that strategies can tune.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LaunchConfig {
    pub block_size: u32,
    pub vector_width: u32,
    pub pipeline_depth: u32,
}
impl Default for LaunchConfig {
    fn default() -> Self {
        Self { block_size: 256, vector_width: 1, pipeline_depth: 1 }
    }
}

/// Fixed launch ABI identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAbi {
    pub version: u32,
    pub args: Vec<AbiArg>,
    pub launch: LaunchConfig,
    pub workspace_bytes: usize,
}
impl KernelAbi {
    /// Returns a stable hash for runtime validation.
    pub fn hash(&self) -> String {
        format!("v{}:{:?}:{:?}:{}", self.version, self.args, self.launch, self.workspace_bytes)
    }
    /// Returns canonical ABI bytes for artifact headers and launch validation.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.hash().into_bytes()
    }
    /// Returns the typed ABI digest used by HAL loaded-kernel handles.
    pub fn abi_hash(&self) -> AbiHash {
        AbiHash(self.hash())
    }
    /// Produces the backend-neutral metadata retained beside a loaded artifact.
    pub fn launch_metadata(&self, entry: &KernelId) -> Result<KernelLaunchMetadata, KernelError> {
        let arguments = self
            .args
            .iter()
            .map(|argument| match argument {
                AbiArg::Buffer { .. } => Ok(LaunchArgKind::Buffer),
                AbiArg::Scalar { dtype } => scalar_width(*dtype).map(|byte_len| LaunchArgKind::Scalar { byte_len }),
                AbiArg::Shape { rank } => Ok(LaunchArgKind::Shape { rank: *rank }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(KernelLaunchMetadata {
            version: self.version,
            entry: entry.0.clone(),
            arguments,
            block: [self.launch.block_size, 1, 1],
            shared_bytes: 0,
        })
    }
    /// Encodes a launch argument payload after checking the ABI arity.
    pub fn encode(&self, args: &[KernelArg]) -> Result<EncodedLaunchArgs, KernelError> {
        if args.len() != self.args.len() {
            return Err(KernelError::InvalidAbi(format!("expected {} arguments, got {}", self.args.len(), args.len())));
        }
        let mut out = Vec::new();
        let mut bindings = Vec::new();
        let mut seen_slots = std::collections::BTreeSet::new();
        let mut session_device = None;
        for (decl, arg) in self.args.iter().zip(args) {
            match (decl, arg) {
                (
                    AbiArg::Buffer { dtype, writable, alignment },
                    KernelArg::Buffer { dtype: actual, writable: aw, alignment: aa, slot, buffer },
                ) if dtype == actual && writable == aw && aa >= alignment => {
                    if let Some(expected) = session_device {
                        let actual = buffer.device();
                        if expected != actual {
                            return Err(KernelError::CrossSessionBinding { expected, actual });
                        }
                    }
                    else {
                        session_device = Some(buffer.device());
                    }
                    if !seen_slots.insert(*slot) {
                        return Err(KernelError::DuplicateBinding(*slot));
                    }
                    bindings.push(BufferBinding { slot: *slot, buffer: buffer.clone(), device_id: buffer.device() });
                    out.extend_from_slice(&slot.to_le_bytes());
                }
                (AbiArg::Buffer { .. }, KernelArg::BufferSlot { slot }) => {
                    return Err(KernelError::MissingBinding(*slot));
                }
                (AbiArg::Scalar { dtype }, KernelArg::Scalar { dtype: actual, bytes }) if dtype == actual => {
                    let expected = scalar_width(*dtype)? as usize;
                    if bytes.len() != expected {
                        return Err(KernelError::InvalidAbi(format!("scalar width: expected {expected}, got {}", bytes.len())));
                    }
                    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                    out.extend_from_slice(bytes);
                }
                (AbiArg::Shape { rank }, KernelArg::Shape { values }) if *rank as usize == values.len() => {
                    for value in values {
                        out.extend_from_slice(&value.to_le_bytes());
                    }
                }
                _ => return Err(KernelError::InvalidAbi("argument contract mismatch".into())),
            }
        }
        EncodedLaunchArgs::try_new(out, self.canonical_bytes(), bindings).map_err(|error| KernelError::InvalidAbi(error.detail))
    }
    /// 解码 opaque launch payload，并验证固定参数数量和类型编码。
    pub fn decode(&self, bytes: &[u8]) -> Result<Vec<DecodedArg>, KernelError> {
        let mut cursor = 0usize;
        let mut result = Vec::with_capacity(self.args.len());
        for decl in &self.args {
            match decl {
                AbiArg::Buffer { .. } => {
                    let end = cursor.checked_add(4).ok_or_else(|| KernelError::InvalidAbi("payload overflow".into()))?;
                    if end > bytes.len() {
                        return Err(KernelError::InvalidAbi("truncated buffer slot".into()));
                    }
                    result.push(DecodedArg::Buffer { slot: u32::from_le_bytes(bytes[cursor..end].try_into().unwrap()) });
                    cursor = end;
                }
                AbiArg::Scalar { .. } => {
                    let end = cursor.checked_add(4).ok_or_else(|| KernelError::InvalidAbi("payload overflow".into()))?;
                    if end > bytes.len() {
                        return Err(KernelError::InvalidAbi("truncated scalar length".into()));
                    }
                    let len = u32::from_le_bytes(bytes[cursor..end].try_into().unwrap()) as usize;
                    cursor = end;
                    let end = cursor.checked_add(len).ok_or_else(|| KernelError::InvalidAbi("payload overflow".into()))?;
                    if end > bytes.len() {
                        return Err(KernelError::InvalidAbi("truncated scalar".into()));
                    }
                    result.push(DecodedArg::Scalar { bytes: bytes[cursor..end].to_vec() });
                    cursor = end;
                }
                AbiArg::Shape { rank } => {
                    let len = *rank as usize * 8;
                    let end = cursor.checked_add(len).ok_or_else(|| KernelError::InvalidAbi("payload overflow".into()))?;
                    if end > bytes.len() {
                        return Err(KernelError::InvalidAbi("truncated shape".into()));
                    }
                    let mut values = Vec::with_capacity(*rank as usize);
                    for chunk in bytes[cursor..end].chunks_exact(8) {
                        values.push(u64::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    result.push(DecodedArg::Shape { values });
                    cursor = end;
                }
            }
        }
        if cursor != bytes.len() {
            return Err(KernelError::InvalidAbi("trailing ABI bytes".into()));
        }
        Ok(result)
    }
}

fn scalar_width(dtype: DType) -> Result<u32, KernelError> {
    match dtype {
        DType::F32 | DType::I32 => Ok(4),
        DType::F16 | DType::BF16 => Ok(2),
        DType::I64 => Ok(8),
    }
}

/// Typed ABI argument using an opaque buffer slot.
pub enum KernelArg {
    Buffer {
        slot: u32,
        dtype: DType,
        writable: bool,
        alignment: u32,
        buffer: Arc<dyn Buffer>,
    },
    /// A buffer slot without a concrete binding; encoding always rejects it.
    BufferSlot {
        slot: u32,
    },
    Scalar {
        dtype: DType,
        bytes: Vec<u8>,
    },
    Shape {
        values: Vec<u64>,
    },
}

impl fmt::Debug for KernelArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buffer { slot, dtype, writable, alignment, buffer } => f
                .debug_struct("Buffer")
                .field("slot", slot)
                .field("dtype", dtype)
                .field("writable", writable)
                .field("alignment", alignment)
                .field("device", &buffer.device())
                .field("identity", &buffer.identity())
                .finish(),
            Self::BufferSlot { slot } => f.debug_struct("BufferSlot").field("slot", slot).finish(),
            Self::Scalar { dtype, bytes } => f.debug_struct("Scalar").field("dtype", dtype).field("bytes", bytes).finish(),
            Self::Shape { values } => f.debug_struct("Shape").field("values", values).finish(),
        }
    }
}

impl Clone for KernelArg {
    fn clone(&self) -> Self {
        match self {
            Self::Buffer { slot, dtype, writable, alignment, buffer } => {
                Self::Buffer { slot: *slot, dtype: *dtype, writable: *writable, alignment: *alignment, buffer: buffer.clone() }
            }
            Self::BufferSlot { slot } => Self::BufferSlot { slot: *slot },
            Self::Scalar { dtype, bytes } => Self::Scalar { dtype: *dtype, bytes: bytes.clone() },
            Self::Shape { values } => Self::Shape { values: values.clone() },
        }
    }
}
/// Decoded ABI argument; buffer handles remain opaque slots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodedArg {
    Buffer { slot: u32 },
    Scalar { bytes: Vec<u8> },
    Shape { values: Vec<u64> },
}

/// An ABI argument with explicit ownership and access semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AbiArg {
    Buffer { dtype: DType, writable: bool, alignment: u32 },
    Scalar { dtype: DType },
    Shape { rank: u8 },
}

/// Generated or hand-authored candidate metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelRecipe {
    pub id: String,
    pub source: CandidateSource,
    pub target: KernelTarget,
    pub config: LaunchConfig,
    pub required_backend: BackendId,
    pub workspace_bytes: usize,
}

/// Candidate origin; both origins use the same ABI and validation path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateSource {
    Generated,
    Handwritten,
}

/// Deterministic generated candidate registry for the first vertical slice.
#[derive(Clone, Debug, Default)]
pub struct StrategyRegistry;
impl StrategyRegistry {
    /// Generates bounded baseline and launch variants for one backend.
    pub fn generate(&self, backend: BackendId, target: KernelTarget, max: usize) -> Vec<KernelRecipe> {
        [64u32, 128, 256, 512]
            .into_iter()
            .take(max.min(4))
            .map(|block_size| KernelRecipe {
                id: format!("generated/{backend:?}/{block_size}"),
                source: CandidateSource::Generated,
                target,
                config: LaunchConfig { block_size, ..LaunchConfig::default() },
                required_backend: backend,
                workspace_bytes: 0,
            })
            .collect()
    }
}

/// A target compiler implemented by a backend crate.
pub trait TargetCompiler: Send + Sync {
    fn target(&self) -> KernelTarget;
    fn compile(&self, ir: &KernelModule, abi: &KernelAbi, fingerprint: &DeviceFingerprint) -> Result<Vec<u8>, KernelError>;
}

/// Kernel construction or verification failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelError {
    Unsupported(String),
    InvalidAbi(String),
    Compile(String),
    /// A buffer argument did not provide a concrete binding.
    MissingBinding(u32),
    /// Two ABI arguments claimed the same buffer slot.
    DuplicateBinding(u32),
    /// Buffer arguments came from different device sessions.
    CrossSessionBinding {
        expected: titan_types::DeviceId,
        actual: titan_types::DeviceId,
    },
}
impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for KernelError {}
