//! WGSL lowering for the supported Titan Kernel IR subset.

use titan_kernel::{KernelAbi, KernelError, KernelModule};
use titan_types::{AbiHash, BackendId, DeviceFingerprint, KernelId, KernelLaunchMetadata};

/// WGSL bytes consumed by `DeviceSession::load`.
pub(super) struct WgslArtifact(Vec<u8>);

impl WgslArtifact {
    pub(super) fn from_driver_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut source = bytes.to_vec();
        if source.is_empty() {
            return Err("WGSL artifact is empty");
        }
        if let Some(nul) = source.iter().position(|byte| *byte == 0) {
            if nul + 1 != source.len() {
                return Err("WGSL artifact contains an interior NUL byte");
            }
        } else {
            source.push(0);
        }
        let text = std::str::from_utf8(&source[..source.len() - 1]).map_err(|_| "WGSL artifact is not UTF-8")?;
        if !text.contains("@compute") {
            return Err("artifact is not WGSL compute source");
        }
        Ok(Self(source))
    }

    pub(super) fn source(&self) -> &str {
        let end = self.0.len() - 1;
        std::str::from_utf8(&self.0[..end]).expect("validated UTF-8 WGSL")
    }
}

pub(super) struct LoweredWgsl {
    artifact: WgslArtifact,
    entry: String,
}

impl LoweredWgsl {
    pub(super) fn entry(&self) -> &str {
        &self.entry
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.artifact.0
    }
}

/// Kernel layout kind retained beside a loaded WebGPU pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuKernelKind {
    GemmF32,
    ElementwiseAddF32,
}

impl WgpuKernelKind {
    pub fn from_kernel_id(kernel_id: &str) -> Result<Self, KernelError> {
        match kernel_id {
            "gemm.f32" => Ok(Self::GemmF32),
            "elementwise.add.f32" => Ok(Self::ElementwiseAddF32),
            _ => Err(KernelError::Unsupported(format!("unsupported WebGPU kernel `{kernel_id}`"))),
        }
    }
}

pub(super) fn wgsl_entry_name(kernel_id: &str) -> Result<String, KernelError> {
    let mut name = String::from("titan_");
    for character in kernel_id.bytes() {
        match character {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => name.push(character as char),
            b'.' | b'-' => name.push('_'),
            _ => return Err(KernelError::Unsupported("kernel ID cannot be represented as a WGSL identifier".into())),
        }
    }
    if kernel_id.is_empty() {
        return Err(KernelError::Unsupported("kernel ID cannot be empty".into()));
    }
    Ok(name)
}

fn validate_device(fingerprint: &DeviceFingerprint) -> Result<(), KernelError> {
    if fingerprint.device.backend != BackendId::Wgpu {
        return Err(KernelError::Unsupported("WebGPU compiler requires a WebGPU device fingerprint".into()));
    }
    Ok(())
}

fn require_abi(actual: &KernelAbi, expected: &KernelAbi, detail: &str) -> Result<(), KernelError> {
    if actual.version != expected.version || actual.args != expected.args || actual.launch != expected.launch {
        return Err(KernelError::InvalidAbi(detail.into()));
    }
    Ok(())
}

fn require_empty_entry_block(ir: &KernelModule, detail: &str) -> Result<(), KernelError> {
    let block = ir
        .blocks
        .iter()
        .find(|block| block.id == ir.entry)
        .ok_or_else(|| KernelError::InvalidAbi("missing entry block".into()))?;
    if !block.params.is_empty() || !block.instructions.is_empty() {
        return Err(KernelError::InvalidAbi(detail.into()));
    }
    Ok(())
}

fn gemm_wgsl(entry: &str) -> String {
    format!(
        r#"struct GemmDims {{
  m: i32,
  n: i32,
  k: i32,
}}

@group(0) @binding(0) var<storage, read> gemm_a: array<f32>;
@group(0) @binding(1) var<storage, read> gemm_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> gemm_c: array<f32>;
@group(0) @binding(3) var<uniform> gemm_dims: GemmDims;

@compute @workgroup_size(128, 1, 1)
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let idx = gid.x;
  let mn = u32(gemm_dims.m) * u32(gemm_dims.n);
  if idx >= mn {{
    return;
  }}
  let col = idx % u32(gemm_dims.n);
  let row = idx / u32(gemm_dims.n);
  var acc: f32 = 0.0;
  for (var t: u32 = 0u; t < u32(gemm_dims.k); t = t + 1u) {{
    let a_idx = row * u32(gemm_dims.k) + t;
    let b_idx = t * u32(gemm_dims.n) + col;
    acc = acc + gemm_a[a_idx] * gemm_b[b_idx];
  }}
  gemm_c[idx] = acc;
}}
"#
    )
}

fn elementwise_add_wgsl(entry: &str) -> String {
    format!(
        r#"struct AddParams {{
  count: i32,
}}

@group(0) @binding(0) var<storage, read> add_lhs: array<f32>;
@group(0) @binding(1) var<storage, read> add_rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> add_out: array<f32>;
@group(0) @binding(3) var<uniform> add_params: AddParams;

@compute @workgroup_size(128, 1, 1)
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let idx = gid.x;
  if idx >= u32(add_params.count) {{
    return;
  }}
  add_out[idx] = add_lhs[idx] + add_rhs[idx];
}}
"#
    )
}

/// Lowers supported Titan Kernel IR into a WebGPU-loadable WGSL artifact.
pub(super) fn lower(ir: &KernelModule, abi: &KernelAbi, fingerprint: &DeviceFingerprint) -> Result<LoweredWgsl, KernelError> {
    validate_device(fingerprint)?;
    ir.verify()?;
    let entry = wgsl_entry_name(&ir.kernel_id.0)?;
    let source = if ir.kernel_id.0 == "gemm.f32" {
        require_abi(
            abi,
            &crate::gemm_f32_abi(),
            "WebGPU GEMM lowering requires the canonical gemm.f32 ABI",
        )?;
        require_empty_entry_block(ir, "WebGPU GEMM lowering requires the canonical empty gemm.f32 IR entry block")?;
        gemm_wgsl(&entry)
    } else if ir.kernel_id.0 == "elementwise.add.f32" {
        require_abi(
            abi,
            &crate::elementwise_add_f32_abi(),
            "WebGPU elementwise lowering requires three aligned f32 buffers and one i32 element-count scalar",
        )?;
        require_empty_entry_block(
            ir,
            "WebGPU elementwise lowering requires the canonical empty elementwise.add.f32 IR entry block",
        )?;
        elementwise_add_wgsl(&entry)
    } else {
        return Err(KernelError::Unsupported(format!("unsupported WebGPU kernel `{}`", ir.kernel_id.0)));
    };
    let mut artifact = source.into_bytes();
    artifact.push(0);
    Ok(LoweredWgsl { artifact: WgslArtifact(artifact), entry })
}

/// WebGPU compiler for the supported structured Titan Kernel IR subset.
#[derive(Debug, Default, Clone, Copy)]
pub struct WgpuCompiler;

/// A WebGPU artifact and the launch contract required to load it.
#[derive(Clone, Debug)]
pub struct WgpuArtifact {
    wgsl: Vec<u8>,
    abi_hash: AbiHash,
    metadata: KernelLaunchMetadata,
    kind: WgpuKernelKind,
}

impl WgpuArtifact {
    /// Returns the NUL-terminated WGSL consumed by `DeviceSession::load`.
    pub fn wgsl(&self) -> &[u8] {
        &self.wgsl
    }

    /// Returns the ABI hash checked by `DeviceSession::launch`.
    pub fn abi_hash(&self) -> &AbiHash {
        &self.abi_hash
    }

    /// Returns the retained launch metadata.
    pub fn metadata(&self) -> &KernelLaunchMetadata {
        &self.metadata
    }

    /// Returns the kernel layout kind used to build bind groups.
    pub fn kind(&self) -> WgpuKernelKind {
        self.kind
    }

    fn into_wgsl(self) -> Vec<u8> {
        self.wgsl
    }
}

impl WgpuCompiler {
    /// Lowers supported Titan Kernel IR into a WebGPU-loadable WGSL artifact.
    pub fn compile_artifact(
        &self,
        ir: &KernelModule,
        abi: &KernelAbi,
        fingerprint: &DeviceFingerprint,
    ) -> Result<WgpuArtifact, KernelError> {
        let lowered = lower(ir, abi, fingerprint)?;
        let kind = WgpuKernelKind::from_kernel_id(&ir.kernel_id.0)?;
        let entry = KernelId(lowered.entry().to_owned());
        let metadata = abi.launch_metadata(&entry)?;
        Ok(WgpuArtifact {
            wgsl: lowered.into_bytes(),
            abi_hash: abi.abi_hash(),
            metadata,
            kind,
        })
    }
}

impl titan_kernel::TargetCompiler for WgpuCompiler {
    fn target(&self) -> titan_kernel::KernelTarget {
        titan_kernel::KernelTarget::WgpuWgsl
    }

    fn compile(
        &self,
        ir: &KernelModule,
        abi: &KernelAbi,
        fingerprint: &DeviceFingerprint,
    ) -> Result<Vec<u8>, KernelError> {
        self.compile_artifact(ir, abi, fingerprint).map(WgpuArtifact::into_wgsl)
    }
}
