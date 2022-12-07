#![warn(missing_docs)]
//! A deterministic kernel IR that can be lowered to supported Titan backends.

use titan_hal::Backend;

/// Code-generation targets supported by the portable kernel layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelTarget {
    CpuSimd,
    Ptx,
    Hip,
    Metal,
    Wgsl,
}

/// Tunable launch parameters shared by generated and hand-authored kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    pub block_size: usize,
    pub vector_width: usize,
    pub pipeline_depth: usize,
    pub shared_memory_padding: usize,
}
impl Default for LaunchConfig {
    fn default() -> Self {
        Self { block_size: 256, vector_width: 1, pipeline_depth: 1, shared_memory_padding: 0 }
    }
}

/// A backend-neutral matrix multiplication kernel specification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatmulKernel {
    pub name: String,
    pub config: LaunchConfig,
}
impl MatmulKernel {
    /// Creates a named matrix multiplication kernel.
    pub fn new(name: impl Into<String>, config: LaunchConfig) -> Self {
        Self { name: name.into(), config }
    }
    /// Lowers this kernel into inspectable target source.
    pub fn compile(&self, target: KernelTarget) -> CompiledKernel {
        let language = match target {
            KernelTarget::CpuSimd => "rust-simd",
            KernelTarget::Ptx => "ptx",
            KernelTarget::Hip => "hip",
            KernelTarget::Metal => "metal",
            KernelTarget::Wgsl => "wgsl",
        };
        let source = format!(
            "// titan {language} kernel: {}\n// block={}, vector={}, pipeline={}, smem_padding={}\nmatmul(a, b, out);",
            self.name,
            self.config.block_size,
            self.config.vector_width,
            self.config.pipeline_depth,
            self.config.shared_memory_padding
        );
        CompiledKernel { target, source, config: self.config }
    }
    /// Returns the actual CPU launch configuration for a concrete backend.
    pub fn for_backend<B: Backend>(&self) -> CompiledKernel {
        self.compile(if B::NAME == "cpu" { KernelTarget::CpuSimd } else { KernelTarget::Wgsl })
    }
}

/// Lowered kernel source and its launch contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledKernel {
    pub target: KernelTarget,
    pub source: String,
    pub config: LaunchConfig,
}
