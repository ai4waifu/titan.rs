use titan_kernel::{KernelTarget, LaunchConfig, MatmulKernel};
#[test]
fn generates_portable_source() {
    let compiled = MatmulKernel::new("matmul", LaunchConfig::default()).compile(KernelTarget::Wgsl);
    assert!(compiled.source.contains("wgsl"));
}
