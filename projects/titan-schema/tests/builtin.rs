use titan_schema::{OpSpec, TensorSpec, builtin_operator_ids, builtin_registry};
use titan_types::{
    BackendId, DType, DeterminismPolicy, DeviceFingerprint, Layout, OperatorId, PrecisionPolicy, Shape, Strides,
    WorkspacePolicy,
};

#[test]
fn builtin_registry_exposes_generated_cpu_baselines() {
    let registry = builtin_registry();
    let device = DeviceFingerprint {
        device: titan_types::DeviceId { backend: BackendId::Cpu, ordinal: 0 },
        model: "test".into(),
        driver: "native".into(),
        capability_revision: "avx2-fma".into(),
    };
    for id in builtin_operator_ids() {
        let schema = registry.get(&id).unwrap();
        let tensor =
            TensorSpec { dtype: DType::F32, shape: Shape(vec![4]), strides: Strides(vec![1]), layout: Layout::Contiguous };
        let spec = OpSpec {
            operator: OperatorId(id.0.clone()),
            inputs: vec![tensor.clone()],
            outputs: vec![tensor],
            attrs: vec![],
            precision: PrecisionPolicy::Strict,
            determinism: DeterminismPolicy::Deterministic,
            workspace: WorkspacePolicy { max_bytes: 0 },
        };
        let recipes = schema.generate(&spec, &device, 1).unwrap();
        assert_eq!(recipes[0].source, titan_kernel::CandidateSource::Generated);
        schema.baseline_abi(&spec).unwrap();
    }
}
