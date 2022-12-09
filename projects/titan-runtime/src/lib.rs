#![warn(missing_docs)]
//! The single eager/compiled dispatch coordinator.

use std::{collections::HashMap, path::Path, time::Duration};
use titan_autotune::{Autotuner, TuneBudget};
use titan_backend_cpu::compile_elementwise_add_f32;
use titan_backend_cuda::{
    BroadcastAddF32Descriptor, Conv2dF32Descriptor, CudaCompiler, GemmF32Descriptor, ScaledDotProductAttentionF32Descriptor,
    broadcast_add_f32_abi as cuda_broadcast_add_abi, concat_f32_abi as cuda_concat_abi, conv2d_f32_abi as cuda_conv2d_abi,
    elementwise_add_f32_abi as cuda_add_abi, gelu_f32_abi as cuda_gelu_abi, gemm_f32_abi as cuda_gemm_abi,
    group_norm_f32_abi as cuda_group_norm_abi, layer_norm_f32_abi as cuda_layer_norm_abi,
    quick_gelu_f32_abi as cuda_quick_gelu_abi, reduction_sum_f32_abi as cuda_reduction_sum_abi,
    resize_nearest2d_f32_abi as cuda_resize_nearest2d_abi, scaled_dot_product_attention_f32_abi as cuda_attention_abi,
    silu_f32_abi as cuda_silu_abi, slice_f32_abi as cuda_slice_abi, softmax_f32_abi as cuda_softmax_abi,
    transpose_f32_abi as cuda_transpose_abi,
};
use titan_graph::{Graph, OpRequest};
use titan_hal::LaunchGeometry;
use titan_kernel::{AddressSpace, BasicBlock, BlockId, Instruction, IrType, KernelAbi, KernelArg, KernelModule, ValueId};
use titan_profiler::Profiler;
use titan_schema::builtin_registry;
use titan_tensor::TensorHandle;
use titan_types::{
    AbiHash, BackendId, CandidateId, DType, DeviceFingerprint, KernelId, KernelLaunchMetadata, OperatorId, SourceSpan,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ArtifactCacheKey {
    kernel: KernelId,
    abi_hash: AbiHash,
    device: DeviceFingerprint,
}

#[derive(Clone, Debug)]
struct CachedArtifact {
    bytes: Vec<u8>,
    abi: KernelAbi,
    metadata: KernelLaunchMetadata,
}

/// Runtime tuning behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TunePolicy {
    Off,
    ReadOnly,
    OnMiss,
    Refresh,
}
/// Explicit fallback behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackPolicy {
    Error,
    ExplicitCpu,
}
/// Runtime configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub tune_policy: TunePolicy,
    pub fallback: FallbackPolicy,
    pub budget: TuneBudget,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { tune_policy: TunePolicy::OnMiss, fallback: FallbackPolicy::Error, budget: TuneBudget::default() }
    }
}

/// Hard resource limits used before admitting a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceBudget {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub concurrency: u32,
}
/// Resource request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceRequest {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub concurrency: u32,
}
/// Resource admission result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudgetReport {
    pub feasible: bool,
    pub device_available: u64,
    pub host_available: u64,
    pub concurrency_available: u32,
}
impl ResourceBudget {
    /// Checks hard limits without allocating.
    pub fn assess(self, request: ResourceRequest) -> ResourceBudgetReport {
        ResourceBudgetReport {
            feasible: request.device_bytes <= self.device_bytes
                && request.host_bytes <= self.host_bytes
                && request.concurrency <= self.concurrency,
            device_available: self.device_bytes.saturating_sub(request.device_bytes),
            host_available: self.host_bytes.saturating_sub(request.host_bytes),
            concurrency_available: self.concurrency.saturating_sub(request.concurrency),
        }
    }
}

/// Runtime state shared by eager and graph dispatch.
pub struct Runtime {
    config: RuntimeConfig,
    tuner: Autotuner,
    profiler: Profiler,
    schemas: titan_schema::SchemaRegistry,
    artifacts: HashMap<ArtifactCacheKey, CachedArtifact>,
    cache_hits: u64,
    cache_misses: u64,
}
impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("config", &self.config)
            .field("profiler", &self.profiler)
            .field("artifact_count", &self.artifacts.len())
            .finish()
    }
}

/// Asynchronously submitted execution metadata and outputs.
#[derive(Clone, Debug)]
pub struct ExecutionHandle {
    outputs: Vec<TensorHandle>,
    candidate: CandidateId,
    kernel: KernelId,
}
impl ExecutionHandle {
    /// Returns output handles without synchronizing.
    pub fn outputs(&self) -> &[TensorHandle] {
        &self.outputs
    }
    /// Returns the selected candidate identity.
    pub fn candidate_id(&self) -> &CandidateId {
        &self.candidate
    }
    /// Returns the loaded kernel identity.
    pub fn kernel_id(&self) -> &KernelId {
        &self.kernel
    }
    /// Waits for completion; CPU reference is already complete.
    pub fn wait(self) -> Result<ExecutionResult, ExecutionError> {
        Ok(ExecutionResult { outputs: self.outputs })
    }
}
/// Completed execution result.
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    pub outputs: Vec<TensorHandle>,
}
/// 编译选项；首版只允许精确 shape 计划。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompileOptions {
    /// 是否允许调优。
    pub tune: bool,
}
/// 已规范化的图执行计划。
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    requests: Vec<OpRequest>,
}
/// 图输入绑定。
#[derive(Clone, Debug, Default)]
pub struct GraphBindings;
impl ExecutionPlan {
    /// 返回计划中的节点数量。
    pub fn node_count(&self) -> usize {
        self.requests.len()
    }
    /// 使用与 eager 相同的 request dispatch 管线执行计划。
    pub fn execute(&self, runtime: &mut Runtime, _bindings: &GraphBindings) -> Result<ExecutionHandle, ExecutionError> {
        let request = self.requests.first().cloned().ok_or_else(|| ExecutionError {
            operator: OperatorId("graph.empty".into()),
            source: SourceSpan { file: "<graph>".into(), line: 0, column: 0 },
            phase: "compile",
            message: "graph has no executable nodes".into(),
        })?;
        runtime.execute(request)
    }
}
/// Structured runtime dispatch failure.
#[derive(Clone, Debug)]
pub struct ExecutionError {
    pub operator: OperatorId,
    pub source: SourceSpan,
    pub phase: &'static str,
    pub message: String,
}
impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.phase, self.message)
    }
}
impl std::error::Error for ExecutionError {}

fn hal_execution_error(operator: &OperatorId, source: &SourceSpan, phase: &'static str, message: String) -> ExecutionError {
    ExecutionError { operator: operator.clone(), source: source.clone(), phase, message }
}

fn cuda_add_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("elementwise.add.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock {
            id: BlockId(0),
            params: vec![],
            instructions: vec![
                (
                    ValueId(0),
                    Instruction::Parameter {
                        index: 0,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (
                    ValueId(1),
                    Instruction::Parameter {
                        index: 1,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (
                    ValueId(2),
                    Instruction::Parameter {
                        index: 2,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (ValueId(3), Instruction::Parameter { index: 3, ty: IrType::I32 }),
                (ValueId(4), Instruction::Load { ptr: ValueId(0), ty: IrType::F32 }),
                (ValueId(5), Instruction::Load { ptr: ValueId(1), ty: IrType::F32 }),
                (ValueId(6), Instruction::Add { lhs: ValueId(4), rhs: ValueId(5) }),
                (ValueId(7), Instruction::Store { ptr: ValueId(2), value: ValueId(6) }),
            ],
        }],
        abi,
    }
}

fn cuda_gemm_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("gemm.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_conv2d_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("conv2d.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_attention_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("scaled_dot_product_attention.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_broadcast_add_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("broadcast.add.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_silu_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("silu.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_gelu_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("gelu.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_quick_gelu_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("quick_gelu.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_softmax_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("softmax.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_reduction_sum_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("reduction.sum.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_concat_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("concat.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_transpose_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("transpose.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_slice_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("slice.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_resize_nearest2d_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("resize.nearest2d.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_layer_norm_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("layer_norm.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cuda_group_norm_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("group_norm.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut step = 1i64;
    for axis in (0..shape.len()).rev() {
        strides[axis] = step;
        step = step.saturating_mul(shape[axis] as i64);
    }
    strides
}

fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
    strides == contiguous_strides(shape)
}

fn coords(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        out[axis] = index % shape[axis];
        index /= shape[axis];
    }
    out
}

fn flatten(coords: &[usize], shape: &[usize]) -> usize {
    coords.iter().zip(shape).fold(0, |index, (coordinate, dimension)| index * dimension + coordinate)
}

fn ints(attrs: &titan_types::AttrMap, key: &str) -> Result<Vec<usize>, String> {
    match attrs.get(key) {
        Some(titan_types::AttrValue::Ints(values)) => {
            values.iter().map(|value| usize::try_from(*value).map_err(|_| format!("{key} must be non-negative"))).collect()
        }
        _ => Err(format!("missing integer-list attribute {key}")),
    }
}

fn int(attrs: &titan_types::AttrMap, key: &str, default: Option<usize>) -> Result<usize, String> {
    match attrs.get(key) {
        Some(titan_types::AttrValue::Int(value)) => usize::try_from(*value).map_err(|_| format!("{key} must be non-negative")),
        None => default.ok_or_else(|| format!("missing integer attribute {key}")),
        _ => Err(format!("{key} must be an integer")),
    }
}

fn bool_attr(attrs: &titan_types::AttrMap, key: &str, default: bool) -> Result<bool, String> {
    match attrs.get(key) {
        Some(titan_types::AttrValue::Bool(value)) => Ok(*value),
        None => Ok(default),
        _ => Err(format!("{key} must be a bool")),
    }
}

fn float_attr(attrs: &titan_types::AttrMap, key: &str, default: Option<f32>) -> Result<f32, String> {
    match attrs.get(key) {
        Some(titan_types::AttrValue::Float(value)) => Ok(f64::from_bits(*value) as f32),
        None => default.ok_or_else(|| format!("missing float attribute {key}")),
        _ => Err(format!("{key} must be a float")),
    }
}

fn quick_gelu_slope(attrs: &titan_types::AttrMap) -> Result<f32, String> {
    if attrs.keys().any(|key| key != "slope") {
        return Err("QuickGELU only accepts the slope attribute".into());
    }
    let slope = float_attr(attrs, "slope", Some(1.702))?;
    if !slope.is_finite() || slope <= 0.0 {
        return Err("QuickGELU slope must be finite and positive".into());
    }
    Ok(slope)
}

fn broadcast_add(inputs: &[Vec<f32>], shapes: &[Vec<usize>], output_shape: &[usize]) -> Result<Vec<f32>, String> {
    if inputs.len() != 2 || shapes.len() != 2 || shapes[0].len() != shapes[1].len() || shapes[0].len() != output_shape.len() {
        return Err("broadcast add requires two inputs with matching ranks".into());
    }
    let expected = shapes[0]
        .iter()
        .zip(&shapes[1])
        .map(|(lhs, rhs)| match (*lhs, *rhs) {
            (lhs, rhs) if lhs == rhs => Ok(lhs),
            (1, rhs) => Ok(rhs),
            (lhs, 1) => Ok(lhs),
            _ => Err("broadcast add dimensions must match or equal one"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if output_shape != expected {
        return Err("broadcast add output shape mismatch".into());
    }
    let mut output = vec![0.0; numel(output_shape)];
    for (out_index, value) in output.iter_mut().enumerate() {
        let coordinate = coords(out_index, output_shape);
        let lhs_coordinate = coordinate
            .iter()
            .zip(&shapes[0])
            .map(|(coordinate, dimension)| if *dimension == 1 { 0 } else { *coordinate })
            .collect::<Vec<_>>();
        let rhs_coordinate = coordinate
            .iter()
            .zip(&shapes[1])
            .map(|(coordinate, dimension)| if *dimension == 1 { 0 } else { *coordinate })
            .collect::<Vec<_>>();
        *value = inputs[0][flatten(&lhs_coordinate, &shapes[0])] + inputs[1][flatten(&rhs_coordinate, &shapes[1])];
    }
    Ok(output)
}

fn unary_same_shape(
    input: &[f32],
    shape: &[usize],
    output_shape: &[usize],
    operation: &str,
    f: impl Fn(f32) -> f32,
) -> Result<Vec<f32>, String> {
    if output_shape != shape {
        return Err(format!("{operation} output shape must match input"));
    }
    Ok(input.iter().copied().map(f).collect())
}

fn erf_approximation(value: f32) -> f32 {
    let sign = value.signum();
    let x = value.abs() as f64;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let mut p = 1.061_405_429;
    p = p * t - 1.453_152_027;
    p = p * t + 1.421_413_741;
    p = p * t - 0.284_496_736;
    p = p * t + 0.254_829_592;
    let polynomial = p * t;
    (sign as f64 * (1.0 - polynomial * (-x * x).exp())) as f32
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + erf_approximation(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn quick_gelu(value: f32, slope: f32) -> f32 {
    value * (1.0 / (1.0 + (-slope * value).exp()))
}

fn resize_nearest2d(input: &[f32], shape: &[usize], output_shape: &[usize]) -> Result<Vec<f32>, String> {
    if shape.len() != 4 || output_shape.len() != 4 {
        return Err("nearest resize requires rank-4 NCHW input and output".into());
    }
    let [batch, channels, input_height, input_width]: [usize; 4] = shape.try_into().expect("rank checked");
    let [out_batch, out_channels, output_height, output_width]: [usize; 4] = output_shape.try_into().expect("rank checked");
    if batch != out_batch || channels != out_channels {
        return Err("nearest resize must preserve N and C dimensions".into());
    }
    if input_height == 0 || input_width == 0 || output_height == 0 || output_width == 0 {
        return Err("nearest resize requires non-zero spatial dimensions".into());
    }
    let mut output = vec![0.0; numel(output_shape)];
    for n in 0..batch {
        for c in 0..channels {
            for out_h in 0..output_height {
                let in_h = out_h * input_height / output_height;
                for out_w in 0..output_width {
                    let in_w = out_w * input_width / output_width;
                    let input_index = ((n * channels + c) * input_height + in_h) * input_width + in_w;
                    let output_index = ((n * channels + c) * output_height + out_h) * output_width + out_w;
                    output[output_index] = input[input_index];
                }
            }
        }
    }
    Ok(output)
}

fn normalization_affine<'a>(
    inputs: &'a [Vec<f32>],
    shapes: &[Vec<usize>],
    feature_count: usize,
    operation: &str,
) -> Result<(Option<&'a [f32]>, Option<&'a [f32]>), String> {
    if !matches!(inputs.len(), 1..=3) {
        return Err(format!("{operation} requires input with optional weight and bias"));
    }
    if inputs.len() >= 2 && (shapes[1].as_slice() != [feature_count] || inputs[1].len() != feature_count) {
        return Err(format!("{operation} weight must have shape [{feature_count}]"));
    }
    if inputs.len() == 3 && (shapes[2].as_slice() != [feature_count] || inputs[2].len() != feature_count) {
        return Err(format!("{operation} bias must have shape [{feature_count}]"));
    }
    Ok((inputs.get(1).map(Vec::as_slice), inputs.get(2).map(Vec::as_slice)))
}

fn normalization_epsilon(attrs: &titan_types::AttrMap, operation: &str) -> Result<f32, String> {
    let epsilon = float_attr(attrs, "epsilon", None)?;
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(format!("{operation} epsilon must be finite and non-negative"));
    }
    Ok(epsilon)
}

fn layer_norm(
    inputs: &[Vec<f32>],
    shapes: &[Vec<usize>],
    attrs: &titan_types::AttrMap,
    output_shape: &[usize],
) -> Result<Vec<f32>, String> {
    let shape = &shapes[0];
    if shape.is_empty() || output_shape != shape {
        return Err("layer norm requires a non-scalar input and matching output shape".into());
    }
    let features = *shape.last().expect("rank checked");
    if features == 0 {
        return Err("layer norm requires a non-zero last dimension".into());
    }
    let epsilon = normalization_epsilon(attrs, "layer norm")?;
    let (weight, bias) = normalization_affine(inputs, shapes, features, "layer norm")?;
    let mut output = vec![0.0; inputs[0].len()];
    for (row, output_row) in inputs[0].chunks_exact(features).zip(output.chunks_exact_mut(features)) {
        let mean = row.iter().map(|value| *value as f64).sum::<f64>() / features as f64;
        let variance = row.iter().map(|value| (*value as f64 - mean).powi(2)).sum::<f64>() / features as f64;
        let inverse_stddev = 1.0 / (variance + epsilon as f64).sqrt();
        for (index, value) in row.iter().enumerate() {
            let mut normalized = ((*value as f64 - mean) * inverse_stddev) as f32;
            if let Some(weight) = weight {
                normalized *= weight[index];
            }
            if let Some(bias) = bias {
                normalized += bias[index];
            }
            output_row[index] = normalized;
        }
    }
    Ok(output)
}

fn group_norm(
    inputs: &[Vec<f32>],
    shapes: &[Vec<usize>],
    attrs: &titan_types::AttrMap,
    output_shape: &[usize],
) -> Result<Vec<f32>, String> {
    let shape = &shapes[0];
    if shape.len() != 4 || output_shape != shape {
        return Err("group norm requires rank-4 NCHW input and matching output shape".into());
    }
    let groups = int(attrs, "groups", None)?;
    let epsilon = normalization_epsilon(attrs, "group norm")?;
    let [batch, channels, height, width]: [usize; 4] = shape.clone().try_into().expect("rank checked");
    if channels == 0 || height == 0 || width == 0 || groups == 0 || channels % groups != 0 {
        return Err("group norm groups must be non-zero and divide non-zero channels".into());
    }
    let (weight, bias) = normalization_affine(inputs, shapes, channels, "group norm")?;
    let channels_per_group = channels / groups;
    let group_elements = channels_per_group * height * width;
    let mut output = vec![0.0; inputs[0].len()];
    for n in 0..batch {
        for group in 0..groups {
            let channel_start = group * channels_per_group;
            let mut sum = 0.0f64;
            for channel in channel_start..channel_start + channels_per_group {
                for spatial in 0..height * width {
                    sum += inputs[0][((n * channels + channel) * height * width) + spatial] as f64;
                }
            }
            let mean = sum / group_elements as f64;
            let mut squared_sum = 0.0f64;
            for channel in channel_start..channel_start + channels_per_group {
                for spatial in 0..height * width {
                    let value = inputs[0][((n * channels + channel) * height * width) + spatial] as f64;
                    squared_sum += (value - mean).powi(2);
                }
            }
            let inverse_stddev = 1.0 / (squared_sum / group_elements as f64 + epsilon as f64).sqrt();
            for channel in channel_start..channel_start + channels_per_group {
                for spatial in 0..height * width {
                    let index = ((n * channels + channel) * height * width) + spatial;
                    let mut normalized = ((inputs[0][index] as f64 - mean) * inverse_stddev) as f32;
                    if let Some(weight) = weight {
                        normalized *= weight[channel];
                    }
                    if let Some(bias) = bias {
                        normalized += bias[channel];
                    }
                    output[index] = normalized;
                }
            }
        }
    }
    Ok(output)
}

fn transpose(input: &[f32], shape: &[usize], attrs: &titan_types::AttrMap, output_shape: &[usize]) -> Result<Vec<f32>, String> {
    let permutation = ints(attrs, "permutation")?;
    if permutation.len() != shape.len() || permutation.iter().any(|axis| *axis >= shape.len()) || {
        let mut unique = permutation.clone();
        unique.sort_unstable();
        unique.dedup();
        unique.len() != shape.len()
    } {
        return Err("transpose permutation must contain every axis exactly once".into());
    }
    if output_shape != permutation.iter().map(|axis| shape[*axis]).collect::<Vec<_>>() {
        return Err("transpose output shape mismatch".into());
    }
    let mut output = vec![0.0; input.len()];
    for (out_index, value) in output.iter_mut().enumerate() {
        let out_coordinates = coords(out_index, output_shape);
        let mut in_coordinates = vec![0; shape.len()];
        for (out_axis, in_axis) in permutation.iter().enumerate() {
            in_coordinates[*in_axis] = out_coordinates[out_axis];
        }
        *value = input[flatten(&in_coordinates, shape)];
    }
    Ok(output)
}

fn slice(input: &[f32], shape: &[usize], attrs: &titan_types::AttrMap, output_shape: &[usize]) -> Result<Vec<f32>, String> {
    let starts = ints(attrs, "starts")?;
    let ends = ints(attrs, "ends")?;
    let axes = ints(attrs, "axes")?;
    if starts.len() != ends.len() || starts.len() != axes.len() || axes.iter().any(|axis| *axis >= shape.len()) {
        return Err("slice attributes have invalid axes".into());
    }
    let mut unique_axes = axes.clone();
    unique_axes.sort_unstable();
    unique_axes.dedup();
    if unique_axes.len() != axes.len() {
        return Err("slice axes must be unique".into());
    }
    let mut offsets = vec![0; shape.len()];
    let mut expected = shape.to_vec();
    for index in 0..axes.len() {
        let axis = axes[index];
        if starts[index] > ends[index] || ends[index] > shape[axis] {
            return Err("slice bounds are invalid".into());
        }
        offsets[axis] = starts[index];
        expected[axis] = ends[index] - starts[index];
    }
    if output_shape != expected {
        return Err("slice output shape mismatch".into());
    }
    let mut output = vec![0.0; numel(output_shape)];
    for (out_index, value) in output.iter_mut().enumerate() {
        let mut in_coordinates = coords(out_index, output_shape);
        for axis in 0..shape.len() {
            in_coordinates[axis] += offsets[axis];
        }
        *value = input[flatten(&in_coordinates, shape)];
    }
    Ok(output)
}

fn concat(
    inputs: &[Vec<f32>],
    shapes: &[Vec<usize>],
    attrs: &titan_types::AttrMap,
    output_shape: &[usize],
) -> Result<Vec<f32>, String> {
    let axis = int(attrs, "axis", Some(0))?;
    if inputs.len() < 2 || shapes.iter().any(|shape| shape.len() != output_shape.len()) || axis >= output_shape.len() {
        return Err("concat requires compatible ranks and at least two inputs".into());
    }
    let mut axis_size = 0usize;
    for shape in shapes {
        for dimension in 0..shape.len() {
            if dimension != axis && shape[dimension] != output_shape[dimension] {
                return Err("concat non-axis dimensions must match".into());
            }
        }
        axis_size = axis_size.checked_add(shape[axis]).ok_or("concat axis overflows host usize")?;
    }
    if axis_size != output_shape[axis] {
        return Err("concat output shape mismatch".into());
    }
    let mut output = vec![0.0; numel(output_shape)];
    for out_index in 0..output.len() {
        let coordinate = coords(out_index, output_shape);
        let mut base = 0usize;
        for (input, shape) in inputs.iter().zip(shapes) {
            if coordinate[axis] < base + shape[axis] {
                let mut input_coordinate = coordinate.clone();
                input_coordinate[axis] -= base;
                output[out_index] = input[flatten(&input_coordinate, shape)];
                break;
            }
            base += shape[axis];
        }
    }
    Ok(output)
}

fn reduction_sum(
    input: &[f32],
    shape: &[usize],
    attrs: &titan_types::AttrMap,
    output_shape: &[usize],
) -> Result<Vec<f32>, String> {
    let axes = ints(attrs, "axes")?;
    let keepdims = bool_attr(attrs, "keepdims", false)?;
    if axes.is_empty() || axes.iter().any(|axis| *axis >= shape.len()) {
        return Err("reduction axes are invalid".into());
    }
    let mut reduced = axes.clone();
    reduced.sort_unstable();
    reduced.dedup();
    if reduced.len() != axes.len() {
        return Err("reduction axes must be unique".into());
    }
    let expected: Vec<usize> = if keepdims {
        shape.iter().enumerate().map(|(axis, dimension)| if reduced.contains(&axis) { 1 } else { *dimension }).collect()
    }
    else {
        shape.iter().enumerate().filter_map(|(axis, dimension)| (!reduced.contains(&axis)).then_some(*dimension)).collect()
    };
    if output_shape != expected {
        return Err("reduction output shape mismatch".into());
    }
    let mut output = vec![0.0; numel(output_shape)];
    for (input_index, value) in input.iter().enumerate() {
        let in_coordinate = coords(input_index, shape);
        let out_coordinate: Vec<usize> = if keepdims {
            in_coordinate.iter().enumerate().map(|(axis, value)| if reduced.contains(&axis) { 0 } else { *value }).collect()
        }
        else {
            in_coordinate.iter().enumerate().filter_map(|(axis, value)| (!reduced.contains(&axis)).then_some(*value)).collect()
        };
        output[flatten(&out_coordinate, output_shape)] += value;
    }
    Ok(output)
}

fn softmax(input: &[f32], shape: &[usize], attrs: &titan_types::AttrMap, output_shape: &[usize]) -> Result<Vec<f32>, String> {
    let axis = int(attrs, "axis", Some(shape.len().saturating_sub(1)))?;
    if axis >= shape.len() || output_shape != shape {
        return Err("softmax axis or output shape is invalid".into());
    }
    let outer = numel(&shape[..axis]);
    let axis_len = shape[axis];
    let inner = numel(&shape[axis + 1..]);
    let mut output = vec![0.0; input.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let base = (outer_index * axis_len) * inner + inner_index;
            let maximum = (0..axis_len).map(|axis_index| input[base + axis_index * inner]).fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = (0..axis_len).map(|axis_index| (input[base + axis_index * inner] - maximum).exp()).sum();
            for axis_index in 0..axis_len {
                output[base + axis_index * inner] = (input[base + axis_index * inner] - maximum).exp() / denominator;
            }
        }
    }
    Ok(output)
}

fn gemm(
    inputs: &[Vec<f32>],
    shapes: &[Vec<usize>],
    attrs: &titan_types::AttrMap,
    output_shape: &[usize],
) -> Result<Vec<f32>, String> {
    if inputs.len() != 2 || shapes.len() != 2 || shapes[0].len() != 2 || shapes[1].len() != 2 {
        return Err("gemm requires exactly two rank-2 inputs".into());
    }
    let transpose_lhs = bool_attr(attrs, "transpose_lhs", false)?;
    let transpose_rhs = bool_attr(attrs, "transpose_rhs", false)?;
    let [lhs_rows, lhs_cols]: [usize; 2] = shapes[0].as_slice().try_into().expect("rank checked");
    let [rhs_rows, rhs_cols]: [usize; 2] = shapes[1].as_slice().try_into().expect("rank checked");
    let (m, k) = if transpose_lhs { (lhs_cols, lhs_rows) } else { (lhs_rows, lhs_cols) };
    let (rhs_k, n) = if transpose_rhs { (rhs_cols, rhs_rows) } else { (rhs_rows, rhs_cols) };
    if k != rhs_k {
        return Err("gemm inner dimensions must match".into());
    }
    if output_shape != [m, n] {
        return Err("gemm output shape mismatch".into());
    }
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                let lhs_index = if transpose_lhs { inner * lhs_cols + row } else { row * lhs_cols + inner };
                let rhs_index = if transpose_rhs { column * rhs_cols + inner } else { inner * rhs_cols + column };
                sum += inputs[0][lhs_index] * inputs[1][rhs_index];
            }
            output[row * n + column] = sum;
        }
    }
    Ok(output)
}

fn conv2d(
    inputs: &[Vec<f32>],
    shapes: &[Vec<usize>],
    attrs: &titan_types::AttrMap,
    output_shape: &[usize],
) -> Result<Vec<f32>, String> {
    if !matches!(inputs.len(), 2..=3) || shapes.len() != inputs.len() || shapes[0].len() != 4 || shapes[1].len() != 4 {
        return Err("conv2d requires rank-4 NCHW input, rank-4 OIHW weight, and optional bias".into());
    }
    let stride_h = int(attrs, "stride_h", None)?;
    let stride_w = int(attrs, "stride_w", None)?;
    let pad_h = int(attrs, "pad_h", None)?;
    let pad_w = int(attrs, "pad_w", None)?;
    let dilation_h = int(attrs, "dilation_h", None)?;
    let dilation_w = int(attrs, "dilation_w", None)?;
    let groups = int(attrs, "groups", None)?;
    if stride_h == 0 || stride_w == 0 || dilation_h == 0 || dilation_w == 0 || groups == 0 {
        return Err("conv2d stride, dilation, and groups must be non-zero".into());
    }
    let [batch, input_channels, input_height, input_width]: [usize; 4] = shapes[0].as_slice().try_into().expect("rank checked");
    let [output_channels, weight_channels, kernel_height, kernel_width]: [usize; 4] =
        shapes[1].as_slice().try_into().expect("rank checked");
    if input_channels == 0
        || input_height == 0
        || input_width == 0
        || output_channels == 0
        || kernel_height == 0
        || kernel_width == 0
        || input_channels % groups != 0
        || output_channels % groups != 0
        || weight_channels != input_channels / groups
    {
        return Err("conv2d channels, groups, and kernel dimensions are invalid".into());
    }
    if inputs.len() == 3 && (shapes[2].as_slice() != [output_channels] || inputs[2].len() != output_channels) {
        return Err("conv2d bias must have shape [output_channels]".into());
    }
    let effective_kernel_h = dilation_h
        .checked_mul(kernel_height - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "conv2d kernel geometry overflows".to_string())?;
    let effective_kernel_w = dilation_w
        .checked_mul(kernel_width - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "conv2d kernel geometry overflows".to_string())?;
    let padded_height = input_height
        .checked_add(pad_h.checked_mul(2).ok_or_else(|| "conv2d padding overflows".to_string())?)
        .ok_or_else(|| "conv2d padding overflows".to_string())?;
    let padded_width = input_width
        .checked_add(pad_w.checked_mul(2).ok_or_else(|| "conv2d padding overflows".to_string())?)
        .ok_or_else(|| "conv2d padding overflows".to_string())?;
    if effective_kernel_h > padded_height || effective_kernel_w > padded_width {
        return Err("conv2d kernel geometry exceeds padded input".into());
    }
    let output_height = (padded_height - effective_kernel_h) / stride_h + 1;
    let output_width = (padded_width - effective_kernel_w) / stride_w + 1;
    if output_shape != [batch, output_channels, output_height, output_width] {
        return Err("conv2d output shape mismatch".into());
    }
    let input_channels_per_group = input_channels / groups;
    let output_channels_per_group = output_channels / groups;
    let mut output = vec![0.0; numel(output_shape)];
    for n in 0..batch {
        for output_channel in 0..output_channels {
            let group = output_channel / output_channels_per_group;
            for out_h in 0..output_height {
                for out_w in 0..output_width {
                    let mut sum = inputs.get(2).map_or(0.0, |bias| bias[output_channel]);
                    for channel in 0..input_channels_per_group {
                        for kernel_h in 0..kernel_height {
                            let input_h = out_h * stride_h + kernel_h * dilation_h;
                            if input_h < pad_h || input_h - pad_h >= input_height {
                                continue;
                            }
                            for kernel_w in 0..kernel_width {
                                let input_w = out_w * stride_w + kernel_w * dilation_w;
                                if input_w < pad_w || input_w - pad_w >= input_width {
                                    continue;
                                }
                                let input_channel = group * input_channels_per_group + channel;
                                let input_index = ((n * input_channels + input_channel) * input_height + (input_h - pad_h))
                                    * input_width
                                    + (input_w - pad_w);
                                let weight_index = ((output_channel * input_channels_per_group + channel) * kernel_height
                                    + kernel_h)
                                    * kernel_width
                                    + kernel_w;
                                sum += inputs[0][input_index] * inputs[1][weight_index];
                            }
                        }
                    }
                    output[((n * output_channels + output_channel) * output_height + out_h) * output_width + out_w] = sum;
                }
            }
        }
    }
    Ok(output)
}

fn scaled_dot_product_attention(
    inputs: &[Vec<f32>],
    shapes: &[Vec<usize>],
    attrs: &titan_types::AttrMap,
    output_shape: &[usize],
) -> Result<Vec<f32>, String> {
    if inputs.len() != 3 || shapes.len() != 3 || shapes.iter().any(|shape| shape.len() != 4) {
        return Err("scaled dot-product attention requires exactly three rank-4 BHTD inputs".into());
    }
    if attrs.keys().any(|key| key.contains("mask") || key.contains("causal")) {
        return Err("scaled dot-product attention mask and causal attributes are not implemented".into());
    }
    let [batch, heads, query_tokens, depth]: [usize; 4] = shapes[0].as_slice().try_into().expect("rank checked");
    let [key_batch, key_heads, key_tokens, key_depth]: [usize; 4] = shapes[1].as_slice().try_into().expect("rank checked");
    let [value_batch, value_heads, value_tokens, value_depth]: [usize; 4] =
        shapes[2].as_slice().try_into().expect("rank checked");
    if batch != key_batch
        || batch != value_batch
        || heads != key_heads
        || heads != value_heads
        || depth != key_depth
        || depth != value_depth
        || key_tokens != value_tokens
    {
        return Err("scaled dot-product attention B, H, D, and K/V sequence dimensions must match".into());
    }
    if query_tokens == 0 || key_tokens == 0 || depth == 0 {
        return Err("scaled dot-product attention requires non-zero Tq, Tk, and D dimensions".into());
    }
    if output_shape != [batch, heads, query_tokens, depth] {
        return Err("scaled dot-product attention output shape mismatch".into());
    }
    let scale = 1.0 / (depth as f32).sqrt();
    let mut output = vec![0.0; numel(output_shape)];
    for batch_index in 0..batch {
        for head in 0..heads {
            for query_token in 0..query_tokens {
                let mut scores = vec![0.0; key_tokens];
                for key_token in 0..key_tokens {
                    for dimension in 0..depth {
                        let query_index = ((batch_index * heads + head) * query_tokens + query_token) * depth + dimension;
                        let key_index = ((batch_index * heads + head) * key_tokens + key_token) * depth + dimension;
                        scores[key_token] += inputs[0][query_index] * inputs[1][key_index];
                    }
                    scores[key_token] *= scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
                for dimension in 0..depth {
                    let mut result = 0.0;
                    for key_token in 0..key_tokens {
                        let weight = (scores[key_token] - maximum).exp() / denominator;
                        let value_index = ((batch_index * heads + head) * key_tokens + key_token) * depth + dimension;
                        result += weight * inputs[2][value_index];
                    }
                    output[((batch_index * heads + head) * query_tokens + query_token) * depth + dimension] = result;
                }
            }
        }
    }
    Ok(output)
}

impl Runtime {
    /// Opens a runtime with default policy.
    pub fn open(path: impl AsRef<Path>) -> Self {
        Self::with_config(path, RuntimeConfig::default())
    }
    /// Opens a runtime with explicit tuning and fallback policy.
    pub fn with_config(path: impl AsRef<Path>, config: RuntimeConfig) -> Self {
        Self {
            config,
            tuner: Autotuner::open(path.as_ref()),
            profiler: Profiler::default(),
            schemas: builtin_registry(),
            artifacts: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    /// Executes one normalized request through the single runtime entry point.
    pub fn execute(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let name = request.operator.0.as_str();
        if name == "gemm" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_gemm(request);
        }
        if name == "conv2d" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_conv2d(request);
        }
        if name == "scaled_dot_product_attention"
            && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda)
        {
            return self.execute_cuda_attention(request);
        }
        if name == "broadcast.add" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_broadcast_add(request);
        }
        if name == "silu" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_silu(request);
        }
        if name == "gelu" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_gelu(request);
        }
        if name == "quick_gelu" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_quick_gelu(request);
        }
        if name == "softmax" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_softmax(request);
        }
        if name == "reduction.sum" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_reduction_sum(request);
        }
        if name == "concat" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_concat(request);
        }
        if name == "transpose" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_transpose(request);
        }
        if name == "slice" && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda) {
            return self.execute_cuda_slice(request);
        }
        if matches!(name, "resize.nearest2d" | "resize_nearest2d")
            && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda)
        {
            return self.execute_cuda_resize_nearest2d(request);
        }
        if matches!(name, "layer_norm" | "layer.norm")
            && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda)
        {
            return self.execute_cuda_layer_norm(request);
        }
        if matches!(name, "group_norm" | "group.norm")
            && request.inputs.first().is_some_and(|input| input.device().backend == BackendId::Cuda)
        {
            return self.execute_cuda_group_norm(request);
        }
        if matches!(
            name,
            "reshape"
                | "transpose"
                | "slice"
                | "concat"
                | "reduction.sum"
                | "softmax"
                | "broadcast.add"
                | "silu"
                | "gelu"
                | "quick_gelu"
                | "resize.nearest2d"
                | "resize_nearest2d"
                | "layer_norm"
                | "layer.norm"
                | "group_norm"
                | "group.norm"
                | "gemm"
                | "conv2d"
                | "scaled_dot_product_attention"
        ) {
            return self.execute_cpu_reference(request);
        }
        self.execute_compiled_add(request)
    }

    fn execute_cuda_silu(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA SiLU requires exactly one input and one output".into()));
        }
        if !request.attrs.is_empty() {
            return Err(fail("contract", "CUDA SiLU does not accept attributes".into()));
        }
        let input = &request.inputs[0];
        let output_spec = &request.outputs[0];
        if input.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA SiLU requires F32 input and output".into()));
        }
        if input.device().backend != BackendId::Cuda {
            return Err(fail("contract", "CUDA SiLU requires a CUDA input device".into()));
        }
        if !is_contiguous(input.shape(), input.strides()) {
            return Err(fail("contract", "CUDA SiLU requires a contiguous input".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA SiLU output shape exceeds host usize".into()))?;
        if output_shape != input.shape()
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA SiLU requires a contiguous output with the same shape as the input".into()));
        }
        let session = input.session().ok_or_else(|| fail("contract", "CUDA SiLU input has no backend storage".into()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("silu.f32".into());
        let compiled = self
            .cached_cuda_silu_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let element_count = i32::try_from(numel(&output_shape))
            .map_err(|_| fail("contract", "CUDA SiLU element count exceeds i32 ABI".into()))?;
        let args = compiled
            .abi
            .encode(&vec![
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: element_count.to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [numel(&output_shape).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_gelu(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA GELU requires exactly one input and one output".into()));
        }
        if !request.attrs.is_empty() {
            return Err(fail("contract", "CUDA GELU does not accept attributes".into()));
        }
        let input = &request.inputs[0];
        let output_spec = &request.outputs[0];
        if input.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA GELU requires F32 input and output".into()));
        }
        if input.device().backend != BackendId::Cuda {
            return Err(fail("contract", "CUDA GELU requires a CUDA input device".into()));
        }
        if !is_contiguous(input.shape(), input.strides()) {
            return Err(fail("contract", "CUDA GELU requires a contiguous input".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA GELU output shape exceeds host usize".into()))?;
        if output_shape != input.shape()
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA GELU requires a contiguous output with the same shape as the input".into()));
        }
        let session = input.session().ok_or_else(|| fail("contract", "CUDA GELU input has no backend storage".into()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("gelu.f32".into());
        let compiled = self
            .cached_cuda_gelu_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let element_count = i32::try_from(numel(&output_shape))
            .map_err(|_| fail("contract", "CUDA GELU element count exceeds i32 ABI".into()))?;
        let args = compiled
            .abi
            .encode(&vec![
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: element_count.to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [numel(&output_shape).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_quick_gelu(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA QuickGELU requires exactly one input and one output".into()));
        }
        let slope = quick_gelu_slope(&request.attrs).map_err(|message| fail("contract", message))?;
        let input = &request.inputs[0];
        let output_spec = &request.outputs[0];
        if input.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA QuickGELU requires F32 input and output".into()));
        }
        if input.device().backend != BackendId::Cuda {
            return Err(fail("contract", "CUDA QuickGELU requires a CUDA input device".into()));
        }
        if !is_contiguous(input.shape(), input.strides()) {
            return Err(fail("contract", "CUDA QuickGELU requires a contiguous input".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA QuickGELU output shape exceeds host usize".into()))?;
        if output_shape != input.shape()
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail(
                "contract",
                "CUDA QuickGELU requires a contiguous output with the same shape as the input".into(),
            ));
        }
        let session = input.session().ok_or_else(|| fail("contract", "CUDA QuickGELU input has no backend storage".into()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("quick_gelu.f32".into());
        let compiled = self
            .cached_cuda_quick_gelu_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let element_count = i32::try_from(numel(&output_shape))
            .map_err(|_| fail("contract", "CUDA QuickGELU element count exceeds i32 ABI".into()))?;
        let args = compiled
            .abi
            .encode(&vec![
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: element_count.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::F32, bytes: slope.to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [numel(&output_shape).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_softmax(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA softmax requires exactly one input and one output".into()));
        }
        if request.attrs.keys().any(|key| key != "axis") {
            return Err(fail("contract", "CUDA softmax only accepts the axis attribute".into()));
        }
        let input = &request.inputs[0];
        let output_spec = &request.outputs[0];
        if input.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA softmax requires F32 input and output".into()));
        }
        if input.device().backend != BackendId::Cuda {
            return Err(fail("contract", "CUDA softmax requires a CUDA input device".into()));
        }
        if input.shape().is_empty() {
            return Err(fail("contract", "CUDA softmax requires a non-scalar input".into()));
        }
        if !is_contiguous(input.shape(), input.strides()) {
            return Err(fail("contract", "CUDA softmax requires a contiguous input".into()));
        }
        let axis = int(&request.attrs, "axis", Some(input.shape().len() - 1)).map_err(|message| fail("contract", message))?;
        if axis != input.shape().len() - 1 {
            return Err(fail("contract", "CUDA softmax only supports the last axis".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA softmax output shape exceeds host usize".into()))?;
        if output_shape != input.shape()
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA softmax requires a contiguous output with the same shape as the input".into()));
        }
        let rows = numel(&input.shape()[..input.shape().len() - 1]);
        let cols = *input.shape().last().expect("non-scalar softmax input");
        if rows == 0 || cols == 0 {
            return Err(fail("contract", "CUDA softmax requires non-zero dimensions".into()));
        }
        let rows = i32::try_from(rows).map_err(|_| fail("contract", "CUDA softmax row count exceeds i32 ABI".into()))?;
        let cols = i32::try_from(cols).map_err(|_| fail("contract", "CUDA softmax axis length exceeds i32 ABI".into()))?;
        let session = input.session().ok_or_else(|| fail("contract", "CUDA softmax input has no backend storage".into()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("softmax.f32".into());
        let compiled = self
            .cached_cuda_softmax_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let args = compiled
            .abi
            .encode(&vec![
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: rows.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: cols.to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [(u32::try_from(rows).unwrap() as usize).div_ceil(block as usize).max(1) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_reduction_sum(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA reduction.sum requires exactly one input and one output".into()));
        }
        if request.attrs.keys().any(|key| key != "axes" && key != "keepdims") {
            return Err(fail("contract", "CUDA reduction.sum only accepts axes and keepdims attributes".into()));
        }
        let input = &request.inputs[0];
        let spec = &request.outputs[0];
        if input.dtype() != DType::F32 || spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA reduction.sum requires F32 input and output".into()));
        }
        if input.shape().len() < 2 || !is_contiguous(input.shape(), input.strides()) {
            return Err(fail("contract", "CUDA reduction.sum requires a contiguous non-scalar input".into()));
        }
        let axes = ints(&request.attrs, "axes").map_err(|message| fail("contract", message))?;
        let keepdims = bool_attr(&request.attrs, "keepdims", false).map_err(|message| fail("contract", message))?;
        let last = input.shape().len() - 1;
        if axes != [last] || keepdims {
            return Err(fail("contract", "CUDA reduction.sum only supports axes=[last axis] with keepdims=false".into()));
        }
        let output_shape = spec
            .shape
            .0
            .iter()
            .map(|d| usize::try_from(*d))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA reduction.sum output shape exceeds host usize".into()))?;
        if output_shape != input.shape()[..last]
            || spec.layout != titan_types::Layout::Contiguous
            || spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail(
                "contract",
                "CUDA reduction.sum output shape must remove the final axis and be contiguous".into(),
            ));
        }
        let rows = numel(&output_shape);
        let cols = input.shape()[last];
        if rows == 0 || cols == 0 {
            return Err(fail("contract", "CUDA reduction.sum requires non-zero dimensions".into()));
        }
        let rows = i32::try_from(rows).map_err(|_| fail("contract", "CUDA reduction.sum row count exceeds i32 ABI".into()))?;
        let cols =
            i32::try_from(cols).map_err(|_| fail("contract", "CUDA reduction.sum axis length exceeds i32 ABI".into()))?;
        let session =
            input.session().ok_or_else(|| fail("contract", "CUDA reduction.sum input has no backend storage".into()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape)
            .map_err(|e| hal_execution_error(&operator, &source, "allocate", e.to_string()))?;
        let kernel_id = KernelId("reduction.sum.f32".into());
        let compiled = self
            .cached_cuda_reduction_sum_artifact(session.fingerprint(), &kernel_id)
            .map_err(|m| hal_execution_error(&operator, &source, "compile", m))?;
        let args = compiled
            .abi
            .encode(&vec![
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: rows.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: cols.to_le_bytes().to_vec() },
            ])
            .map_err(|e| hal_execution_error(&operator, &source, "abi", e.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|e| hal_execution_error(&operator, &source, "load", e.to_string()))?;
        let stream = session.create_stream().map_err(|e| hal_execution_error(&operator, &source, "stream", e.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [(rows as usize).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|e| hal_execution_error(&operator, &source, "launch", e.to_string()))?;
        session.wait(event.as_ref()).map_err(|e| hal_execution_error(&operator, &source, "event", e.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_concat(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 2 || request.outputs.len() != 1 || request.attrs.len() != 1 {
            return Err(fail("contract", "CUDA concat requires exactly two inputs, one output, and axis=0".into()));
        }
        let axis = int(&request.attrs, "axis", None).map_err(|message| fail("contract", message))?;
        if axis != 0 {
            return Err(fail("contract", "CUDA concat only supports rank-2 axis=0".into()));
        }
        let lhs = &request.inputs[0];
        let rhs = &request.inputs[1];
        let spec = &request.outputs[0];
        if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA concat requires F32 inputs and output".into()));
        }
        if lhs.device().backend != BackendId::Cuda
            || rhs.device().backend != BackendId::Cuda
            || lhs.shape().len() != 2
            || rhs.shape().len() != 2
            || !is_contiguous(lhs.shape(), lhs.strides())
            || !is_contiguous(rhs.shape(), rhs.strides())
        {
            return Err(fail("contract", "CUDA concat requires contiguous CUDA F32 rank-2 inputs".into()));
        }
        if lhs.shape()[1] != rhs.shape()[1] {
            return Err(fail("contract", "CUDA concat requires matching non-axis dimensions".into()));
        }
        let output_shape = vec![lhs.shape()[0] + rhs.shape()[0], lhs.shape()[1]];
        if spec.shape.0 != output_shape.iter().map(|dimension| *dimension as u64).collect::<Vec<_>>()
            || spec.layout != titan_types::Layout::Contiguous
            || spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA concat output shape and layout must be contiguous and exact".into()));
        }
        let session = lhs.session().ok_or_else(|| fail("contract", "CUDA concat lhs has no backend storage".into()))?;
        let rhs_session = rhs.session().ok_or_else(|| fail("contract", "CUDA concat rhs has no backend storage".into()))?;
        if !std::sync::Arc::ptr_eq(session, rhs_session) {
            return Err(fail("contract", "CUDA concat requires inputs from the same session".into()));
        }
        let lhs_elements = i32::try_from(numel(lhs.shape()))
            .map_err(|_| fail("contract", "CUDA concat lhs element count exceeds i32 ABI".into()))?;
        let total_elements = i32::try_from(numel(&output_shape))
            .map_err(|_| fail("contract", "CUDA concat output element count exceeds i32 ABI".into()))?;
        if lhs_elements == 0 || total_elements == 0 {
            return Err(fail("contract", "CUDA concat requires non-zero dimensions".into()));
        }
        let output = TensorHandle::allocate_f32(session.clone(), output_shape)
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("concat.f32".into());
        let compiled = self
            .cached_cuda_concat_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let args = compiled
            .abi
            .encode(&vec![
                KernelArg::Buffer { slot: 0, dtype: DType::F32, writable: false, alignment: 4, buffer: lhs.buffer().unwrap() },
                KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: rhs.buffer().unwrap() },
                KernelArg::Buffer {
                    slot: 2,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: lhs_elements.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: total_elements.to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [(total_elements as usize).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_slice(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA slice requires one input and output".into()));
        }
        let input = &request.inputs[0];
        let output_spec = &request.outputs[0];
        if input.device().backend != BackendId::Cuda
            || input.dtype() != DType::F32
            || output_spec.dtype != DType::F32
            || input.shape().len() != 1
            || !is_contiguous(input.shape(), input.strides())
        {
            return Err(fail("contract", "CUDA slice requires contiguous rank-1 F32 input".into()));
        }
        let axes = ints(&request.attrs, "axes").map_err(|m| fail("contract", m))?;
        let starts = ints(&request.attrs, "starts").map_err(|m| fail("contract", m))?;
        let stops = ints(&request.attrs, "stops").map_err(|m| fail("contract", m))?;
        let steps = ints(&request.attrs, "steps").map_err(|m| fail("contract", m))?;
        if axes != [0] || starts.len() != 1 || stops.len() != 1 || steps != [1] {
            return Err(fail("contract", "CUDA slice supports axis=0 and step=1 only".into()));
        }
        let start = starts[0];
        let stop = stops[0];
        if stop < start || stop > input.shape()[0] {
            return Err(fail("contract", "CUDA slice bounds are invalid".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|d| usize::try_from(*d))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "output shape exceeds usize".into()))?;
        if output_shape != [stop - start] || output_spec.layout != titan_types::Layout::Contiguous {
            return Err(fail("contract", "CUDA slice output shape/layout mismatch".into()));
        }
        let session = input.session().ok_or_else(|| fail("contract", "missing CUDA session".into()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape.clone())
            .map_err(|e| hal_execution_error(&operator, &source, "allocate", e.to_string()))?;
        let kernel_id = KernelId("slice.f32".into());
        let compiled = self
            .cached_cuda_slice_artifact(session.fingerprint(), &kernel_id)
            .map_err(|m| hal_execution_error(&operator, &source, "compile", m))?;
        let args = compiled
            .abi
            .encode(&[
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: (start as i32).to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: 1i32.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: ((stop - start) as i32).to_le_bytes().to_vec() },
            ])
            .map_err(|e| hal_execution_error(&operator, &source, "abi", e.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|e| hal_execution_error(&operator, &source, "load", e.to_string()))?;
        let stream = session.create_stream().map_err(|e| hal_execution_error(&operator, &source, "stream", e.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [((stop - start).div_ceil(block as usize)) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|e| hal_execution_error(&operator, &source, "launch", e.to_string()))?;
        session.wait(event.as_ref()).map_err(|e| hal_execution_error(&operator, &source, "event", e.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_transpose(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA transpose requires exactly one input and one output".into()));
        }
        if request.attrs.keys().any(|key| key != "permutation") {
            return Err(fail("contract", "CUDA transpose only accepts a permutation attribute".into()));
        }
        let input = &request.inputs[0];
        let spec = &request.outputs[0];
        if input.dtype() != DType::F32 || spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA transpose requires F32 input and output".into()));
        }
        if input.shape().len() != 2 || !is_contiguous(input.shape(), input.strides()) {
            return Err(fail("contract", "CUDA transpose requires a contiguous rank-2 input".into()));
        }
        let permutation = ints(&request.attrs, "permutation").map_err(|message| fail("contract", message))?;
        if permutation != [1, 0] {
            return Err(fail("contract", "CUDA transpose only supports permutation=[1, 0]".into()));
        }
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let output_shape = spec
            .shape
            .0
            .iter()
            .map(|d| usize::try_from(*d))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA transpose output shape exceeds host usize".into()))?;
        if output_shape != [cols, rows]
            || spec.layout != titan_types::Layout::Contiguous
            || spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA transpose output shape must be [cols, rows] and contiguous".into()));
        }
        if rows == 0 || cols == 0 {
            return Err(fail("contract", "CUDA transpose requires non-zero dimensions".into()));
        }
        let rows = i32::try_from(rows).map_err(|_| fail("contract", "CUDA transpose row count exceeds i32 ABI".into()))?;
        let cols = i32::try_from(cols).map_err(|_| fail("contract", "CUDA transpose column count exceeds i32 ABI".into()))?;
        let session = input.session().ok_or_else(|| fail("contract", "CUDA transpose input has no backend storage".into()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape)
            .map_err(|e| hal_execution_error(&operator, &source, "allocate", e.to_string()))?;
        let kernel_id = KernelId("transpose.f32".into());
        let compiled = self
            .cached_cuda_transpose_artifact(session.fingerprint(), &kernel_id)
            .map_err(|m| hal_execution_error(&operator, &source, "compile", m))?;
        let args = compiled
            .abi
            .encode(&vec![
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: rows.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: cols.to_le_bytes().to_vec() },
            ])
            .map_err(|e| hal_execution_error(&operator, &source, "abi", e.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|e| hal_execution_error(&operator, &source, "load", e.to_string()))?;
        let stream = session.create_stream().map_err(|e| hal_execution_error(&operator, &source, "stream", e.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [((rows as usize * cols as usize).div_ceil(block as usize)) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|e| hal_execution_error(&operator, &source, "launch", e.to_string()))?;
        session.wait(event.as_ref()).map_err(|e| hal_execution_error(&operator, &source, "event", e.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_resize_nearest2d(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 1 || request.outputs.len() != 1 || !request.attrs.is_empty() {
            return Err(fail("contract", "CUDA nearest resize requires one input/output and no attributes".into()));
        }
        let input = &request.inputs[0];
        let spec = &request.outputs[0];
        let shape = input.shape();
        if input.dtype() != DType::F32
            || spec.dtype != DType::F32
            || input.device().backend != BackendId::Cuda
            || shape.len() != 4
        {
            return Err(fail("contract", "CUDA nearest resize requires contiguous CUDA F32 rank-4 NCHW".into()));
        }
        if !is_contiguous(shape, input.strides()) || spec.layout != titan_types::Layout::Contiguous {
            return Err(fail("contract", "CUDA nearest resize requires contiguous tensors".into()));
        }
        let out_shape = spec
            .shape
            .0
            .iter()
            .map(|d| usize::try_from(*d))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "output shape overflow".into()))?;
        if out_shape.len() != 4
            || out_shape[0] != shape[0]
            || out_shape[1] != shape[1]
            || out_shape[2] == 0
            || out_shape[3] == 0
            || shape[2] == 0
            || shape[3] == 0
            || spec.strides.0 != contiguous_strides(&out_shape)
        {
            return Err(fail("contract", "invalid NCHW nearest resize output shape".into()));
        }
        let session = input.session().ok_or_else(|| fail("contract", "CUDA input has no storage".into()))?;
        let output =
            TensorHandle::allocate_f32(session.clone(), out_shape.clone()).map_err(|e| fail("allocate", e.to_string()))?;
        let dims = [shape[0], shape[1], shape[2], shape[3], out_shape[2], out_shape[3]].map(|v| i32::try_from(v).unwrap());
        let kid = KernelId("resize.nearest2d.f32".into());
        let compiled = self.cached_cuda_resize_artifact(session.fingerprint(), &kid).map_err(|m| fail("compile", m))?;
        let mut args = vec![
            KernelArg::Buffer { slot: 0, dtype: DType::F32, writable: false, alignment: 4, buffer: input.buffer().unwrap() },
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: true, alignment: 4, buffer: output.buffer().unwrap() },
        ];
        for value in dims {
            args.push(KernelArg::Scalar { dtype: DType::I32, bytes: value.to_le_bytes().to_vec() });
        }
        let encoded = compiled.abi.encode(&args).map_err(|e| fail("abi", e.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|e| fail("load", e.to_string()))?;
        let stream = session.create_stream().map_err(|e| fail("stream", e.to_string()))?;
        let count = numel(&out_shape);
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [count.div_ceil(block as usize).max(1) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event =
            session.launch(stream.as_ref(), kernel.as_ref(), &encoded, &geometry).map_err(|e| fail("launch", e.to_string()))?;
        session.wait(event.as_ref()).map_err(|e| fail("event", e.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kid })
    }

    fn execute_cuda_layer_norm(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if !(1..=3).contains(&request.inputs.len()) || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA LayerNorm requires input and optional gamma/beta plus one output".into()));
        }
        if request.attrs.keys().any(|key| key != "axis" && key != "epsilon") {
            return Err(fail("contract", "CUDA LayerNorm only accepts axis and epsilon attributes".into()));
        }
        let input = &request.inputs[0];
        let output_spec = &request.outputs[0];
        if input.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA LayerNorm requires F32 input and output".into()));
        }
        if input.device().backend != BackendId::Cuda || !is_contiguous(input.shape(), input.strides()) {
            return Err(fail("contract", "CUDA LayerNorm requires a contiguous CUDA input".into()));
        }
        if input.shape().is_empty() {
            return Err(fail("contract", "CUDA LayerNorm requires a non-scalar input".into()));
        }
        let axis = int(&request.attrs, "axis", Some(input.shape().len() - 1)).map_err(|message| fail("contract", message))?;
        if axis != input.shape().len() - 1 {
            return Err(fail("contract", "CUDA LayerNorm only supports the last axis".into()));
        }
        let epsilon = float_attr(&request.attrs, "epsilon", Some(1e-5)).map_err(|message| fail("contract", message))?;
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(fail("contract", "CUDA LayerNorm epsilon must be finite and non-negative".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|d| usize::try_from(*d))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA LayerNorm output shape exceeds host usize".into()))?;
        if output_shape != input.shape()
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA LayerNorm requires a contiguous output with the same shape as input".into()));
        }
        let cols = *input.shape().last().expect("non-scalar");
        let rows = numel(&input.shape()[..input.shape().len() - 1]);
        if rows == 0 || cols == 0 {
            return Err(fail("contract", "CUDA LayerNorm requires non-zero dimensions".into()));
        }
        let session = input.session().ok_or_else(|| fail("contract", "CUDA LayerNorm input has no backend storage".into()))?;
        for affine in request.inputs.iter().skip(1) {
            if affine.dtype() != DType::F32
                || affine.device() != input.device()
                || !is_contiguous(affine.shape(), affine.strides())
                || affine.shape() != [cols]
            {
                return Err(fail(
                    "contract",
                    "CUDA LayerNorm affine inputs must be contiguous CUDA F32 vectors matching the last axis".into(),
                ));
            }
            if affine.session().is_none() || !std::sync::Arc::ptr_eq(affine.session().expect("checked"), session) {
                return Err(fail("contract", "CUDA LayerNorm affine inputs must share the input session".into()));
            }
        }
        let output = TensorHandle::allocate_f32(session.clone(), output_shape.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let gamma = request.inputs.get(1).map(|x| x.buffer().unwrap());
        let beta = request.inputs.get(2).map(|x| x.buffer().unwrap());
        let has_gamma = gamma.is_some();
        let has_beta = beta.is_some();
        let kernel_id = KernelId("layer_norm.f32".into());
        let compiled = self
            .cached_cuda_layer_norm_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let args = compiled
            .abi
            .encode(&[
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().unwrap(),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: gamma.unwrap_or_else(|| input.buffer().unwrap()),
                },
                KernelArg::Buffer {
                    slot: 2,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: beta.unwrap_or_else(|| input.buffer().unwrap()),
                },
                KernelArg::Buffer {
                    slot: 3,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: (rows as i32).to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: (cols as i32).to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::F32, bytes: epsilon.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: (has_gamma as i32).to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: (has_beta as i32).to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let geometry = LaunchGeometry {
            grid: [u32::try_from(rows).unwrap().max(1), 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_group_norm(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if !(1..=3).contains(&request.inputs.len()) || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA GroupNorm requires input and optional gamma/beta plus one output".into()));
        }
        if request.attrs.keys().any(|key| key != "groups" && key != "epsilon") {
            return Err(fail("contract", "CUDA GroupNorm only accepts groups and epsilon attributes".into()));
        }
        let input = &request.inputs[0];
        let output_spec = &request.outputs[0];
        if input.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA GroupNorm requires F32 input and output".into()));
        }
        if input.device().backend != BackendId::Cuda
            || input.shape().len() != 4
            || !is_contiguous(input.shape(), input.strides())
        {
            return Err(fail("contract", "CUDA GroupNorm requires a contiguous rank-4 NCHW CUDA input".into()));
        }
        let dims: [usize; 4] = input.shape().try_into().expect("rank checked");
        let [n, channels, height, width] = dims;
        let groups = int(&request.attrs, "groups", None).map_err(|message| fail("contract", message))?;
        let groups = usize::try_from(groups).map_err(|_| fail("contract", "CUDA GroupNorm groups must be positive".into()))?;
        if n == 0 || channels == 0 || height == 0 || width == 0 || groups == 0 || channels % groups != 0 {
            return Err(fail("contract", "CUDA GroupNorm groups must be non-zero and divide non-zero channels".into()));
        }
        let epsilon = float_attr(&request.attrs, "epsilon", Some(1e-5)).map_err(|message| fail("contract", message))?;
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(fail("contract", "CUDA GroupNorm epsilon must be finite and non-negative".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|d| usize::try_from(*d))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA GroupNorm output shape exceeds host usize".into()))?;
        if output_shape != input.shape()
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail(
                "contract",
                "CUDA GroupNorm requires a contiguous output with the same NCHW shape as input".into(),
            ));
        }
        let session = input.session().ok_or_else(|| fail("contract", "CUDA GroupNorm input has no backend storage".into()))?;
        for affine in request.inputs.iter().skip(1) {
            if affine.dtype() != DType::F32
                || affine.device() != input.device()
                || !is_contiguous(affine.shape(), affine.strides())
                || affine.shape() != [channels]
                || affine.session().is_none()
                || !std::sync::Arc::ptr_eq(affine.session().expect("checked"), session)
            {
                return Err(fail(
                    "contract",
                    "CUDA GroupNorm affine inputs must be contiguous CUDA F32 channel vectors sharing the input session".into(),
                ));
            }
        }
        let output = TensorHandle::allocate_f32(session.clone(), output_shape)
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let gamma = request.inputs.get(1).map(|x| x.buffer().expect("tensor buffer"));
        let beta = request.inputs.get(2).map(|x| x.buffer().expect("tensor buffer"));
        let kernel_id = KernelId("group_norm.f32".into());
        let compiled = self
            .cached_cuda_group_norm_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let as_i32 = |value: usize, label| {
            i32::try_from(value).map_err(|_| fail("contract", format!("CUDA GroupNorm {label} exceeds i32 ABI")))
        };
        let args = compiled
            .abi
            .encode(&[
                KernelArg::Buffer {
                    slot: 0,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: input.buffer().expect("tensor buffer"),
                },
                KernelArg::Buffer {
                    slot: 1,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: gamma.clone().unwrap_or_else(|| input.buffer().expect("tensor buffer")),
                },
                KernelArg::Buffer {
                    slot: 2,
                    dtype: DType::F32,
                    writable: false,
                    alignment: 4,
                    buffer: beta.clone().unwrap_or_else(|| input.buffer().expect("tensor buffer")),
                },
                KernelArg::Buffer {
                    slot: 3,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().expect("tensor buffer"),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: as_i32(n, "N")?.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: as_i32(channels, "C")?.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: as_i32(height, "H")?.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: as_i32(width, "W")?.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: as_i32(groups, "groups")?.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::F32, bytes: epsilon.to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: (gamma.is_some() as i32).to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: (beta.is_some() as i32).to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let grid = n
            .checked_mul(groups)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| fail("contract", "CUDA GroupNorm N * groups exceeds launch grid".into()))?;
        let event = session
            .launch(
                stream.as_ref(),
                kernel.as_ref(),
                &args,
                &LaunchGeometry {
                    grid: [grid, 1, 1],
                    block: compiled.metadata.block,
                    shared_bytes: compiled.metadata.shared_bytes,
                },
            )
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_gemm(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 2 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA GEMM requires two inputs and one output".into()));
        }
        if request.attrs.keys().any(|key| key != "transpose_lhs" && key != "transpose_rhs") {
            return Err(fail("contract", "CUDA GEMM received unsupported attributes".into()));
        }
        let lhs = &request.inputs[0];
        let rhs = &request.inputs[1];
        let output_spec = &request.outputs[0];
        if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA GEMM requires F32 input and output tensors".into()));
        }
        if lhs.device().backend != BackendId::Cuda || rhs.device() != lhs.device() {
            return Err(fail("contract", "CUDA GEMM requires matching CUDA input devices".into()));
        }
        if lhs.shape().len() != 2
            || rhs.shape().len() != 2
            || !is_contiguous(lhs.shape(), lhs.strides())
            || !is_contiguous(rhs.shape(), rhs.strides())
        {
            return Err(fail("contract", "CUDA GEMM requires contiguous rank-2 inputs".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA GEMM output shape exceeds host usize".into()))?;
        if output_shape.len() != 2
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA GEMM requires a contiguous rank-2 output".into()));
        }
        let lhs_session = lhs.session().ok_or_else(|| fail("contract", "CUDA GEMM lhs has no backend storage".into()))?;
        let rhs_session = rhs.session().ok_or_else(|| fail("contract", "CUDA GEMM rhs has no backend storage".into()))?;
        if !std::sync::Arc::ptr_eq(lhs_session, rhs_session) {
            return Err(fail("contract", "CUDA GEMM requires inputs from the same session".into()));
        }
        let transpose_lhs = bool_attr(&request.attrs, "transpose_lhs", false).map_err(|message| fail("contract", message))?;
        let transpose_rhs = bool_attr(&request.attrs, "transpose_rhs", false).map_err(|message| fail("contract", message))?;
        let m = u32::try_from(lhs.shape()[0]).map_err(|_| fail("contract", "CUDA GEMM M exceeds u32 ABI".into()))?;
        let k = u32::try_from(lhs.shape()[1]).map_err(|_| fail("contract", "CUDA GEMM K exceeds u32 ABI".into()))?;
        let n = u32::try_from(rhs.shape()[1]).map_err(|_| fail("contract", "CUDA GEMM N exceeds u32 ABI".into()))?;
        let rhs_rows =
            u32::try_from(rhs.shape()[0]).map_err(|_| fail("contract", "CUDA GEMM rhs rows exceed u32 ABI".into()))?;
        let output_rows =
            u32::try_from(output_shape[0]).map_err(|_| fail("contract", "CUDA GEMM output rows exceed u32 ABI".into()))?;
        let output_columns =
            u32::try_from(output_shape[1]).map_err(|_| fail("contract", "CUDA GEMM output columns exceed u32 ABI".into()))?;
        GemmF32Descriptor {
            m,
            n,
            k,
            lhs_shape: [m, k],
            rhs_shape: [rhs_rows, n],
            output_shape: [output_rows, output_columns],
            lhs_dtype: lhs.dtype(),
            rhs_dtype: rhs.dtype(),
            output_dtype: output_spec.dtype,
            lhs_contiguous: is_contiguous(lhs.shape(), lhs.strides()),
            rhs_contiguous: is_contiguous(rhs.shape(), rhs.strides()),
            output_contiguous: true,
            transpose_lhs,
            transpose_rhs,
        }
        .validate()
        .map_err(|error| fail("contract", error.to_string()))?;
        let output = TensorHandle::allocate_f32(lhs_session.clone(), output_shape)
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("gemm.f32".into());
        let compiled = self
            .cached_cuda_gemm_artifact(lhs_session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let args = compiled
            .abi
            .encode(&[
                KernelArg::Buffer { slot: 0, dtype: DType::F32, writable: false, alignment: 4, buffer: lhs.buffer().unwrap() },
                KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: rhs.buffer().unwrap() },
                KernelArg::Buffer {
                    slot: 2,
                    dtype: DType::F32,
                    writable: true,
                    alignment: 4,
                    buffer: output.buffer().unwrap(),
                },
                KernelArg::Scalar { dtype: DType::I32, bytes: (m as i32).to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: (n as i32).to_le_bytes().to_vec() },
                KernelArg::Scalar { dtype: DType::I32, bytes: (k as i32).to_le_bytes().to_vec() },
            ])
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = lhs_session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream = lhs_session
            .create_stream()
            .map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [(m as usize * n as usize).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = lhs_session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        lhs_session
            .wait(event.as_ref())
            .map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_conv2d(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if !matches!(request.inputs.len(), 2..=3) || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA Conv2D requires input, weight, optional bias, and one output".into()));
        }
        if request.attrs.keys().any(|key| {
            !matches!(key.as_str(), "stride_h" | "stride_w" | "pad_h" | "pad_w" | "dilation_h" | "dilation_w" | "groups")
        }) {
            return Err(fail("contract", "CUDA Conv2D received unsupported attributes".into()));
        }
        let input = &request.inputs[0];
        let weight = &request.inputs[1];
        let bias = request.inputs.get(2);
        let output_spec = &request.outputs[0];
        if input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || bias.is_some_and(|tensor| tensor.dtype() != DType::F32)
            || output_spec.dtype != DType::F32
        {
            return Err(fail("contract", "CUDA Conv2D requires F32 input, weight, optional bias, and output".into()));
        }
        if input.device().backend != BackendId::Cuda
            || weight.device() != input.device()
            || bias.is_some_and(|tensor| tensor.device() != input.device())
        {
            return Err(fail("contract", "CUDA Conv2D requires matching CUDA input devices".into()));
        }
        if input.shape().len() != 4
            || weight.shape().len() != 4
            || bias.is_some_and(|tensor| tensor.shape().len() != 1)
            || !is_contiguous(input.shape(), input.strides())
            || !is_contiguous(weight.shape(), weight.strides())
            || bias.is_some_and(|tensor| !is_contiguous(tensor.shape(), tensor.strides()))
        {
            return Err(fail(
                "contract",
                "CUDA Conv2D requires contiguous NCHW input, OIHW weight, and optional rank-1 bias".into(),
            ));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA Conv2D output shape exceeds host usize".into()))?;
        if output_shape.len() != 4
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA Conv2D requires a contiguous rank-4 output".into()));
        }
        let session = input.session().ok_or_else(|| fail("contract", "CUDA Conv2D input has no backend storage".into()))?;
        if !std::sync::Arc::ptr_eq(
            session,
            weight.session().ok_or_else(|| fail("contract", "CUDA Conv2D weight has no backend storage".into()))?,
        ) || bias
            .is_some_and(|tensor| !std::sync::Arc::ptr_eq(session, tensor.session().expect("CUDA Conv2D bias session checked")))
        {
            return Err(fail("contract", "CUDA Conv2D requires inputs from the same session".into()));
        }
        let input_shape: [usize; 4] = input.shape().try_into().expect("rank checked");
        let weight_shape: [usize; 4] = weight.shape().try_into().expect("rank checked");
        let output_shape_array: [usize; 4] = output_shape.clone().try_into().expect("rank checked");
        let as_u32 = |shape: [usize; 4], label: &str| {
            shape
                .map(|dimension| u32::try_from(dimension))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map(|values| [values[0], values[1], values[2], values[3]])
                .map_err(|_| fail("contract", format!("CUDA Conv2D {label} exceeds u32 ABI")))
        };
        let stride_h = int(&request.attrs, "stride_h", None).map_err(|message| fail("contract", message))?;
        let stride_w = int(&request.attrs, "stride_w", None).map_err(|message| fail("contract", message))?;
        let pad_h = int(&request.attrs, "pad_h", None).map_err(|message| fail("contract", message))?;
        let pad_w = int(&request.attrs, "pad_w", None).map_err(|message| fail("contract", message))?;
        let dilation_h = int(&request.attrs, "dilation_h", None).map_err(|message| fail("contract", message))?;
        let dilation_w = int(&request.attrs, "dilation_w", None).map_err(|message| fail("contract", message))?;
        let groups = int(&request.attrs, "groups", None).map_err(|message| fail("contract", message))?;
        let descriptor = Conv2dF32Descriptor {
            input_shape: as_u32(input_shape, "input shape")?,
            weight_shape: as_u32(weight_shape, "weight shape")?,
            bias_shape: bias
                .map(|tensor| u32::try_from(tensor.shape()[0]).map(|length| [length]))
                .transpose()
                .map_err(|_| fail("contract", "CUDA Conv2D bias shape exceeds u32 ABI".into()))?,
            output_shape: as_u32(output_shape_array, "output shape")?,
            input_dtype: input.dtype(),
            weight_dtype: weight.dtype(),
            bias_dtype: bias.map(titan_tensor::TensorHandle::dtype),
            output_dtype: output_spec.dtype,
            input_contiguous: true,
            weight_contiguous: true,
            bias_contiguous: bias.map(|_| true),
            output_contiguous: true,
            stride_h: u32::try_from(stride_h).map_err(|_| fail("contract", "CUDA Conv2D stride_h exceeds u32 ABI".into()))?,
            stride_w: u32::try_from(stride_w).map_err(|_| fail("contract", "CUDA Conv2D stride_w exceeds u32 ABI".into()))?,
            pad_h: u32::try_from(pad_h).map_err(|_| fail("contract", "CUDA Conv2D pad_h exceeds u32 ABI".into()))?,
            pad_w: u32::try_from(pad_w).map_err(|_| fail("contract", "CUDA Conv2D pad_w exceeds u32 ABI".into()))?,
            dilation_h: u32::try_from(dilation_h)
                .map_err(|_| fail("contract", "CUDA Conv2D dilation_h exceeds u32 ABI".into()))?,
            dilation_w: u32::try_from(dilation_w)
                .map_err(|_| fail("contract", "CUDA Conv2D dilation_w exceeds u32 ABI".into()))?,
            groups: u32::try_from(groups).map_err(|_| fail("contract", "CUDA Conv2D groups exceeds u32 ABI".into()))?,
        };
        descriptor.validate().map_err(|error| fail("contract", error.to_string()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape)
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let bias_buffer = match bias {
            Some(tensor) => tensor.buffer().expect("CUDA Conv2D bias storage checked"),
            None => session
                .allocate(4, 4)
                .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?,
        };
        let kernel_id = KernelId("conv2d.f32".into());
        let compiled = self
            .cached_cuda_conv2d_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let mut arguments = vec![
            KernelArg::Buffer { slot: 0, dtype: DType::F32, writable: false, alignment: 4, buffer: input.buffer().unwrap() },
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: weight.buffer().unwrap() },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: bias_buffer },
            KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: true, alignment: 4, buffer: output.buffer().unwrap() },
        ];
        for scalar in [
            descriptor.input_shape[0],
            descriptor.input_shape[1],
            descriptor.input_shape[2],
            descriptor.input_shape[3],
            descriptor.weight_shape[0],
            descriptor.weight_shape[2],
            descriptor.weight_shape[3],
            descriptor.output_shape[2],
            descriptor.output_shape[3],
            descriptor.stride_h,
            descriptor.stride_w,
            descriptor.pad_h,
            descriptor.pad_w,
            descriptor.dilation_h,
            descriptor.dilation_w,
            descriptor.groups,
            u32::from(bias.is_some()),
        ] {
            arguments.push(KernelArg::Scalar { dtype: DType::I32, bytes: (scalar as i32).to_le_bytes().to_vec() });
        }
        let args = compiled
            .abi
            .encode(&arguments)
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [numel(&descriptor.output_shape.map(|dimension| dimension as usize)).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_attention(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 3 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA scaled dot-product attention requires exactly Q, K, V and one output".into()));
        }
        if !request.attrs.is_empty() {
            return Err(fail(
                "contract",
                "CUDA scaled dot-product attention does not implement mask, causal, or other attributes".into(),
            ));
        }
        let query = &request.inputs[0];
        let key = &request.inputs[1];
        let value = &request.inputs[2];
        let output_spec = &request.outputs[0];
        if query.dtype() != DType::F32
            || key.dtype() != DType::F32
            || value.dtype() != DType::F32
            || output_spec.dtype != DType::F32
        {
            return Err(fail("contract", "CUDA scaled dot-product attention requires F32 Q, K, V, and output".into()));
        }
        if query.device().backend != BackendId::Cuda || key.device() != query.device() || value.device() != query.device() {
            return Err(fail("contract", "CUDA scaled dot-product attention requires matching CUDA input devices".into()));
        }
        if query.shape().len() != 4
            || key.shape().len() != 4
            || value.shape().len() != 4
            || !is_contiguous(query.shape(), query.strides())
            || !is_contiguous(key.shape(), key.strides())
            || !is_contiguous(value.shape(), value.strides())
        {
            return Err(fail(
                "contract",
                "CUDA scaled dot-product attention requires contiguous rank-4 BHTD Q, K, and V".into(),
            ));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA attention output shape exceeds host usize".into()))?;
        if output_shape.len() != 4
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail("contract", "CUDA scaled dot-product attention requires a contiguous rank-4 output".into()));
        }
        let session = query.session().ok_or_else(|| fail("contract", "CUDA attention Q has no backend storage".into()))?;
        let key_session = key.session().ok_or_else(|| fail("contract", "CUDA attention K has no backend storage".into()))?;
        let value_session =
            value.session().ok_or_else(|| fail("contract", "CUDA attention V has no backend storage".into()))?;
        if !std::sync::Arc::ptr_eq(session, key_session) || !std::sync::Arc::ptr_eq(session, value_session) {
            return Err(fail(
                "contract",
                "CUDA scaled dot-product attention requires Q, K, and V from the same session".into(),
            ));
        }
        let shape_as_u32 = |shape: &[usize], label: &str| {
            shape
                .iter()
                .map(|dimension| u32::try_from(*dimension))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| fail("contract", format!("CUDA attention {label} exceeds u32 ABI")))
                .and_then(|dimensions| {
                    dimensions.try_into().map_err(|_| fail("contract", format!("CUDA attention {label} must be rank-4")))
                })
        };
        let descriptor = ScaledDotProductAttentionF32Descriptor {
            query_shape: shape_as_u32(query.shape(), "Q shape")?,
            key_shape: shape_as_u32(key.shape(), "K shape")?,
            value_shape: shape_as_u32(value.shape(), "V shape")?,
            output_shape: shape_as_u32(&output_shape, "output shape")?,
            query_dtype: query.dtype(),
            key_dtype: key.dtype(),
            value_dtype: value.dtype(),
            output_dtype: output_spec.dtype,
            query_contiguous: true,
            key_contiguous: true,
            value_contiguous: true,
            output_contiguous: true,
            has_mask: false,
            causal: false,
        };
        descriptor.validate().map_err(|error| fail("contract", error.to_string()))?;
        let output = TensorHandle::allocate_f32(session.clone(), output_shape)
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("scaled_dot_product_attention.f32".into());
        let compiled = self
            .cached_cuda_attention_artifact(session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let mut arguments = vec![
            KernelArg::Buffer { slot: 0, dtype: DType::F32, writable: false, alignment: 4, buffer: query.buffer().unwrap() },
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: key.buffer().unwrap() },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: value.buffer().unwrap() },
            KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: true, alignment: 4, buffer: output.buffer().unwrap() },
        ];
        for scalar in [
            descriptor.query_shape[0],
            descriptor.query_shape[1],
            descriptor.query_shape[2],
            descriptor.key_shape[2],
            descriptor.query_shape[3],
        ] {
            arguments.push(KernelArg::Scalar { dtype: DType::I32, bytes: (scalar as i32).to_le_bytes().to_vec() });
        }
        let args = compiled
            .abi
            .encode(&arguments)
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream =
            session.create_stream().map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [numel(&descriptor.output_shape.map(|dimension| dimension as usize)).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        session.wait(event.as_ref()).map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_cuda_broadcast_add(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail = |phase, message| ExecutionError { operator: operator.clone(), source: source.clone(), phase, message };
        if request.inputs.len() != 2 || request.outputs.len() != 1 {
            return Err(fail("contract", "CUDA broadcast add requires exactly two inputs and one output".into()));
        }
        if !request.attrs.is_empty() {
            return Err(fail("contract", "CUDA broadcast add does not accept attributes".into()));
        }
        let lhs = &request.inputs[0];
        let rhs = &request.inputs[1];
        let output_spec = &request.outputs[0];
        if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(fail("contract", "CUDA broadcast add requires F32 input and output tensors".into()));
        }
        if lhs.device().backend != BackendId::Cuda || rhs.device() != lhs.device() {
            return Err(fail("contract", "CUDA broadcast add requires matching CUDA input devices".into()));
        }
        if lhs.shape().len() != rhs.shape().len() {
            return Err(fail("contract", "CUDA broadcast add requires inputs with the same logical rank".into()));
        }
        if !(1..=4).contains(&lhs.shape().len()) {
            return Err(fail("contract", "CUDA broadcast add supports ranks one through four".into()));
        }
        if !is_contiguous(lhs.shape(), lhs.strides()) || !is_contiguous(rhs.shape(), rhs.strides()) {
            return Err(fail("contract", "CUDA broadcast add requires contiguous inputs".into()));
        }
        let output_shape = output_spec
            .shape
            .0
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| fail("contract", "CUDA broadcast add output shape exceeds host usize".into()))?;
        if output_shape.len() != lhs.shape().len()
            || output_spec.layout != titan_types::Layout::Contiguous
            || output_spec.strides.0 != contiguous_strides(&output_shape)
        {
            return Err(fail(
                "contract",
                "CUDA broadcast add requires a contiguous output with the same rank as the inputs".into(),
            ));
        }
        let lhs_session =
            lhs.session().ok_or_else(|| fail("contract", "CUDA broadcast add lhs has no backend storage".into()))?;
        let rhs_session =
            rhs.session().ok_or_else(|| fail("contract", "CUDA broadcast add rhs has no backend storage".into()))?;
        if !std::sync::Arc::ptr_eq(lhs_session, rhs_session) {
            return Err(fail("contract", "CUDA broadcast add requires inputs from the same session".into()));
        }
        let pad_shape = |shape: &[usize]| -> Result<[u32; 4], ExecutionError> {
            let mut padded = [1u32; 4];
            let offset = 4 - shape.len();
            for (index, dimension) in shape.iter().enumerate() {
                padded[offset + index] = u32::try_from(*dimension)
                    .map_err(|_| fail("contract", "CUDA broadcast add shape dimension exceeds u32 ABI".into()))?;
            }
            Ok(padded)
        };
        let descriptor = BroadcastAddF32Descriptor {
            lhs_shape: pad_shape(lhs.shape())?,
            rhs_shape: pad_shape(rhs.shape())?,
            output_shape: pad_shape(&output_shape)?,
            rank: lhs.shape().len() as u8,
            lhs_dtype: lhs.dtype(),
            rhs_dtype: rhs.dtype(),
            output_dtype: output_spec.dtype,
            lhs_contiguous: true,
            rhs_contiguous: true,
            output_contiguous: true,
        };
        descriptor.validate().map_err(|error| fail("contract", error.to_string()))?;
        let output = TensorHandle::allocate_f32(lhs_session.clone(), output_shape.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("broadcast.add.f32".into());
        let compiled = self
            .cached_cuda_broadcast_add_artifact(lhs_session.fingerprint(), &kernel_id)
            .map_err(|message| hal_execution_error(&operator, &source, "compile", message))?;
        let output_count = i32::try_from(numel(&output_shape))
            .map_err(|_| fail("contract", "CUDA broadcast add element count exceeds i32 ABI".into()))?;
        let mut arguments = vec![
            KernelArg::Buffer { slot: 0, dtype: DType::F32, writable: false, alignment: 4, buffer: lhs.buffer().unwrap() },
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: rhs.buffer().unwrap() },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: true, alignment: 4, buffer: output.buffer().unwrap() },
            KernelArg::Scalar { dtype: DType::I32, bytes: output_count.to_le_bytes().to_vec() },
        ];
        for scalar in descriptor.lhs_shape.into_iter().chain(descriptor.rhs_shape).chain(descriptor.output_shape) {
            arguments.push(KernelArg::Scalar { dtype: DType::I32, bytes: (scalar as i32).to_le_bytes().to_vec() });
        }
        let args = compiled
            .abi
            .encode(&arguments)
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = lhs_session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream = lhs_session
            .create_stream()
            .map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [numel(&output_shape).div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = lhs_session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        lhs_session
            .wait(event.as_ref())
            .map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        Ok(ExecutionHandle { outputs: vec![output], candidate: CandidateId("cuda/driver".into()), kernel: kernel_id })
    }

    fn execute_compiled_add(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        if request.inputs.is_empty() {
            return Err(ExecutionError { operator, source, phase: "validate", message: "operator requires inputs".into() });
        }
        if !matches!(request.operator.0.as_str(), "elementwise.add.f32" | "elementwise.add" | "elementwise.fused") {
            if self.schemas.get(&request.operator).is_none() {
                return Err(ExecutionError {
                    operator,
                    source,
                    phase: "schema",
                    message: "operator schema is not registered".into(),
                });
            }
            return Err(ExecutionError {
                operator,
                source,
                phase: "dispatch",
                message: "CPU generated dispatch is not yet executable".into(),
            });
        }
        if request.inputs.len() != 2 || request.outputs.len() != 1 {
            return Err(ExecutionError {
                operator,
                source,
                phase: "contract",
                message: "f32 add requires two inputs and one output".into(),
            });
        }
        let lhs = &request.inputs[0];
        let rhs = &request.inputs[1];
        let output_spec = &request.outputs[0];
        if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || output_spec.dtype != DType::F32 {
            return Err(ExecutionError { operator, source, phase: "contract", message: "f32 add requires F32 tensors".into() });
        }
        let output_shape: Vec<usize> = match output_spec.shape.0.iter().map(|x| usize::try_from(*x)).collect() {
            Ok(shape) => shape,
            Err(_) => {
                return Err(ExecutionError {
                    operator,
                    source,
                    phase: "contract",
                    message: "shape dimension exceeds host usize".into(),
                });
            }
        };
        if lhs.shape() != rhs.shape() || output_shape != lhs.shape() {
            return Err(ExecutionError {
                operator,
                source,
                phase: "contract",
                message: "elementwise add shapes must match".into(),
            });
        }
        if lhs.buffer().is_none() || rhs.buffer().is_none() {
            return Err(ExecutionError {
                operator,
                source,
                phase: "contract",
                message: "inputs must retain backend storage".into(),
            });
        }
        if !matches!(lhs.device().backend, BackendId::Cpu | BackendId::Cuda) || rhs.device() != lhs.device() {
            return Err(ExecutionError {
                operator,
                source,
                phase: "contract",
                message: "f32 add requires matching CPU or CUDA device".into(),
            });
        }
        let lhs_session = lhs.session().ok_or_else(|| ExecutionError {
            operator: operator.clone(),
            source: source.clone(),
            phase: "contract",
            message: "missing lhs session".into(),
        })?;
        let rhs_session = rhs.session().ok_or_else(|| ExecutionError {
            operator: operator.clone(),
            source: source.clone(),
            phase: "contract",
            message: "missing rhs session".into(),
        })?;
        if !std::sync::Arc::ptr_eq(lhs_session, rhs_session) {
            return Err(ExecutionError { operator, source, phase: "contract", message: "cross-session tensor inputs".into() });
        }
        let output = TensorHandle::allocate_f32(lhs_session.clone(), output_shape)
            .map_err(|error| hal_execution_error(&operator, &source, "allocate", error.to_string()))?;
        let kernel_id = KernelId("elementwise.add.f32".into());
        let compiled = self
            .cached_add_artifact(lhs_session.fingerprint(), &kernel_id)
            .map_err(|error| hal_execution_error(&operator, &source, "compile", error))?;
        let mut kernel_args = vec![
            KernelArg::Buffer { slot: 0, dtype: DType::F32, writable: false, alignment: 4, buffer: lhs.buffer().unwrap() },
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: rhs.buffer().unwrap() },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: true, alignment: 4, buffer: output.buffer().unwrap() },
        ];
        if lhs.device().backend == BackendId::Cuda {
            let count = i32::try_from(lhs.shape().iter().product::<usize>()).map_err(|_| ExecutionError {
                operator: operator.clone(),
                source: source.clone(),
                phase: "contract",
                message: "CUDA add element count exceeds i32 ABI".into(),
            })?;
            kernel_args.push(KernelArg::Scalar { dtype: DType::I32, bytes: count.to_le_bytes().to_vec() });
        }
        let args = compiled
            .abi
            .encode(&kernel_args)
            .map_err(|error| hal_execution_error(&operator, &source, "abi", error.to_string()))?;
        let kernel = lhs_session
            .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
            .map_err(|error| hal_execution_error(&operator, &source, "load", error.to_string()))?;
        let stream = lhs_session
            .create_stream()
            .map_err(|error| hal_execution_error(&operator, &source, "stream", error.to_string()))?;
        let count = lhs.shape().iter().product::<usize>();
        let block = compiled.metadata.block[0].max(1);
        let geometry = LaunchGeometry {
            grid: [count.div_ceil(block as usize) as u32, 1, 1],
            block: compiled.metadata.block,
            shared_bytes: compiled.metadata.shared_bytes,
        };
        let event = lhs_session
            .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
            .map_err(|error| hal_execution_error(&operator, &source, "launch", error.to_string()))?;
        lhs_session
            .wait(event.as_ref())
            .map_err(|error| hal_execution_error(&operator, &source, "event", error.to_string()))?;
        let candidate = match lhs.device().backend {
            BackendId::Cpu => CandidateId("cpu/reference".into()),
            BackendId::Cuda => CandidateId("cuda/driver".into()),
            _ => unreachable!(),
        };
        Ok(ExecutionHandle { outputs: vec![output], candidate, kernel: kernel_id })
    }

    fn execute_cpu_reference(&mut self, request: OpRequest) -> Result<ExecutionHandle, ExecutionError> {
        let operator = request.operator.clone();
        let source = request.source.clone();
        let fail =
            |message: String| ExecutionError { operator: operator.clone(), source: source.clone(), phase: "contract", message };
        if request.outputs.len() != 1 || request.inputs.is_empty() {
            return Err(fail("reference operator requires inputs and one output".into()));
        }
        if matches!(
            request.operator.0.as_str(),
            "reshape"
                | "transpose"
                | "slice"
                | "reduction.sum"
                | "softmax"
                | "silu"
                | "gelu"
                | "quick_gelu"
                | "resize.nearest2d"
                | "resize_nearest2d"
        ) && request.inputs.len() != 1
        {
            return Err(fail("reference unary operator requires exactly one input".into()));
        }
        if request.operator.0 == "broadcast.add" && request.inputs.len() != 2 {
            return Err(fail("broadcast add requires exactly two inputs".into()));
        }
        if request.operator.0 == "gemm" && request.inputs.len() != 2 {
            return Err(fail("gemm requires exactly two inputs".into()));
        }
        if request.operator.0 == "conv2d" && !matches!(request.inputs.len(), 2..=3) {
            return Err(fail("conv2d requires input, weight, and optional bias".into()));
        }
        if request.operator.0 == "scaled_dot_product_attention" && request.inputs.len() != 3 {
            return Err(fail("scaled dot-product attention requires exactly three inputs; masks are not implemented".into()));
        }
        if matches!(request.operator.0.as_str(), "layer_norm" | "layer.norm" | "group_norm" | "group.norm")
            && !matches!(request.inputs.len(), 1..=3)
        {
            return Err(fail("normalization requires input with optional weight and bias".into()));
        }
        let output_spec = &request.outputs[0];
        if output_spec.dtype != DType::F32 {
            return Err(fail("reference operators currently require F32 output".into()));
        }
        let session = request.inputs[0].session().ok_or_else(|| fail("inputs must retain backend storage".into()))?.clone();
        if session.device().backend != BackendId::Cpu {
            return Err(fail("reference operators require a CPU device".into()));
        }
        if request.inputs.iter().any(|input| {
            input.dtype() != DType::F32
                || input.session().is_none()
                || input.device() != session.device()
                || !is_contiguous(input.shape(), input.strides())
        }) {
            return Err(fail("reference operators require matching contiguous F32 inputs".into()));
        }
        if request.operator.0 == "scaled_dot_product_attention"
            && request.inputs.iter().any(|input| !std::sync::Arc::ptr_eq(input.session().expect("session checked"), &session))
        {
            return Err(fail("scaled dot-product attention requires inputs from the same session".into()));
        }
        let out_shape: Vec<usize> = output_spec
            .shape
            .0
            .iter()
            .map(|d| usize::try_from(*d))
            .collect::<Result<_, _>>()
            .map_err(|_| fail("shape dimension exceeds host usize".into()))?;
        if output_spec.layout != titan_types::Layout::Contiguous || output_spec.strides.0 != contiguous_strides(&out_shape) {
            return Err(fail("reference output must be contiguous".into()));
        }
        let values =
            request.inputs.iter().map(|x| x.to_vec_f32().map_err(|e| fail(e.to_string()))).collect::<Result<Vec<_>, _>>()?;
        let shapes = request.inputs.iter().map(|x| x.shape().to_vec()).collect::<Vec<_>>();
        let name = request.operator.0.as_str();
        let result = match name {
            "reshape" => {
                if values.len() != 1 || values[0].len() != numel(&out_shape) {
                    return Err(fail("reshape element count mismatch".into()));
                }
                values[0].clone()
            }
            "transpose" => transpose(&values[0], &shapes[0], &request.attrs, &out_shape).map_err(fail)?,
            "slice" => slice(&values[0], &shapes[0], &request.attrs, &out_shape).map_err(fail)?,
            "concat" => concat(&values, &shapes, &request.attrs, &out_shape).map_err(fail)?,
            "reduction.sum" => reduction_sum(&values[0], &shapes[0], &request.attrs, &out_shape).map_err(fail)?,
            "softmax" => softmax(&values[0], &shapes[0], &request.attrs, &out_shape).map_err(fail)?,
            "gemm" => gemm(&values, &shapes, &request.attrs, &out_shape).map_err(fail)?,
            "conv2d" => conv2d(&values, &shapes, &request.attrs, &out_shape).map_err(fail)?,
            "scaled_dot_product_attention" => {
                scaled_dot_product_attention(&values, &shapes, &request.attrs, &out_shape).map_err(fail)?
            }
            "broadcast.add" => broadcast_add(&values, &shapes, &out_shape).map_err(fail)?,
            "silu" => unary_same_shape(&values[0], &shapes[0], &out_shape, "silu", |value| value / (1.0 + (-value).exp()))
                .map_err(fail)?,
            "gelu" => unary_same_shape(&values[0], &shapes[0], &out_shape, "gelu", gelu).map_err(fail)?,
            "quick_gelu" => {
                let slope = quick_gelu_slope(&request.attrs).map_err(fail)?;
                unary_same_shape(&values[0], &shapes[0], &out_shape, "quick_gelu", |value| quick_gelu(value, slope))
                    .map_err(fail)?
            }
            "resize.nearest2d" | "resize_nearest2d" => resize_nearest2d(&values[0], &shapes[0], &out_shape).map_err(fail)?,
            "layer_norm" | "layer.norm" => layer_norm(&values, &shapes, &request.attrs, &out_shape).map_err(fail)?,
            "group_norm" | "group.norm" => group_norm(&values, &shapes, &request.attrs, &out_shape).map_err(fail)?,
            _ => {
                return Err(ExecutionError {
                    operator,
                    source,
                    phase: "dispatch",
                    message: "CPU reference operator is not implemented".into(),
                });
            }
        };
        let output = TensorHandle::from_f32_vec(session, out_shape, &result)
            .map_err(|e| hal_execution_error(&operator, &source, "allocate", e.to_string()))?;
        Ok(ExecutionHandle {
            outputs: vec![output],
            candidate: CandidateId("cpu/reference".into()),
            kernel: KernelId(operator.0.clone()),
        })
    }

    fn cached_add_artifact(&mut self, fingerprint: &DeviceFingerprint, kernel_id: &KernelId) -> Result<CachedArtifact, String> {
        let abi = match fingerprint.device.backend {
            BackendId::Cpu => titan_backend_cpu::elementwise_add_f32_abi(),
            BackendId::Cuda => cuda_add_abi(),
            backend => return Err(format!("add is not implemented for {backend:?}")),
        };
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let (bytes, metadata) = match fingerprint.device.backend {
            BackendId::Cpu => {
                let (bytes, compiled_abi) = compile_elementwise_add_f32(fingerprint).map_err(|error| error.to_string())?;
                let metadata = compiled_abi.launch_metadata(kernel_id).map_err(|error| error.to_string())?;
                (bytes, metadata)
            }
            BackendId::Cuda => {
                let compiler = CudaCompiler;
                let artifact = compiler
                    .compile_artifact(&cuda_add_ir(abi.clone()), &abi, fingerprint)
                    .map_err(|error| error.to_string())?;
                (artifact.ptx().to_vec(), artifact.metadata().clone())
            }
            _ => unreachable!(),
        };
        let artifact = CachedArtifact { bytes, abi, metadata };
        self.artifacts.insert(key, artifact.clone());
        Ok(artifact)
    }

    fn cached_cuda_gemm_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA GEMM artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_gemm_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact =
            CudaCompiler.compile_artifact(&cuda_gemm_ir(abi.clone()), &abi, fingerprint).map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_conv2d_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA Conv2D artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_conv2d_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_conv2d_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_attention_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA attention artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_attention_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_attention_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_broadcast_add_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA broadcast add artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_broadcast_add_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_broadcast_add_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_silu_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA SiLU artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_silu_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact =
            CudaCompiler.compile_artifact(&cuda_silu_ir(abi.clone()), &abi, fingerprint).map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_gelu_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA GELU artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_gelu_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact =
            CudaCompiler.compile_artifact(&cuda_gelu_ir(abi.clone()), &abi, fingerprint).map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_quick_gelu_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA QuickGELU artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_quick_gelu_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_quick_gelu_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_softmax_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA softmax artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_softmax_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_softmax_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_reduction_sum_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA reduction.sum artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_reduction_sum_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact =
            CudaCompiler.compile_artifact(&cuda_reduction_sum_ir(abi.clone()), &abi, fingerprint).map_err(|e| e.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_concat_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA concat artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_concat_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_concat_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_slice_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA slice artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_slice_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact =
            CudaCompiler.compile_artifact(&cuda_slice_ir(abi.clone()), &abi, fingerprint).map_err(|e| e.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_transpose_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA transpose artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_transpose_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_transpose_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_resize_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA resize artifact requested for non-CUDA session".into());
        }
        let abi = cuda_resize_nearest2d_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(a) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(a.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_resize_nearest2d_ir(abi.clone()), &abi, fingerprint)
            .map_err(|e| e.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_layer_norm_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA LayerNorm artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_layer_norm_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_layer_norm_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    fn cached_cuda_group_norm_artifact(
        &mut self,
        fingerprint: &DeviceFingerprint,
        kernel_id: &KernelId,
    ) -> Result<CachedArtifact, String> {
        if fingerprint.device.backend != BackendId::Cuda {
            return Err("CUDA GroupNorm artifact requested for a non-CUDA session".into());
        }
        let abi = cuda_group_norm_abi();
        let key = ArtifactCacheKey { kernel: kernel_id.clone(), abi_hash: abi.abi_hash(), device: fingerprint.clone() };
        if let Some(artifact) = self.artifacts.get(&key) {
            self.cache_hits += 1;
            return Ok(artifact.clone());
        }
        self.cache_misses += 1;
        let artifact = CudaCompiler
            .compile_artifact(&cuda_group_norm_ir(abi.clone()), &abi, fingerprint)
            .map_err(|error| error.to_string())?;
        let cached = CachedArtifact { bytes: artifact.ptx().to_vec(), abi, metadata: artifact.metadata().clone() };
        self.artifacts.insert(key, cached.clone());
        Ok(cached)
    }

    /// Returns `(cache hits, cache misses)` for compiled artifacts.
    pub fn artifact_cache_stats(&self) -> (u64, u64) {
        (self.cache_hits, self.cache_misses)
    }
    /// 编译 typed graph；每个节点保留同一 OpRequest 协议。
    pub fn compile(&mut self, graph: Graph, _options: CompileOptions) -> Result<ExecutionPlan, ExecutionError> {
        if graph.nodes.is_empty() {
            return Err(ExecutionError {
                operator: OperatorId("graph.empty".into()),
                source: SourceSpan { file: "<graph>".into(), line: 0, column: 0 },
                phase: "compile",
                message: "graph has no nodes".into(),
            });
        }
        Err(ExecutionError {
            operator: OperatorId("graph.compile".into()),
            source: SourceSpan { file: "<graph>".into(), line: 0, column: 0 },
            phase: "compile",
            message: "graph lowering requires runtime tensor bindings".into(),
        })
    }
    /// Returns the active runtime configuration.
    pub fn config(&self) -> RuntimeConfig {
        self.config
    }
    /// Returns collected telemetry.
    pub fn telemetry(&self) -> &Profiler {
        &self.profiler
    }
    /// Records a feedback observation. Candidate promotion is implemented by the v2 tuner store.
    pub fn record_autotune_feedback(&mut self, _candidate: &str, _observed: Duration, _incumbent: Duration) -> bool {
        false
    }
}
