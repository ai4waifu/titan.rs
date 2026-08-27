#![warn(missing_docs)]
//! 算子 schema 与生成式 kernel 候选协议。

use std::fmt;
use titan_kernel::{AbiArg, KernelAbi, KernelRecipe, KernelTarget, LaunchConfig, StrategyRegistry};
use titan_types::{
    BackendId, DType, DeterminismPolicy, DeviceFingerprint, Layout, OperatorId, PrecisionPolicy, Shape, Strides,
    WorkspacePolicy,
};

/// 经 schema 验证后的算子规格。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpSpec {
    /// 算子标识。
    pub operator: OperatorId,
    /// 输入数据类型。
    pub inputs: Vec<TensorSpec>,
    /// 输出数据类型。
    pub outputs: Vec<TensorSpec>,
    /// 规范化属性字节。
    pub attrs: Vec<u8>,
    /// 精度策略。
    pub precision: PrecisionPolicy,
    /// 确定性策略。
    pub determinism: DeterminismPolicy,
    /// workspace 策略。
    pub workspace: WorkspacePolicy,
}

/// 算子输入/输出的完整 metadata。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    /// 数据类型。
    pub dtype: DType,
    /// 精确 shape。
    pub shape: Shape,
    /// 精确 stride。
    pub strides: Strides,
    /// 内存布局。
    pub layout: Layout,
}

/// 输出分配要求。
pub type OutputSpec = TensorSpec;

/// reference 执行上下文；不暴露 Tensor 或后端句柄。
pub trait ReferenceKernel: Send + Sync {
    /// 使用 host bytes 执行正确性参考计算。
    fn execute(&self, spec: &OpSpec, inputs: &[&[u8]], outputs: &mut [&mut [u8]]) -> Result<(), SchemaError>;
}

/// 算子策略契约。
pub trait OperatorSchema: Send + Sync {
    /// 返回稳定算子 id。
    fn operator(&self) -> &OperatorId;
    /// schema 版本。
    fn version(&self) -> u32;
    /// 校验并推导输出规格。
    fn validate(&self, spec: &OpSpec) -> Result<(), SchemaError>;
    /// 生成同设备 baseline 和调优候选。
    fn generate(
        &self,
        spec: &OpSpec,
        device: &DeviceFingerprint,
        max_candidates: usize,
    ) -> Result<Vec<KernelRecipe>, SchemaError>;
    /// 返回 CPU correctness oracle。
    fn reference(&self) -> &dyn ReferenceKernel;
    /// 返回 generated baseline ABI。
    fn baseline_abi(&self, spec: &OpSpec) -> Result<KernelAbi, SchemaError>;
    /// 判断设备能力是否满足 schema。
    fn supports(&self, device: &DeviceFingerprint) -> bool;
}

/// 稳定的首批 schema 注册表。
#[derive(Default)]
pub struct SchemaRegistry {
    entries: Vec<Box<dyn OperatorSchema>>,
}
impl SchemaRegistry {
    /// 创建空注册表。
    pub fn new() -> Self {
        Self::default()
    }
    /// 注册 schema，拒绝重复 operator id。
    pub fn register(&mut self, schema: Box<dyn OperatorSchema>) -> Result<(), SchemaError> {
        if self.entries.iter().any(|item| item.operator() == schema.operator()) {
            return Err(SchemaError::Duplicate(schema.operator().clone()));
        }
        self.entries.push(schema);
        Ok(())
    }
    /// 查找 schema。
    pub fn get(&self, id: &OperatorId) -> Option<&dyn OperatorSchema> {
        self.entries.iter().find(|item| item.operator() == id).map(|item| item.as_ref())
    }
}

/// schema 层错误。
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchemaError {
    Duplicate(OperatorId),
    Invalid(String),
    Unsupported(BackendId),
}
impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SchemaError {}

/// 返回首批 schema id，供支持矩阵生成器使用。
pub fn builtin_operator_ids() -> [OperatorId; 3] {
    [OperatorId("elementwise.fused".into()), OperatorId("reduction.sum".into()), OperatorId("matmul.f32".into())]
}

struct NoopReference;
impl ReferenceKernel for NoopReference {
    fn execute(&self, _spec: &OpSpec, _inputs: &[&[u8]], _outputs: &mut [&mut [u8]]) -> Result<(), SchemaError> {
        Err(SchemaError::Invalid("reference kernel not implemented for this schema".into()))
    }
}

struct BuiltinSchema {
    id: OperatorId,
    reference: NoopReference,
}
impl BuiltinSchema {
    fn new(name: &'static str) -> Self {
        Self { id: OperatorId(name.into()), reference: NoopReference }
    }
}
impl OperatorSchema for BuiltinSchema {
    fn operator(&self) -> &OperatorId {
        &self.id
    }
    fn version(&self) -> u32 {
        1
    }
    fn validate(&self, spec: &OpSpec) -> Result<(), SchemaError> {
        if spec.operator != self.id || spec.outputs.is_empty() {
            return Err(SchemaError::Invalid("operator/spec mismatch or missing output".into()));
        }
        if spec.inputs.iter().chain(&spec.outputs).any(|t| t.dtype != DType::F32) {
            return Err(SchemaError::Invalid("builtin baseline currently requires F32".into()));
        }
        Ok(())
    }
    fn generate(
        &self,
        spec: &OpSpec,
        device: &DeviceFingerprint,
        max_candidates: usize,
    ) -> Result<Vec<KernelRecipe>, SchemaError> {
        self.validate(spec)?;
        if device.device.backend != BackendId::Cpu {
            return Err(SchemaError::Unsupported(device.device.backend));
        }
        Ok(StrategyRegistry::default().generate(BackendId::Cpu, KernelTarget::CpuAvx2, max_candidates.max(1)))
    }
    fn reference(&self) -> &dyn ReferenceKernel {
        &self.reference
    }
    fn baseline_abi(&self, spec: &OpSpec) -> Result<KernelAbi, SchemaError> {
        self.validate(spec)?;
        Ok(KernelAbi {
            version: 1,
            args: vec![AbiArg::Buffer { dtype: DType::F32, writable: false, alignment: 4 }; spec.inputs.len()],
            launch: LaunchConfig::default(),
            workspace_bytes: 0,
        })
    }
    fn supports(&self, device: &DeviceFingerprint) -> bool {
        device.device.backend == BackendId::Cpu
    }
}

/// 构造首批 schema 注册表；所有条目都带有 generated baseline 协议。
pub fn builtin_registry() -> SchemaRegistry {
    let mut registry = SchemaRegistry::new();
    for name in ["elementwise.fused", "reduction.sum", "matmul.f32"] {
        registry.register(Box::new(BuiltinSchema::new(name))).expect("builtin ids are unique");
    }
    registry
}
