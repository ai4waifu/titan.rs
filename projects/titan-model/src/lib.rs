#![warn(missing_docs)]
//! User-facing model definitions and portable deployment metadata.

use serde::{Deserialize, Serialize};
use titan_hal::Backend;
use titan_runtime::Runtime;
use titan_tensor::{Tensor, TensorError};

/// Version shared by all read-only model and run HTTP representations.
pub const API_SCHEMA_VERSION: u32 = 1;

/// A model that accepts and returns a rank-two f32 tensor on one backend.
pub trait Module<B: Backend> {
    fn forward(&self, input: &Tensor<B, f32, 2>, runtime: &mut Runtime, backend: B) -> Result<Tensor<B, f32, 2>, TensorError>;
}

/// Stable identifier for a model family (for example `language.transformer`).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Hash, Serialize)]
pub struct ModelFamilyId(pub String);

/// Stable identifier for a concrete model variant and schema version.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Hash, Serialize)]
pub struct ModelVariantId(pub String);

/// Capabilities advertised by a model implementation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelCapabilities {
    /// Whether training is supported.
    pub training: bool,
    /// Whether generation is supported.
    pub generation: bool,
    /// Whether streaming is supported.
    pub streaming: bool,
    /// Whether native deployment is supported.
    pub native: bool,
    /// Whether lightweight WASM deployment is supported.
    pub wasm: bool,
}

/// Versioned input/output contract for a model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelSchema {
    /// Input schema identifier.
    pub input: String,
    /// Output schema identifier.
    pub output: String,
    /// Schema version.
    pub version: u32,
}

/// Metadata returned by an explicit model registry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelDescriptor {
    /// Stable family identifier.
    pub family: ModelFamilyId,
    /// Stable variant identifier.
    pub variant: ModelVariantId,
    /// Execution capabilities.
    pub capabilities: ModelCapabilities,
    /// Versioned I/O contract.
    pub schema: ModelSchema,
}

/// A loader boundary owned by a concrete model crate.
pub trait ModelLoader {
    /// Describes this loader.
    fn descriptor(&self) -> &ModelDescriptor;
    /// Validates an external model manifest.
    fn load_manifest(&self, manifest: &str) -> Result<ModelDescriptor, ModelError>;
}

/// Explicit, side-effect-free model registry.
#[derive(Default)]
pub struct ModelRegistry {
    entries: Vec<ModelDescriptor>,
}

impl ModelRegistry {
    /// Registers one descriptor, rejecting duplicate family/variant pairs.
    pub fn register(&mut self, descriptor: ModelDescriptor) -> Result<(), ModelError> {
        if self.entries.iter().any(|item| item.family == descriptor.family && item.variant == descriptor.variant) {
            return Err(ModelError::Duplicate(descriptor.family.0, descriptor.variant.0));
        }
        self.entries.push(descriptor);
        Ok(())
    }

    /// Finds a descriptor by stable identifiers.
    pub fn find(&self, family: &ModelFamilyId, variant: &ModelVariantId) -> Option<&ModelDescriptor> {
        self.entries.iter().find(|item| &item.family == family && &item.variant == variant)
    }

    /// Returns all registered descriptors in registration order.
    pub fn descriptors(&self) -> &[ModelDescriptor] {
        &self.entries
    }
}

/// Errors at the model contract boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    Duplicate(String, String),
    InvalidManifest(String),
}

/// A linear layer whose dimensions are encoded in its type.
pub struct Linear<const INPUT: usize, const OUTPUT: usize, B: Backend> {
    weights: Tensor<B, f32, 2>,
}
impl<const INPUT: usize, const OUTPUT: usize, B: Backend> Linear<INPUT, OUTPUT, B> {
    /// Creates a linear layer from exactly INPUT * OUTPUT weights.
    pub fn from_weights(backend: B, weights: Vec<f32>) -> Result<Self, TensorError> {
        Ok(Self { weights: Tensor::from_vec(backend, [INPUT, OUTPUT], weights)? })
    }
}
impl<const INPUT: usize, const OUTPUT: usize, B: Backend> Module<B> for Linear<INPUT, OUTPUT, B> {
    fn forward(&self, input: &Tensor<B, f32, 2>, runtime: &mut Runtime, backend: B) -> Result<Tensor<B, f32, 2>, TensorError> {
        runtime.matmul(input, &self.weights, backend)
    }
}

/// Runtime targets supported by the single-binary deployment contract.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentTarget {
    /// Native CPU/GPU runtime deployment.
    Native,
    /// Lightweight browser or embedded validation deployment.
    Wasm,
}

/// Validation state of a model package manifest.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestState {
    /// The manifest has been checked and may be loaded by a compatible runtime.
    Ready,
    /// A declared model package has no manifest available to inspect.
    Missing,
    /// The manifest was present but did not satisfy its contract.
    Invalid,
}

/// Manifest metadata that can safely be exposed to a read-only client.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelManifestSummary {
    /// Version of the package manifest schema.
    pub schema_version: u32,
    /// Validation outcome, without leaking paths or checksum data.
    pub state: ManifestState,
    /// Targets advertised by the checked manifest.
    pub deployment_targets: Vec<DeploymentTarget>,
}

/// A model directory item returned by `GET /api/models`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelCatalogEntry {
    /// Stable family identifier.
    pub family: ModelFamilyId,
    /// Stable variant identifier.
    pub variant: ModelVariantId,
    /// Versioned input/output schema.
    pub schema: ModelSchema,
    /// Execution capabilities.
    pub capabilities: ModelCapabilities,
    /// Checked package manifest metadata.
    pub manifest: ModelManifestSummary,
    /// A non-sensitive operational summary suitable for a directory view.
    pub diagnostic_summary: String,
}

/// Health derived from runtime observations, not a command to control a run.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunHealth {
    /// Latest liveness and readiness checks passed.
    Healthy,
    /// The run remains available but has a reported warning.
    Degraded,
    /// The run cannot make progress.
    Failed,
    /// No health information has been reported yet.
    Unknown,
}

/// Read-only run state returned by `GET /api/runs/{id}`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunStatus {
    /// Caller-visible run identifier.
    pub run_id: String,
    /// Model selected by this run.
    pub model: ModelCatalogEntry,
    /// Last completed step, if reported.
    pub step: Option<u64>,
    /// Rank responsible for this report, if distributed execution is in use.
    pub rank: Option<u32>,
    /// Runtime graph revision, if known.
    pub graph_version: Option<String>,
    /// Aggregated operational health.
    pub health: RunHealth,
    /// Non-sensitive reason for a non-healthy state.
    pub health_summary: String,
}

/// Envelope returned by read-only API endpoints.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiResponse<T> {
    /// Compatibility version for this payload.
    pub schema_version: u32,
    /// Server-assigned request correlation id.
    pub request_id: String,
    /// RFC 3339 generation time supplied by the API server.
    pub generated_at: String,
    /// Endpoint-specific read-only data.
    pub data: T,
}

/// Stable codes for client-actionable API failures.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorCode {
    /// The client and server cannot agree on the schema version.
    SchemaUnsupported,
    /// The requested model identifier is not present in the directory.
    ModelNotFound,
    /// The requested run identifier is not visible or does not exist.
    RunNotFound,
    /// A manifest failed validation before runtime work could begin.
    ManifestInvalid,
    /// The request was syntactically invalid.
    InvalidRequest,
    /// An unexpected server failure occurred.
    Internal,
}

/// Structured API error returned in place of a success envelope.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiError {
    /// Compatibility version for this error representation.
    pub schema_version: u32,
    /// Server-assigned request correlation id.
    pub request_id: String,
    /// Stable machine-readable category.
    pub code: ApiErrorCode,
    /// Safe human-readable explanation.
    pub message: String,
    /// Whether retrying the same read request can be useful.
    pub retryable: bool,
}

/// JSON Schema documents for the public read-only HTTP payloads.
pub struct ApiJsonSchema;

impl ApiJsonSchema {
    /// Returns Draft 2020-12 schemas keyed by payload name.
    pub fn documents() -> serde_json::Value {
        serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "schema_version": API_SCHEMA_VERSION,
            "models_response": {
                "type": "object",
                "required": ["schema_version", "request_id", "generated_at", "data"],
                "properties": {"schema_version": {"const": API_SCHEMA_VERSION}, "request_id": {"type": "string"}, "generated_at": {"type": "string", "format": "date-time"}, "data": {"type": "array", "items": {"$ref": "#/$defs/model_catalog_entry"}}}
            },
            "run_response": {
                "type": "object",
                "required": ["schema_version", "request_id", "generated_at", "data"],
                "properties": {"schema_version": {"const": API_SCHEMA_VERSION}, "request_id": {"type": "string"}, "generated_at": {"type": "string", "format": "date-time"}, "data": {"$ref": "#/$defs/run_status"}}
            },
            "error": {"type": "object", "required": ["schema_version", "request_id", "code", "message", "retryable"]},
            "$defs": {
                "model_catalog_entry": {"type": "object", "required": ["family", "variant", "schema", "capabilities", "manifest", "diagnostic_summary"]},
                "run_status": {"type": "object", "required": ["run_id", "model", "step", "rank", "graph_version", "health", "health_summary"]}
            }
        })
    }
}

/// Metadata emitted alongside a deployable model binary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeploymentManifest {
    pub model: String,
    pub target: DeploymentTarget,
    pub backend: String,
}
impl DeploymentManifest {
    /// Serializes a small, dependency-free deployment manifest.
    pub fn encode(&self) -> String {
        format!("model={}\ntarget={:?}\nbackend={}", self.model, self.target, self.backend)
    }
}

/// A deliberately small ONNX interchange envelope for the supported core path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OnnxModel {
    pub opset: u32,
    pub graph_name: String,
    pub operators: Vec<String>,
}
impl OnnxModel {
    /// Exports a linear model graph to the interoperable envelope.
    pub fn linear(name: impl Into<String>) -> Self {
        Self { opset: 18, graph_name: name.into(), operators: vec!["MatMul".into()] }
    }

    /// Encodes the envelope in a deterministic text representation.
    pub fn encode(&self) -> String {
        format!("opset={}\ngraph={}\nops={}", self.opset, self.graph_name, self.operators.join(","))
    }
}
