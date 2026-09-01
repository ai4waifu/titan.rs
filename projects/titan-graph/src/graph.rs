//! Core Graph IR types, validation, semantic hash, and debug serialization.

use crate::diagnostic::{codes, IrDiagnostic};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

/// Current Core Graph IR schema version (bump only on breaking field changes).
pub const GRAPH_SCHEMA_VERSION: u32 = 1;

/// Stable graph value identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Stable graph node identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Tensor metadata carried by typed graph values.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Element dtype.
    pub dtype: DType,
    /// Static or partially static shape.
    pub shape: Shape,
    /// Element strides.
    pub strides: Strides,
    /// Physical layout.
    pub layout: Layout,
    /// Alias contract relative to producers/consumers.
    pub alias: AliasContract,
}

/// Node memory/effect contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct EffectContract {
    /// Memory effect class.
    pub memory: MemoryEffect,
    /// Whether the node is deterministic under fixed seeds/policy.
    pub deterministic: bool,
}

/// Equality / broadcast / matmul style constraint (minimal Core IR surface).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphConstraint {
    /// Two values must share the same dtype.
    SameDtype {
        /// Left-hand value.
        lhs: ValueId,
        /// Right-hand value.
        rhs: ValueId,
    },
    /// Two values must share identical shape vectors.
    SameShape {
        /// Left-hand value.
        lhs: ValueId,
        /// Right-hand value.
        rhs: ValueId,
    },
    /// Free-form debug constraint key (solver lands later).
    Custom {
        /// Constraint name.
        name: String,
        /// Opaque detail for diagnostics.
        detail: String,
    },
}

/// Typed graph node.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Stable node id.
    pub id: NodeId,
    /// Operator identity.
    pub operator: OperatorId,
    /// Input values.
    pub inputs: Vec<ValueId>,
    /// Output values.
    pub outputs: Vec<ValueId>,
    /// Canonical attributes.
    pub attrs: AttrMap,
    /// Effects.
    pub effects: EffectContract,
    /// Source span for diagnostics.
    pub source: SourceSpan,
}

/// Core Graph IR document (Living `16` minimum contract).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    /// Schema version (`GRAPH_SCHEMA_VERSION`).
    pub schema: u32,
    /// Graph inputs (subset of `values`).
    #[serde(default)]
    pub inputs: Vec<ValueId>,
    /// Graph outputs (subset of `values`).
    #[serde(default)]
    pub outputs: Vec<ValueId>,
    /// Value table.
    pub values: BTreeMap<ValueId, TensorSpec>,
    /// Nodes in topological construction order.
    pub nodes: Vec<GraphNode>,
    /// Explicit constraints.
    #[serde(default)]
    pub constraints: Vec<GraphConstraint>,
    /// Optional debug bag (not part of semantic hash).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub debug: BTreeMap<String, String>,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            schema: GRAPH_SCHEMA_VERSION,
            inputs: Vec::new(),
            outputs: Vec::new(),
            values: BTreeMap::new(),
            nodes: Vec::new(),
            constraints: Vec::new(),
            debug: BTreeMap::new(),
        }
    }
}

impl Graph {
    /// Creates an empty typed graph at the current schema version.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a value specification.
    pub fn add_value(&mut self, id: ValueId, spec: TensorSpec) -> Result<(), IrDiagnostic> {
        if self.values.insert(id, spec).is_some() {
            return Err(IrDiagnostic::error(codes::DUPLICATE_VALUE, format!("duplicate value id {}", id.0))
                .with_arg("value", id.0.to_string()));
        }
        Ok(())
    }

    /// Marks a value as a graph input.
    pub fn add_input(&mut self, id: ValueId) -> Result<(), IrDiagnostic> {
        if !self.values.contains_key(&id) {
            return Err(unknown_value(id));
        }
        if !self.inputs.contains(&id) {
            self.inputs.push(id);
        }
        Ok(())
    }

    /// Marks a value as a graph output.
    pub fn add_output(&mut self, id: ValueId) -> Result<(), IrDiagnostic> {
        if !self.values.contains_key(&id) {
            return Err(unknown_value(id));
        }
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
        Ok(())
    }

    /// Appends a constraint.
    pub fn add_constraint(&mut self, constraint: GraphConstraint) {
        self.constraints.push(constraint);
    }

    /// Appends a node after checking all references exist and node id is unique.
    pub fn add_node(&mut self, node: GraphNode) -> Result<(), IrDiagnostic> {
        if self.nodes.iter().any(|n| n.id == node.id) {
            return Err(IrDiagnostic::error(codes::DUPLICATE_NODE, format!("duplicate node id {}", node.id.0))
                .with_arg("node", node.id.0.to_string())
                .with_operation(node.operator.0.clone()));
        }
        for id in node.inputs.iter().chain(&node.outputs) {
            if !self.values.contains_key(id) {
                return Err(unknown_value(*id).with_operation(node.operator.0.clone()));
            }
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Validate closed references, schema, and minimal constraints.
    pub fn validate(&self) -> Result<(), IrDiagnostic> {
        if self.schema != GRAPH_SCHEMA_VERSION {
            return Err(IrDiagnostic::error(
                codes::SCHEMA_INVALID,
                format!("unsupported graph schema {} (expected {})", self.schema, GRAPH_SCHEMA_VERSION),
            )
            .with_arg("schema", self.schema.to_string())
            .with_arg("expected", GRAPH_SCHEMA_VERSION.to_string()));
        }
        if self.outputs.is_empty() {
            return Err(IrDiagnostic::error(codes::GRAPH_INVALID, "graph must declare at least one output")
                .with_detail("hint", "add_output"));
        }
        for id in self.inputs.iter().chain(&self.outputs) {
            if !self.values.contains_key(id) {
                return Err(unknown_value(*id));
            }
        }
        let mut seen_nodes = BTreeSet::new();
        for node in &self.nodes {
            if !seen_nodes.insert(node.id) {
                return Err(IrDiagnostic::error(codes::DUPLICATE_NODE, format!("duplicate node id {}", node.id.0))
                    .with_arg("node", node.id.0.to_string()));
            }
            for id in node.inputs.iter().chain(&node.outputs) {
                if !self.values.contains_key(id) {
                    return Err(unknown_value(*id).with_operation(node.operator.0.clone()));
                }
            }
        }
        for c in &self.constraints {
            match c {
                GraphConstraint::SameDtype { lhs, rhs } => {
                    let a = self.values.get(lhs).ok_or_else(|| unknown_value(*lhs))?;
                    let b = self.values.get(rhs).ok_or_else(|| unknown_value(*rhs))?;
                    if a.dtype != b.dtype {
                        return Err(IrDiagnostic::error(
                            codes::SHAPE_CONSTRAINT_UNSAT,
                            format!("dtype mismatch: {:?} vs {:?}", a.dtype, b.dtype),
                        )
                        .with_arg("lhs", lhs.0.to_string())
                        .with_arg("rhs", rhs.0.to_string())
                        .with_detail("constraint", "same_dtype"));
                    }
                }
                GraphConstraint::SameShape { lhs, rhs } => {
                    let a = self.values.get(lhs).ok_or_else(|| unknown_value(*lhs))?;
                    let b = self.values.get(rhs).ok_or_else(|| unknown_value(*rhs))?;
                    if a.shape != b.shape {
                        return Err(IrDiagnostic::error(
                            codes::SHAPE_CONSTRAINT_UNSAT,
                            format!("shape mismatch: {:?} vs {:?}", a.shape, b.shape),
                        )
                        .with_arg("lhs", lhs.0.to_string())
                        .with_arg("rhs", rhs.0.to_string())
                        .with_detail("constraint", "same_shape"));
                    }
                }
                GraphConstraint::Custom { .. } => {}
            }
        }
        Ok(())
    }

    /// Deterministic semantic hash over schema/values/nodes/constraints/I/O (excludes `debug`).
    pub fn semantic_hash(&self) -> String {
        let canonical = SemanticView {
            schema: self.schema,
            inputs: &self.inputs,
            outputs: &self.outputs,
            values: &self.values,
            nodes: &self.nodes,
            constraints: &self.constraints,
        };
        let bytes = serde_json::to_vec(&canonical).expect("semantic view serializes");
        format!("{:016x}", fnv1a64(&bytes))
    }

    /// Debug JSON serialization (includes debug bag).
    pub fn to_json(&self) -> Result<String, IrDiagnostic> {
        serde_json::to_string_pretty(self).map_err(|e| {
            IrDiagnostic::error(codes::SERIALIZE_UNSUPPORTED, format!("graph serialize failed: {e}"))
        })
    }

    /// Parse a debug JSON document.
    pub fn from_json(text: &str) -> Result<Self, IrDiagnostic> {
        serde_json::from_str(text).map_err(|e| {
            IrDiagnostic::error(codes::SERIALIZE_UNSUPPORTED, format!("graph deserialize failed: {e}"))
        })
    }

    /// Serialize → parse → validate roundtrip helper.
    pub fn roundtrip_json(&self) -> Result<Self, IrDiagnostic> {
        let text = self.to_json()?;
        let graph = Self::from_json(&text)?;
        graph.validate()?;
        Ok(graph)
    }
}

#[derive(Serialize)]
struct SemanticView<'a> {
    schema: u32,
    inputs: &'a [ValueId],
    outputs: &'a [ValueId],
    values: &'a BTreeMap<ValueId, TensorSpec>,
    nodes: &'a [GraphNode],
    constraints: &'a [GraphConstraint],
}

fn unknown_value(id: ValueId) -> IrDiagnostic {
    IrDiagnostic::error(codes::UNKNOWN_VALUE, format!("unknown value id {}", id.0)).with_arg("value", id.0.to_string())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Backend-independent operator request retained for runtime dispatch bridges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpRequest {
    /// Typed operator identity used by runtime dispatch.
    pub operator: OperatorId,
    /// Runtime tensor inputs.
    pub inputs: Vec<titan_tensor::TensorHandle>,
    /// Output allocation specifications.
    pub outputs: Vec<TensorSpec>,
    /// Canonical attributes.
    pub attrs: AttrMap,
    /// Memory and alias effects.
    pub effects: EffectContract,
    /// Diagnostic source location.
    pub source: SourceSpan,
}

/// Helper to build a contiguous F32 tensor spec.
pub fn f32_contiguous(shape: Vec<u64>) -> TensorSpec {
    let rank = shape.len();
    let mut strides = vec![1i64; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    TensorSpec {
        dtype: DType::F32,
        shape: Shape(shape),
        strides: Strides(strides),
        layout: Layout::Contiguous,
        alias: AliasContract::NoAlias,
    }
}

/// Empty attrs helper.
pub fn empty_attrs() -> AttrMap {
    BTreeMap::new()
}

/// Source span helper.
pub fn span(file: &str, line: u32, column: u32) -> SourceSpan {
    SourceSpan {
        file: file.into(),
        line,
        column,
    }
}

/// Integer attribute helper.
pub fn attr_int(v: i64) -> AttrValue {
    AttrValue::Int(v)
}
