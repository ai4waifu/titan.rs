#![warn(missing_docs)]
//! Compile-time-friendly graph representations and a lightweight executor.

use std::collections::BTreeMap;
use titan_tensor::TensorHandle;
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

/// Backend-independent operator request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpRequest {
    /// Typed operator identity used by runtime dispatch.
    pub operator: OperatorId,
    /// Runtime tensor inputs.
    pub inputs: Vec<TensorHandle>,
    /// Output allocation specifications.
    pub outputs: Vec<TensorSpec>,
    /// Canonical attributes.
    pub attrs: AttrMap,
    /// Memory and alias effects.
    pub effects: EffectContract,
    /// Diagnostic source location.
    pub source: SourceSpan,
}

/// Stable graph value identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueId(pub u32);
/// Stable graph node identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub u32);
/// Tensor metadata carried by typed graph values.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TensorSpec {
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
    pub layout: Layout,
    pub alias: AliasContract,
}
/// Node memory/effect contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EffectContract {
    pub memory: MemoryEffect,
    pub deterministic: bool,
}
/// Typed graph node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphNode {
    pub id: NodeId,
    pub operator: OperatorId,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attrs: AttrMap,
    pub effects: EffectContract,
    pub source: SourceSpan,
}
/// Typed graph IR.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Graph {
    pub values: BTreeMap<ValueId, TensorSpec>,
    pub nodes: Vec<GraphNode>,
    pub outputs: Vec<ValueId>,
}
impl Graph {
    /// Creates an empty typed graph.
    pub fn new() -> Self {
        Self::default()
    }
    /// Registers a value specification.
    pub fn add_value(&mut self, id: ValueId, spec: TensorSpec) -> Result<(), GraphError> {
        if self.values.insert(id, spec).is_some() {
            return Err(GraphError::DuplicateValue(id));
        }
        Ok(())
    }
    /// Appends a node after checking all references exist.
    pub fn add_node(&mut self, node: GraphNode) -> Result<(), GraphError> {
        if node.inputs.iter().chain(&node.outputs).any(|id| !self.values.contains_key(id)) {
            return Err(GraphError::UnknownValue);
        }
        self.nodes.push(node);
        Ok(())
    }
    /// Returns a deterministic structural hash input.
    pub fn semantic_key(&self) -> String {
        format!("{:?}", self)
    }
}
/// Graph construction failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphError {
    DuplicateValue(ValueId),
    UnknownValue,
}
