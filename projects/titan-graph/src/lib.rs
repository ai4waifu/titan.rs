#![warn(missing_docs)]
//! Titan Graph IR: Core graph contract, pass registry skeleton, and Living `15` diagnostics.
//!
//! This crate owns compile IR schema evolution. DXO must not fork a parallel `DxoGraphIR`.

mod diagnostic;
mod graph;
mod pass;

pub use diagnostic::{codes, IrDiagnostic, Severity};
pub use graph::{
    attr_int, empty_attrs, f32_contiguous, span, EffectContract, GRAPH_SCHEMA_VERSION, Graph, GraphConstraint,
    GraphNode, NodeId, OpRequest, TensorSpec, ValueId,
};
pub use pass::{builtin_pass_registry, PassDecl, PassFailureBehavior, PassRegistry, PassStage};
