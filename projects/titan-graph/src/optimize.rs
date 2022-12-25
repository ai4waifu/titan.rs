//! Core pass pipeline skeleton: validate → DCE → elementwise fusion (report-first).

use crate::diagnostic::{codes, IrDiagnostic};
use crate::graph::{Graph, NodeId, ValueId};
use crate::pass::{PassStage, builtin_pass_registry};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Auditable record of one pass invocation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PassReport {
    /// Pass name.
    pub name: String,
    /// Stage.
    pub stage: PassStage,
    /// Whether the graph changed.
    pub changed: bool,
    /// English notes (not localized).
    pub notes: Vec<String>,
}

/// Run the Core IR pipeline required by `titan-ir-cpu-closed-loop`.
pub fn run_core_pipeline(mut graph: Graph) -> Result<(Graph, Vec<PassReport>), IrDiagnostic> {
    let registry = builtin_pass_registry();
    let mut reports = Vec::new();

    graph.validate()?;
    reports.push(PassReport {
        name: "validate".into(),
        stage: PassStage::Core,
        changed: false,
        notes: vec!["schema and constraints ok".into()],
    });
    let _ = registry.get("validate");

    let before = graph.semantic_hash();
    graph = dead_code_eliminate(graph);
    let after_dce = graph.semantic_hash();
    reports.push(PassReport {
        name: "dce".into(),
        stage: PassStage::Core,
        changed: before != after_dce,
        notes: vec!["removed unreachable pure nodes".into()],
    });

    let before_fusion = graph.semantic_hash();
    let (fused, fusion_notes) = elementwise_fusion(graph)?;
    graph = fused;
    let after_fusion = graph.semantic_hash();
    reports.push(PassReport {
        name: "fusion.elementwise".into(),
        stage: PassStage::Core,
        changed: before_fusion != after_fusion,
        notes: fusion_notes,
    });

    graph.validate()?;
    Ok((graph, reports))
}

fn dead_code_eliminate(mut graph: Graph) -> Graph {
    let mut live_values: BTreeSet<ValueId> = graph.outputs.iter().copied().collect();
    let mut changed = true;
    while changed {
        changed = false;
        for node in graph.nodes.iter().rev() {
            if node.outputs.iter().any(|o| live_values.contains(o)) {
                for i in &node.inputs {
                    if live_values.insert(*i) {
                        changed = true;
                    }
                }
            }
        }
        for input in &graph.inputs {
            live_values.insert(*input);
        }
    }
    let dead_outputs: Vec<ValueId> = graph
        .nodes
        .iter()
        .filter(|node| !node.outputs.iter().any(|o| live_values.contains(o)))
        .flat_map(|node| node.outputs.iter().copied())
        .collect();
    for id in dead_outputs {
        graph.values.remove(&id);
    }
    graph.nodes.retain(|node| node.outputs.iter().any(|o| live_values.contains(o)));
    graph
}

fn is_elementwise(op: &str) -> bool {
    matches!(
        op,
        "elementwise.add.f32"
            | "elementwise.add"
            | "elementwise.mul.f32"
            | "elementwise.mul"
            | "elementwise.relu.f32"
            | "elementwise.relu"
            | "add"
            | "mul"
            | "relu"
    )
}

fn is_relu(op: &str) -> bool {
    matches!(op, "elementwise.relu.f32" | "elementwise.relu" | "relu")
}

/// Fuse `elem → relu` into a single annotated node when the relu has a unique producer.
///
/// Keeps the producer op name and sets `fused_relu=true`. Runtime applies ReLU after the op.
fn elementwise_fusion(mut graph: Graph) -> Result<(Graph, Vec<String>), IrDiagnostic> {
    let mut notes = vec!["scan elementwise→relu chains".into()];
    let mut producers: BTreeMap<ValueId, usize> = BTreeMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        for out in &node.outputs {
            producers.insert(*out, idx);
        }
    }

    let mut remove_relu: BTreeSet<NodeId> = BTreeSet::new();
    let mut plans: Vec<(usize, ValueId, ValueId)> = Vec::new();

    for node in &graph.nodes {
        if !is_relu(node.operator.0.as_str()) || node.inputs.len() != 1 || node.outputs.len() != 1 {
            continue;
        }
        let pred_val = node.inputs[0];
        let relu_out = node.outputs[0];
        let Some(&pred_idx) = producers.get(&pred_val) else {
            continue;
        };
        let pred = &graph.nodes[pred_idx];
        if !is_elementwise(pred.operator.0.as_str()) || is_relu(pred.operator.0.as_str()) {
            continue;
        }
        if pred.outputs != [pred_val] {
            continue;
        }
        if graph.outputs.contains(&pred_val) {
            continue;
        }
        let consumers = graph.nodes.iter().filter(|n| n.inputs.contains(&pred_val)).count();
        if consumers != 1 {
            continue;
        }
        plans.push((pred_idx, pred_val, relu_out));
        remove_relu.insert(node.id);
    }

    if plans.is_empty() {
        notes.push("no rewrite: no eligible elementwise→relu chain".into());
        return Ok((graph, notes));
    }

    for (pred_idx, old_out, new_out) in plans {
        let pred = graph.nodes.get_mut(pred_idx).ok_or_else(|| {
            IrDiagnostic::error(codes::PASS_FAILED, "fusion lost producer node").with_arg("node", pred_idx.to_string())
        })?;
        let op_name = pred.operator.0.clone();
        let node_id = pred.id.0;
        pred.attrs.insert("fused_relu".into(), titan_types::AttrValue::Bool(true));
        pred.outputs = vec![new_out];
        if let Some(spec) = graph.values.remove(&old_out) {
            graph.values.insert(new_out, spec);
        }
        notes.push(format!("fused {op_name} → relu into node {node_id}"));
    }

    graph.nodes.retain(|n| !remove_relu.contains(&n.id));
    Ok((graph, notes))
}
