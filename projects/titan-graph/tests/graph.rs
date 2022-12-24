use titan_graph::{
    attr_int, builtin_pass_registry, codes, empty_attrs, f32_contiguous, span, EffectContract, Graph,
    GraphConstraint, GraphNode, IrDiagnostic, NodeId, PassStage, ValueId, GRAPH_SCHEMA_VERSION,
};
use titan_types::{MemoryEffect, OperatorId};

fn sample_add_graph() -> Graph {
    let mut g = Graph::new();
    g.add_value(ValueId(0), f32_contiguous(vec![2, 2])).unwrap();
    g.add_value(ValueId(1), f32_contiguous(vec![2, 2])).unwrap();
    g.add_value(ValueId(2), f32_contiguous(vec![2, 2])).unwrap();
    g.add_input(ValueId(0)).unwrap();
    g.add_input(ValueId(1)).unwrap();
    g.add_output(ValueId(2)).unwrap();
    g.add_constraint(GraphConstraint::SameShape {
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    g.add_constraint(GraphConstraint::SameDtype {
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    let mut attrs = empty_attrs();
    attrs.insert("alpha".into(), attr_int(1));
    g.add_node(GraphNode {
        id: NodeId(0),
        operator: OperatorId("add".into()),
        inputs: vec![ValueId(0), ValueId(1)],
        outputs: vec![ValueId(2)],
        attrs,
        effects: EffectContract {
            memory: MemoryEffect::Pure,
            deterministic: true,
        },
        source: span("test", 1, 1),
    })
    .unwrap();
    g
}

#[test]
fn schema_version_and_validate() {
    let g = sample_add_graph();
    assert_eq!(g.schema, GRAPH_SCHEMA_VERSION);
    g.validate().expect("valid graph");
}

#[test]
fn semantic_hash_stable_and_ignores_debug() {
    let mut a = sample_add_graph();
    let mut b = sample_add_graph();
    a.debug.insert("note".into(), "first".into());
    b.debug.insert("note".into(), "second".into());
    assert_eq!(a.semantic_hash(), b.semantic_hash());
    b.outputs.push(ValueId(1));
    assert_ne!(a.semantic_hash(), b.semantic_hash());
}

#[test]
fn json_roundtrip_preserves_hash() {
    let g = sample_add_graph();
    let hash = g.semantic_hash();
    let again = g.roundtrip_json().expect("roundtrip");
    assert_eq!(again.semantic_hash(), hash);
    assert_eq!(again, g);
}

#[test]
fn validation_emits_living15_json() {
    let mut g = Graph::new();
    g.add_value(ValueId(0), f32_contiguous(vec![2])).unwrap();
    // no outputs → GRAPH_INVALID
    let err = g.validate().unwrap_err();
    assert_eq!(err.code, codes::GRAPH_INVALID);
    let json = err.to_json();
    let parsed: IrDiagnostic = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.code, codes::GRAPH_INVALID);
    assert_eq!(parsed.severity, titan_graph::Severity::Error);
}

#[test]
fn shape_constraint_unsat_diagnostic() {
    let mut g = Graph::new();
    g.add_value(ValueId(0), f32_contiguous(vec![2, 2])).unwrap();
    g.add_value(ValueId(1), f32_contiguous(vec![3, 3])).unwrap();
    g.add_output(ValueId(0)).unwrap();
    g.add_constraint(GraphConstraint::SameShape {
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    let err = g.validate().unwrap_err();
    assert_eq!(err.code, codes::SHAPE_CONSTRAINT_UNSAT);
    assert_eq!(err.args.get("lhs").map(String::as_str), Some("0"));
    assert!(err.to_json().contains("DXO_IR_SHAPE_CONSTRAINT_UNSAT"));
}

#[test]
fn pass_registry_builtin_contract() {
    let registry = builtin_pass_registry();
    assert!(registry.get("validate").is_some());
    assert!(registry.get("fusion.elementwise").is_some());
    let core = registry.by_stage(PassStage::Core);
    assert!(core.len() >= 3);
    let decl = registry.get("dce").unwrap();
    let diag = decl.failure_diagnostic("output removed");
    assert_eq!(diag.code, codes::PASS_FAILED);
    assert_eq!(diag.args.get("pass").map(String::as_str), Some("dce"));
}
