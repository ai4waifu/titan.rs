//! titan-ir-cpu-closed-loop: add/mul/relu graph → core passes → CPU plan → eager parity.

use std::{collections::HashMap, sync::Arc};

use titan_backend_cpu::CpuDriver;
use titan_graph::{
    empty_attrs, f32_contiguous, span, EffectContract, Graph, GraphConstraint, GraphNode, NodeId, OpRequest, ValueId,
};
use titan_hal::BackendDriver;
use titan_runtime::{CompileOptions, GraphBindings, Runtime};
use titan_tensor::{Device, Tensor};
use titan_types::{MemoryEffect, OperatorId};

fn device() -> Device {
    let driver = Arc::new(CpuDriver);
    Device::from_session(driver.open(driver.enumerate().unwrap()[0].device).unwrap())
}

fn build_graph() -> Graph {
    // y = relu((a + b) * c)
    let mut g = Graph::new();
    for id in 0..=5 {
        g.add_value(ValueId(id), f32_contiguous(vec![4])).unwrap();
    }
    g.add_input(ValueId(0)).unwrap();
    g.add_input(ValueId(1)).unwrap();
    g.add_input(ValueId(2)).unwrap();
    g.add_output(ValueId(5)).unwrap();
    g.add_node(GraphNode {
        id: NodeId(0),
        operator: OperatorId("elementwise.add.f32".into()),
        inputs: vec![ValueId(0), ValueId(1)],
        outputs: vec![ValueId(3)],
        attrs: empty_attrs(),
        effects: EffectContract {
            memory: MemoryEffect::Pure,
            deterministic: true,
        },
        source: span("cpu_ir_closed_loop", 1, 1),
    })
    .unwrap();
    g.add_node(GraphNode {
        id: NodeId(1),
        operator: OperatorId("elementwise.mul.f32".into()),
        inputs: vec![ValueId(3), ValueId(2)],
        outputs: vec![ValueId(4)],
        attrs: empty_attrs(),
        effects: EffectContract {
            memory: MemoryEffect::Pure,
            deterministic: true,
        },
        source: span("cpu_ir_closed_loop", 2, 1),
    })
    .unwrap();
    g.add_node(GraphNode {
        id: NodeId(2),
        operator: OperatorId("elementwise.relu.f32".into()),
        inputs: vec![ValueId(4)],
        outputs: vec![ValueId(5)],
        attrs: empty_attrs(),
        effects: EffectContract {
            memory: MemoryEffect::Pure,
            deterministic: true,
        },
        source: span("cpu_ir_closed_loop", 3, 1),
    })
    .unwrap();
    g
}

fn eager_add_mul_relu(runtime: &mut Runtime, a: &Tensor<f32, 1>, b: &Tensor<f32, 1>, c: &Tensor<f32, 1>) -> Vec<f32> {
    let add = runtime
        .execute(OpRequest {
            operator: OperatorId("elementwise.add.f32".into()),
            inputs: vec![a.handle(), b.handle()],
            outputs: vec![f32_contiguous(vec![4])],
            attrs: empty_attrs(),
            effects: EffectContract {
                memory: MemoryEffect::Writes,
                deterministic: true,
            },
            source: span("eager", 1, 1),
        })
        .unwrap()
        .wait()
        .unwrap();
    let mul = runtime
        .execute(OpRequest {
            operator: OperatorId("elementwise.mul.f32".into()),
            inputs: vec![add.outputs[0].clone(), c.handle()],
            outputs: vec![f32_contiguous(vec![4])],
            attrs: empty_attrs(),
            effects: EffectContract {
                memory: MemoryEffect::Writes,
                deterministic: true,
            },
            source: span("eager", 2, 1),
        })
        .unwrap()
        .wait()
        .unwrap();
    runtime
        .execute(OpRequest {
            operator: OperatorId("elementwise.relu.f32".into()),
            inputs: vec![mul.outputs[0].clone()],
            outputs: vec![f32_contiguous(vec![4])],
            attrs: empty_attrs(),
            effects: EffectContract {
                memory: MemoryEffect::Writes,
                deterministic: true,
            },
            source: span("eager", 3, 1),
        })
        .unwrap()
        .wait()
        .unwrap()
        .outputs[0]
        .to_vec_f32()
        .unwrap()
}

#[test]
fn cpu_ir_closed_loop_matches_eager_and_fuses_relu() {
    let d = device();
    let a = Tensor::<f32, 1>::from_slice(&d, [4], &[1.0, -2.0, 3.0, -4.0]).unwrap();
    let b = Tensor::<f32, 1>::from_slice(&d, [4], &[0.5, 1.0, -1.0, 2.0]).unwrap();
    let c = Tensor::<f32, 1>::from_slice(&d, [4], &[2.0, 2.0, 2.0, 2.0]).unwrap();

    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-ir-cpu-closed-loop"));
    let eager = eager_add_mul_relu(&mut runtime, &a, &b, &c);

    let graph = build_graph();
    let plan = runtime.compile(graph, CompileOptions::default()).unwrap();
    assert!(plan.pass_reports().iter().any(|r| r.name == "validate"));
    assert!(plan.pass_reports().iter().any(|r| r.name == "fusion.elementwise" && r.changed));
    // mul→relu fused → 2 nodes left (add + fused mul)
    assert_eq!(plan.node_count(), 2);

    let mut tensors = HashMap::new();
    tensors.insert(ValueId(0), a.handle());
    tensors.insert(ValueId(1), b.handle());
    tensors.insert(ValueId(2), c.handle());
    let compiled = plan
        .execute(&mut runtime, &GraphBindings { tensors })
        .unwrap()
        .wait()
        .unwrap()
        .outputs[0]
        .to_vec_f32()
        .unwrap();

    assert_eq!(compiled, eager);
    // (1+0.5)*2=3; (-2+1)*2=-2→0; (3-1)*2=4; (-4+2)*2=-4→0
    assert_eq!(compiled, vec![3.0, 0.0, 4.0, 0.0]);
}

#[test]
fn cpu_ir_compile_rejects_unsat_constraint() {
    let mut g = Graph::new();
    g.add_value(ValueId(0), f32_contiguous(vec![2])).unwrap();
    g.add_value(ValueId(1), f32_contiguous(vec![3])).unwrap();
    g.add_output(ValueId(0)).unwrap();
    g.add_constraint(GraphConstraint::SameShape {
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-ir-cpu-unsat"));
    let err = runtime.compile(g, CompileOptions::default()).unwrap_err();
    assert_eq!(err.phase, "compile");
    assert!(err.message.contains("DXO_IR_SHAPE_CONSTRAINT_UNSAT"));
}
