use titan_graph::{Graph, TensorSpec, ValueId};
use titan_types::{AliasContract, DType, Layout, Shape, Strides};

#[test]
fn typed_graph_registers_values() {
    let mut graph = Graph::new();
    graph
        .add_value(
            ValueId(0),
            TensorSpec {
                dtype: DType::F32,
                shape: Shape(vec![2, 2]),
                strides: Strides(vec![2, 1]),
                layout: Layout::Contiguous,
                alias: AliasContract::NoAlias,
            },
        )
        .unwrap();
    assert!(graph.values.contains_key(&ValueId(0)));
}
