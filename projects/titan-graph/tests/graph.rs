use titan_graph::{CommandBuffer, Op};
#[test]
fn fuses_and_submits() {
    let mut graph = CommandBuffer::new();
    graph.push(Op::Multiply);
    graph.push(Op::Add);
    graph.push(Op::Dead);
    assert_eq!(graph.submit(|ops| ops).join().unwrap(), vec![Op::FusedMultiplyAdd]);
}
