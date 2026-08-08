#![warn(missing_docs)]
//! Compile-time-friendly graph representations and a lightweight executor.

use std::thread::{self, JoinHandle};

/// Operations represented in a Titan compute graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Op {
    Multiply,
    Add,
    FusedMultiplyAdd,
    Dead,
}

/// An ordered command buffer that can be optimized before submission.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandBuffer {
    operations: Vec<Op>,
}
impl CommandBuffer {
    /// Creates an empty command buffer.
    pub fn new() -> Self {
        Self::default()
    }
    /// Appends an operation.
    pub fn push(&mut self, operation: Op) {
        self.operations.push(operation);
    }
    /// Returns optimized operations, fusing multiply followed by add and removing dead code.
    pub fn optimize(&self) -> Vec<Op> {
        let mut output = Vec::new();
        let mut i = 0;
        while i < self.operations.len() {
            match (&self.operations[i], self.operations.get(i + 1)) {
                (Op::Multiply, Some(Op::Add)) => {
                    output.push(Op::FusedMultiplyAdd);
                    i += 2;
                }
                (Op::Dead, _) => i += 1,
                (op, _) => {
                    output.push(op.clone());
                    i += 1;
                }
            }
        }
        output
    }
    /// Submits work on a dedicated thread, mirroring asynchronous device execution.
    pub fn submit<T: Send + 'static>(&self, work: impl FnOnce(Vec<Op>) -> T + Send + 'static) -> JoinHandle<T> {
        let optimized = self.optimize();
        thread::spawn(move || work(optimized))
    }
}
