#![warn(missing_docs)]
//! Backend-neutral allocation and execution primitives.

use std::fmt::Debug;

pub trait Backend: Clone + Default + Debug + Send + Sync + 'static {
    type Storage<T: Copy + Send + Sync + 'static>: AsRef<[T]> + AsMut<[T]> + Send + Sync;

    const NAME: &'static str;

    fn allocate<T: Copy + Send + Sync + 'static>(&self, values: Vec<T>) -> Self::Storage<T>;
}

/// Portable CPU backend. GPU implementations plug into the same trait later.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cpu;

impl Backend for Cpu {
    type Storage<T: Copy + Send + Sync + 'static> = Vec<T>;
    const NAME: &'static str = "cpu";

    fn allocate<T: Copy + Send + Sync + 'static>(&self, values: Vec<T>) -> Self::Storage<T> {
        values
    }
}
