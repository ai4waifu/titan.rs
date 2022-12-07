use std::{
    error::Error,
    fmt::{Debug, Display, Formatter},
};

mod convert;
mod display;

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, TitanError>;

/// A boxed error kind, wrapping an [TitanErrorKind].
#[derive(Clone)]
pub struct TitanError {
    kind: Box<TitanErrorKind>,
}

/// The kind of [TitanError].
#[derive(Debug, Copy, Clone)]
pub enum TitanErrorKind {
    /// An unknown error.
    UnknownError,
}
