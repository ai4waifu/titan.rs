//! Shared register constructors for PTX entry builders.

use super::super::ast::{Register, RegisterClass};

#[inline]
pub(super) fn predicate(index: u8) -> Register {
    Register::new(RegisterClass::Predicate, index)
}

#[inline]
pub(super) fn b32(index: u8) -> Register {
    Register::new(RegisterClass::B32, index)
}

#[inline]
pub(super) fn b64(index: u8) -> Register {
    Register::new(RegisterClass::B64, index)
}

#[inline]
pub(super) fn f32(index: u8) -> Register {
    Register::new(RegisterClass::F32, index)
}
