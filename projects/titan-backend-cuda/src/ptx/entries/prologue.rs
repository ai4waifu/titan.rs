//! Shared linear thread-index + bounds-guard prologue for PTX entries.

use super::{
    super::ast::{Label, PtxInstruction, U32Value},
    regs::{b32, predicate},
};

/// `mov` cta/ntid/tid → `mad.lo` into `linear_reg`.
pub(super) fn linear_tid(cta_reg: u8, ntid_reg: u8, tid_reg: u8, linear_reg: u8, signed_mad: bool) -> [PtxInstruction; 4] {
    let mad = if signed_mad {
        PtxInstruction::MultiplyAddLoS32 {
            destination: b32(linear_reg),
            left: b32(cta_reg),
            right: b32(ntid_reg),
            addend: b32(tid_reg),
        }
    }
    else {
        PtxInstruction::MadLoU32 {
            destination: b32(linear_reg),
            left: b32(cta_reg),
            right: b32(ntid_reg),
            addend: b32(tid_reg),
        }
    };
    [
        PtxInstruction::MoveCtaIdX { destination: b32(cta_reg) },
        PtxInstruction::MoveNtidX { destination: b32(ntid_reg) },
        PtxInstruction::MoveTidX { destination: b32(tid_reg) },
        mad,
    ]
}

/// `setp.ge.u32` + `@p bra done` against a precomputed linear index.
pub(super) fn bounds_guard(linear_reg: u8, limit: U32Value, pred_reg: u8, done: &Label) -> [PtxInstruction; 2] {
    [
        PtxInstruction::SetPredicateGeU32 { destination: predicate(pred_reg), left: b32(linear_reg), right: limit },
        PtxInstruction::BranchIf { predicate: predicate(pred_reg), target: done.clone() },
    ]
}

/// `linear_tid` followed by `bounds_guard`.
pub(super) fn linear_index_guard(
    cta_reg: u8,
    ntid_reg: u8,
    tid_reg: u8,
    linear_reg: u8,
    limit: U32Value,
    pred_reg: u8,
    done: &Label,
    signed_mad: bool,
) -> [PtxInstruction; 6] {
    let [a, b, c, d] = linear_tid(cta_reg, ntid_reg, tid_reg, linear_reg, signed_mad);
    let [e, f] = bounds_guard(linear_reg, limit, pred_reg, done);
    [a, b, c, d, e, f]
}
