//! Shared linear thread-index, bounds guard, pointer offset, and entry tail helpers.

use super::{
    super::ast::{Identifier, Label, PtxInstruction, U32Value},
    regs::{b32, b64, predicate},
};

/// Standard kernel exit label (`{name}_done`).
pub(super) fn done_label(name: &Identifier) -> Label {
    Label(name.suffix("_done"))
}

/// `done:` followed by `ret`.
pub(super) fn entry_tail(done: &Label) -> [PtxInstruction; 2] {
    [PtxInstruction::DefineLabel(done.clone()), PtxInstruction::Return]
}

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

/// Flat 1-D launch: linear tid in `%r5`, element count limit in `%r1`.
pub(super) fn flat_index_guard(done: &Label) -> [PtxInstruction; 6] {
    linear_index_guard(2, 3, 4, 5, U32Value::Reg(b32(1)), 1, done, true)
}

/// `mul.wide.u32 offset = index * sizeof(f32)`.
pub(super) fn f32_byte_offset(index_reg: u8, offset_reg: u8) -> PtxInstruction {
    PtxInstruction::MultiplyWideU32 { destination: b64(offset_reg), left: b32(index_reg), right: 4 }
}

/// For each `(index, offset)` pair, emit a byte offset from a linear element index.
pub(super) fn f32_byte_offsets(pairs: &[(u8, u8)]) -> Vec<PtxInstruction> {
    pairs.iter().copied().map(|(index, offset)| f32_byte_offset(index, offset)).collect()
}

/// For each `(base, dest)` pair: `dest = base + offset`.
pub(super) fn ptr_plus_offset(offset_reg: u8, ptrs: &[(u8, u8)]) -> Vec<PtxInstruction> {
    ptrs.iter()
        .copied()
        .map(|(base, dest)| PtxInstruction::AddS64 { destination: b64(dest), left: b64(base), right: b64(offset_reg) })
        .collect()
}

/// Byte offset from `index_reg`, then add each base pointer into `dest` registers.
pub(super) fn linear_f32_ptrs(index_reg: u8, offset_reg: u8, ptrs: &[(u8, u8)]) -> Vec<PtxInstruction> {
    let mut instructions = vec![f32_byte_offset(index_reg, offset_reg)];
    instructions.extend(ptr_plus_offset(offset_reg, ptrs));
    instructions
}
