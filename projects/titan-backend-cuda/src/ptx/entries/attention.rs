use super::{
    super::ast::{Entry, F32Value, Identifier, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{bounds_guard, done_label, entry_tail, kernel_label, linear_tid},
    regs::{b32, b64, f32, predicate},
};

pub(super) fn scaled_dot_product_attention_f32(name: Identifier) -> Entry {
    let names = named_params::<9>(&name);
    let parameters = buffer_u32_params(&names, 4);
    let done = done_label(&name);
    let max_loop = kernel_label(&name, "_max_loop");
    let max_inner_loop = kernel_label(&name, "_max_inner_loop");
    let max_inner_done = kernel_label(&name, "_max_inner_done");
    let max_next = kernel_label(&name, "_max_next");
    let max_done = kernel_label(&name, "_max_done");
    let sum_loop = kernel_label(&name, "_sum_loop");
    let sum_inner_loop = kernel_label(&name, "_sum_inner_loop");
    let sum_inner_done = kernel_label(&name, "_sum_inner_done");
    let sum_next = kernel_label(&name, "_sum_next");
    let sum_done = kernel_label(&name, "_sum_done");
    let value_loop = kernel_label(&name, "_value_loop");
    let value_inner_loop = kernel_label(&name, "_value_inner_loop");
    let value_inner_done = kernel_label(&name, "_value_inner_done");
    let value_next = kernel_label(&name, "_value_next");
    let value_done = kernel_label(&name, "_value_done");
    let mut instructions = load_params(
        &names,
        &[
            ParamLoad::Ptr(1),
            ParamLoad::Ptr(2),
            ParamLoad::Ptr(3),
            ParamLoad::Ptr(4),
            ParamLoad::U32(1),
            ParamLoad::U32(2),
            ParamLoad::U32(3),
            ParamLoad::U32(4),
            ParamLoad::U32(5),
        ],
    );
    instructions.extend(linear_tid(6, 7, 8, 9, false));
    instructions.extend([
        PtxInstruction::MulLoU32 { destination: b32(10), left: b32(1), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MulLoU32 { destination: b32(10), left: b32(10), right: U32Value::Reg(b32(3)) },
        PtxInstruction::MulLoU32 { destination: b32(10), left: b32(10), right: U32Value::Reg(b32(5)) },
    ]);
    instructions.extend(bounds_guard(9, U32Value::Reg(b32(10)), 1, &done));
    instructions.extend([
        PtxInstruction::DivU32 { destination: b32(11), left: b32(9), right: U32Value::Reg(b32(5)) },
        PtxInstruction::RemU32 { destination: b32(12), left: b32(9), right: U32Value::Reg(b32(5)) },
        PtxInstruction::DivU32 { destination: b32(13), left: b32(11), right: U32Value::Reg(b32(3)) },
        PtxInstruction::RemU32 { destination: b32(14), left: b32(11), right: U32Value::Reg(b32(3)) },
        PtxInstruction::DivU32 { destination: b32(15), left: b32(13), right: U32Value::Reg(b32(2)) },
        PtxInstruction::RemU32 { destination: b32(16), left: b32(13), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MadLoU32 { destination: b32(17), left: b32(15), right: b32(2), addend: b32(16) },
        PtxInstruction::MadLoU32 { destination: b32(17), left: b32(17), right: b32(3), addend: b32(14) },
        PtxInstruction::MulLoU32 { destination: b32(17), left: b32(17), right: U32Value::Reg(b32(5)) },
        PtxInstruction::CvtRnF32U32 { destination: f32(7), source: b32(5) },
        PtxInstruction::SqrtRnF32 { destination: f32(7), source: F32Value::Reg(f32(7)) },
        PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0xFF800000 },
        PtxInstruction::MoveU32 { destination: b32(20), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(max_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(20), right: U32Value::Reg(b32(4)) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: max_done.clone() },
        PtxInstruction::MadLoU32 { destination: b32(18), left: b32(15), right: b32(2), addend: b32(16) },
        PtxInstruction::MadLoU32 { destination: b32(18), left: b32(18), right: b32(4), addend: b32(20) },
        PtxInstruction::MulLoU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
        PtxInstruction::MoveF32Imm { destination: f32(2), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(max_inner_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(19), right: U32Value::Reg(b32(5)) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: max_inner_done.clone() },
        PtxInstruction::AddU32 { destination: b32(21), left: b32(17), right: U32Value::Reg(b32(19)) },
        PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(19)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(21), right: 4 },
        PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(22), right: 4 },
        PtxInstruction::AddS64 { destination: b64(7), left: b64(1), right: b64(5) },
        PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(6) },
        PtxInstruction::LoadGlobalF32 { destination: f32(5), pointer: b64(7) },
        PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(8) },
        PtxInstruction::FmaRnF32 {
            destination: f32(2),
            a: F32Value::Reg(f32(5)),
            b: F32Value::Reg(f32(6)),
            c: F32Value::Reg(f32(2)),
        },
        PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: max_inner_loop.clone() },
        PtxInstruction::DefineLabel(max_inner_done.clone()),
        PtxInstruction::DivRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::MaxF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::DefineLabel(max_next.clone()),
        PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: max_loop.clone() },
        PtxInstruction::DefineLabel(max_done.clone()),
        PtxInstruction::MoveF32Imm { destination: f32(3), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(20), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(sum_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(20), right: U32Value::Reg(b32(4)) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: sum_done.clone() },
        PtxInstruction::MadLoU32 { destination: b32(18), left: b32(15), right: b32(2), addend: b32(16) },
        PtxInstruction::MadLoU32 { destination: b32(18), left: b32(18), right: b32(4), addend: b32(20) },
        PtxInstruction::MulLoU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
        PtxInstruction::MoveF32Imm { destination: f32(2), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(sum_inner_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(19), right: U32Value::Reg(b32(5)) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: sum_inner_done.clone() },
        PtxInstruction::AddU32 { destination: b32(21), left: b32(17), right: U32Value::Reg(b32(19)) },
        PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(19)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(21), right: 4 },
        PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(22), right: 4 },
        PtxInstruction::AddS64 { destination: b64(7), left: b64(1), right: b64(5) },
        PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(6) },
        PtxInstruction::LoadGlobalF32 { destination: f32(5), pointer: b64(7) },
        PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(8) },
        PtxInstruction::FmaRnF32 {
            destination: f32(2),
            a: F32Value::Reg(f32(5)),
            b: F32Value::Reg(f32(6)),
            c: F32Value::Reg(f32(2)),
        },
        PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: sum_inner_loop.clone() },
        PtxInstruction::DefineLabel(sum_inner_done.clone()),
        PtxInstruction::DivRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(2), source: F32Value::Reg(f32(2)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::DefineLabel(sum_next.clone()),
        PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: sum_loop.clone() },
        PtxInstruction::DefineLabel(sum_done.clone()),
        PtxInstruction::MoveF32Imm { destination: f32(4), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(20), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(value_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(20), right: U32Value::Reg(b32(4)) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: value_done.clone() },
        PtxInstruction::MadLoU32 { destination: b32(18), left: b32(15), right: b32(2), addend: b32(16) },
        PtxInstruction::MadLoU32 { destination: b32(18), left: b32(18), right: b32(4), addend: b32(20) },
        PtxInstruction::MulLoU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
        PtxInstruction::MoveF32Imm { destination: f32(2), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(value_inner_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(19), right: U32Value::Reg(b32(5)) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: value_inner_done.clone() },
        PtxInstruction::AddU32 { destination: b32(21), left: b32(17), right: U32Value::Reg(b32(19)) },
        PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(19)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(21), right: 4 },
        PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(22), right: 4 },
        PtxInstruction::AddS64 { destination: b64(7), left: b64(1), right: b64(5) },
        PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(6) },
        PtxInstruction::LoadGlobalF32 { destination: f32(5), pointer: b64(7) },
        PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(8) },
        PtxInstruction::FmaRnF32 {
            destination: f32(2),
            a: F32Value::Reg(f32(5)),
            b: F32Value::Reg(f32(6)),
            c: F32Value::Reg(f32(2)),
        },
        PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: value_inner_loop.clone() },
        PtxInstruction::DefineLabel(value_inner_done.clone()),
        PtxInstruction::DivRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(2), source: F32Value::Reg(f32(2)) },
        PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(12)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(22), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(3), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(6) },
        PtxInstruction::FmaRnF32 {
            destination: f32(4),
            a: F32Value::Reg(f32(2)),
            b: F32Value::Reg(f32(6)),
            c: F32Value::Reg(f32(4)),
        },
        PtxInstruction::DefineLabel(value_next.clone()),
        PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: value_loop.clone() },
        PtxInstruction::DefineLabel(value_done.clone()),
        PtxInstruction::DivRnF32 { destination: f32(4), left: F32Value::Reg(f32(4)), right: F32Value::Reg(f32(3)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(4), right: b64(5) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(4) },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(2, 23, 9, 8), instructions }
}
