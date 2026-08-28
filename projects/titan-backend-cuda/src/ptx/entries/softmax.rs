use super::{
    super::ast::{Entry, F32Value, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::linear_index_guard,
    regs::{b32, b64, f32, predicate},
};

pub(super) fn softmax_f32(name: Identifier) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = Label(name.suffix("_done"));
    let max_loop = Label(name.suffix("_max_loop"));
    let max_done = Label(name.suffix("_max_done"));
    let sum_loop = Label(name.suffix("_sum_loop"));
    let sum_done = Label(name.suffix("_sum_done"));
    let normalize_loop = Label(name.suffix("_normalize_loop"));
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::U32(2)]);
    instructions.extend(linear_index_guard(3, 4, 5, 6, U32Value::Reg(b32(1)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MulLoU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0xFF7FFFFF },
        PtxInstruction::DefineLabel(max_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: max_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(4) },
        PtxInstruction::MaxF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: max_loop.clone() },
        PtxInstruction::DefineLabel(max_done.clone()),
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::MoveF32Imm { destination: f32(3), bits: 0x00000000 },
        PtxInstruction::DefineLabel(sum_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: sum_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(4) },
        PtxInstruction::SubRnF32 { destination: f32(4), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(4), left: F32Value::Reg(f32(4)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(4), source: F32Value::Reg(f32(4)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(4)) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(4) },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: sum_loop.clone() },
        PtxInstruction::DefineLabel(sum_done.clone()),
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(normalize_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(4), pointer: b64(5) },
        PtxInstruction::DivRnF32 { destination: f32(4), left: F32Value::Reg(f32(4)), right: F32Value::Reg(f32(3)) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(4) },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: normalize_loop.clone() },
        PtxInstruction::DefineLabel(done.clone()),
        PtxInstruction::Return,
    ]);
    Entry { name, parameters, registers: regs(3, 10, 6, 5), instructions }
}
