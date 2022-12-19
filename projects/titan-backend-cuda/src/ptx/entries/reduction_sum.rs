use super::{
    super::ast::{Entry, F32Value, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, linear_f32_ptrs, linear_index_guard},
    regs::{b32, b64, f32, predicate},
};

pub(super) fn reduction_sum_f32(name: Identifier) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = done_label(&name);
    let done_store = Label(name.suffix("_done_store"));
    let loop_label = Label(name.suffix("_loop"));
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::U32(2)]);
    instructions.extend(linear_index_guard(3, 4, 5, 6, U32Value::Reg(b32(1)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MulLoU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
        PtxInstruction::DefineLabel(loop_label.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: done_store.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(4) },
        PtxInstruction::AddRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: loop_label.clone() },
        PtxInstruction::DefineLabel(done_store.clone()),
    ]);
    instructions.extend(linear_f32_ptrs(6, 5, &[(2, 6)]));
    instructions.extend([
        PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(3, 10, 7, 3), instructions }
}
