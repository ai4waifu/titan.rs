use super::{
    super::ast::{Entry, Identifier, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, linear_f32_load, linear_f32_ptrs, linear_f32_store, linear_index_guard},
    regs::{b32, b64},
};

pub(super) fn slice_f32(name: Identifier) -> Entry {
    let names = named_params::<5>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = done_label(&name);
    let mut instructions =
        load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::U32(2), ParamLoad::U32(3)]);
    instructions.extend(linear_index_guard(4, 5, 6, 7, U32Value::Reg(b32(3)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MadLoU32 { destination: b32(8), left: b32(7), right: b32(2), addend: b32(1) },
    ]);
    instructions.extend(linear_f32_ptrs(8, 3, &[(1, 4)]));
    instructions.push(linear_f32_load(4, 1));
    instructions.extend(linear_f32_ptrs(7, 5, &[(2, 6)]));
    instructions.push(linear_f32_store(6, 1));
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(2, 10, 7, 2), instructions }
}
