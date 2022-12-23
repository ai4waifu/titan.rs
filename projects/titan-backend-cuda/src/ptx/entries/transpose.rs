use super::{
    super::ast::{Entry, Identifier, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{bounds_guard, done_label, entry_tail, linear_f32_load, linear_f32_ptrs, linear_f32_store, linear_tid},
    regs::{b32, b64},
};

pub(super) fn transpose_f32(name: Identifier) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = done_label(&name);
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::U32(2)]);
    instructions.extend(linear_tid(3, 4, 5, 6, true));
    instructions.push(PtxInstruction::MulLoU32 { destination: b32(7), left: b32(1), right: U32Value::Reg(b32(2)) });
    instructions.extend(bounds_guard(6, U32Value::Reg(b32(7)), 1, &done));
    instructions.extend([
        PtxInstruction::DivU32 { destination: b32(8), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::RemU32 { destination: b32(9), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::MulLoU32 { destination: b32(10), left: b32(9), right: U32Value::Reg(b32(2)) },
        PtxInstruction::AddU32 { destination: b32(11), left: b32(10), right: U32Value::Reg(b32(8)) },
    ]);
    instructions.extend(linear_f32_ptrs(11, 3, &[(1, 4)]));
    instructions.push(linear_f32_load(4, 1));
    instructions.extend(linear_f32_ptrs(6, 5, &[(2, 6)]));
    instructions.push(linear_f32_store(6, 1));
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(2, 12, 7, 2), instructions }
}
