use super::{
    super::ast::{Entry, Identifier, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, kernel_label, linear_index_guard, linear_f32_load, linear_f32_store},
    regs::{b32, b64, predicate},
};

pub(super) fn concat_f32(name: Identifier) -> Entry {
    let names = named_params::<5>(&name);
    let parameters = buffer_u32_params(&names, 3);
    let done = done_label(&name);
    let right = kernel_label(&name, "_right");
    let mut instructions =
        load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::Ptr(3), ParamLoad::U32(1), ParamLoad::U32(2)]);
    instructions.extend(linear_index_guard(3, 4, 5, 6, U32Value::Reg(b32(2)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(6), right: 4 },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(3), right: b64(4) },
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: right.clone() },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(4) },
        linear_f32_load(6, 1),
        linear_f32_store(5, 1),
        PtxInstruction::Branch { target: done.clone() },
        PtxInstruction::DefineLabel(right),
        PtxInstruction::SubU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(7), left: b32(7), right: 4 },
        PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(7) },
        linear_f32_load(8, 1),
        linear_f32_store(5, 1),
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(3, 8, 9, 2), instructions }
}
