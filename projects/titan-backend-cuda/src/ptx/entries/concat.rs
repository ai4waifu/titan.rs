use super::{
    super::ast::{Entry, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::linear_index_guard,
    regs::{b32, b64, f32, predicate},
};

pub(super) fn concat_f32(name: Identifier) -> Entry {
    let names = named_params::<5>(&name);
    let parameters = buffer_u32_params(&names, 3);
    let done = Label(name.suffix("_done"));
    let right = Label(name.suffix("_right"));
    let mut instructions =
        load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::Ptr(3), ParamLoad::U32(1), ParamLoad::U32(2)]);
    instructions.extend(linear_index_guard(3, 4, 5, 6, U32Value::Reg(b32(2)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(6), right: 4 },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(3), right: b64(4) },
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: right.clone() },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(4) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(6) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(1) },
        PtxInstruction::Branch { target: done.clone() },
        PtxInstruction::DefineLabel(right),
        PtxInstruction::SubU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(7), left: b32(7), right: 4 },
        PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(7) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(8) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(1) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry { name, parameters, registers: regs(3, 8, 9, 2), instructions }
}
