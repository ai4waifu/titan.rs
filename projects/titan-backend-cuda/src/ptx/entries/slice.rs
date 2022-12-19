use super::{
    super::ast::{Entry, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::linear_index_guard,
    regs::{b32, b64, f32},
};

pub(super) fn slice_f32(name: Identifier) -> Entry {
    let names = named_params::<5>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = Label(name.suffix("_done"));
    let mut instructions =
        load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::U32(2), ParamLoad::U32(3)]);
    instructions.extend(linear_index_guard(4, 5, 6, 7, U32Value::Reg(b32(3)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MadLoU32 { destination: b32(8), left: b32(7), right: b32(2), addend: b32(1) },
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(8), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(7), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(5) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry { name, parameters, registers: regs(2, 10, 7, 2), instructions }
}
