use super::{
    super::ast::{Entry, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{bounds_guard, linear_tid},
    regs::{b32, b64, f32},
};

pub(super) fn transpose_f32(name: Identifier) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = Label(name.suffix("_done"));
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::U32(2)]);
    instructions.extend(linear_tid(3, 4, 5, 6, true));
    instructions.push(PtxInstruction::MulLoU32 { destination: b32(7), left: b32(1), right: U32Value::Reg(b32(2)) });
    instructions.extend(bounds_guard(6, U32Value::Reg(b32(7)), 1, &done));
    instructions.extend([
        PtxInstruction::DivU32 { destination: b32(8), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::RemU32 { destination: b32(9), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::MulLoU32 { destination: b32(10), left: b32(9), right: U32Value::Reg(b32(2)) },
        PtxInstruction::AddU32 { destination: b32(11), left: b32(10), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(11), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(6), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(5) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry { name, parameters, registers: regs(2, 12, 7, 2), instructions }
}
