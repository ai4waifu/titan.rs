use super::{
    super::ast::{ElementwiseOperation, Entry, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::linear_index_guard,
    regs::{b32, b64, f32},
};

pub(super) fn elementwise_f32(name: Identifier, operation: ElementwiseOperation) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = buffer_u32_params(&names, 3);
    let done = Label(name.suffix("_done"));
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::Ptr(3), ParamLoad::U32(1)]);
    instructions.extend(linear_index_guard(2, 3, 4, 5, U32Value::Reg(b32(1)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(5), right: 4 },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(1), right: b64(4) },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(4) },
        PtxInstruction::AddS64 { destination: b64(7), left: b64(3), right: b64(4) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::ArithmeticF32 { destination: f32(3), operation, left: f32(1), right: f32(2) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(7), value: f32(3) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry { name, parameters, registers: regs(2, 6, 8, 4), instructions }
}
