use super::{
    super::ast::{Entry, F32Value, Identifier, Label, ParameterKind, PtxInstruction, U32Value},
    params::{ParamLoad, declare_params, load_params, named_params, regs},
    prologue::linear_index_guard,
    regs::{b32, b64, f32},
};

pub(super) fn quick_gelu_f32(name: Identifier) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = declare_params(
        &names,
        &[ParameterKind::GlobalF32Pointer, ParameterKind::GlobalF32Pointer, ParameterKind::U32, ParameterKind::F32],
    );
    let done = Label(name.suffix("_done"));
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::F32(2)]);
    instructions.extend(linear_index_guard(2, 3, 4, 5, U32Value::Reg(b32(1)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(5), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::MulRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::SubRnF32 { destination: f32(3), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(3)) },
        PtxInstruction::MulRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(3), source: F32Value::Reg(f32(3)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3F800000) },
        PtxInstruction::DivRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(3)) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(3) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry { name, parameters, registers: regs(2, 6, 6, 4), instructions }
}
