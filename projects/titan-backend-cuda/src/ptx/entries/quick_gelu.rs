use super::{
    super::ast::{Entry, F32Value, Identifier, ParameterKind, PtxInstruction},
    params::{ParamLoad, declare_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, flat_index_guard, linear_f32_load, linear_f32_ptrs, linear_f32_store},
    regs::f32,
};

pub(super) fn quick_gelu_f32(name: Identifier) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = declare_params(
        &names,
        &[ParameterKind::GlobalF32Pointer, ParameterKind::GlobalF32Pointer, ParameterKind::U32, ParameterKind::F32],
    );
    let done = done_label(&name);
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1), ParamLoad::F32(2)]);
    instructions.extend(flat_index_guard(&done));
    instructions.extend(linear_f32_ptrs(5, 3, &[(1, 4), (2, 5)]));
    instructions.push(linear_f32_load(4, 1));
    instructions.extend([
        PtxInstruction::MulRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::SubRnF32 { destination: f32(3), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(3)) },
        PtxInstruction::MulRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(3), source: F32Value::Reg(f32(3)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3F800000) },
        PtxInstruction::DivRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(3)) },
    ]);
    instructions.push(linear_f32_store(5, 3));
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(2, 6, 6, 4), instructions }
}
