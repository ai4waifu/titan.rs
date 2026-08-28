use super::{
    super::ast::{Entry, F32Value, Identifier, PtxInstruction},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, flat_index_guard, linear_f32_ptrs},
    regs::{b64, f32},
};

pub(super) fn silu_f32(name: Identifier) -> Entry {
    let names = named_params::<3>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = done_label(&name);
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1)]);
    instructions.extend(flat_index_guard(&done));
    instructions.extend(linear_f32_ptrs(5, 3, &[(1, 4), (2, 5)]));
    instructions.extend([
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(2), source: F32Value::Reg(f32(2)) },
        PtxInstruction::AddRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::ImmBits(0x3F800000) },
        PtxInstruction::DivRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(3) },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(2, 6, 6, 4), instructions }
}
