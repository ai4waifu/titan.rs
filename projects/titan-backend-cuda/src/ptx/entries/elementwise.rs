use super::{
    super::ast::{ElementwiseOperation, Entry, Identifier, PtxInstruction},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, flat_index_guard, linear_f32_ptrs},
    regs::{b64, f32},
};

pub(super) fn elementwise_f32(name: Identifier, operation: ElementwiseOperation) -> Entry {
    let names = named_params::<4>(&name);
    let parameters = buffer_u32_params(&names, 3);
    let done = done_label(&name);
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::Ptr(3), ParamLoad::U32(1)]);
    instructions.extend(flat_index_guard(&done));
    instructions.extend(linear_f32_ptrs(5, 4, &[(1, 5), (2, 6), (3, 7)]));
    instructions.extend([
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::ArithmeticF32 { destination: f32(3), operation, left: f32(1), right: f32(2) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(7), value: f32(3) },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(2, 6, 8, 4), instructions }
}
