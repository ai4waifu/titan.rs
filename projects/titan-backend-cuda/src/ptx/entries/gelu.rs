use super::{
    super::ast::{Entry, F32Value, Identifier, Label, PtxInstruction},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, flat_index_guard, linear_f32_ptrs},
    regs::{b64, f32, predicate},
};

pub(super) fn gelu_f32(name: Identifier) -> Entry {
    let names = named_params::<3>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = done_label(&name);
    let negative = Label(name.suffix("_negative"));
    let signed_done = Label(name.suffix("_signed_done"));
    let mut instructions = load_params(&names, &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::U32(1)]);
    instructions.extend(flat_index_guard(&done));
    instructions.extend(linear_f32_ptrs(5, 3, &[(1, 4), (2, 5)]));
    instructions.extend([
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(1)), right: F32Value::ImmBits(0x3F3504F3) },
        PtxInstruction::MoveF32 { destination: f32(3), source: f32(2) },
        PtxInstruction::SetPredicateLtF32 {
            destination: predicate(2),
            left: F32Value::Reg(f32(2)),
            right: F32Value::ImmBits(0x00000000),
        },
        PtxInstruction::BranchIf { predicate: predicate(2), target: negative.clone() },
        PtxInstruction::MoveF32Imm { destination: f32(4), bits: 0x3F800000 },
        PtxInstruction::Branch { target: signed_done.clone() },
        PtxInstruction::DefineLabel(negative.clone()),
        PtxInstruction::MoveF32Imm { destination: f32(4), bits: 0xBF800000 },
        PtxInstruction::SubRnF32 { destination: f32(3), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(3)) },
        PtxInstruction::DefineLabel(signed_done.clone()),
        PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3EA7BA05) },
        PtxInstruction::AddRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::ImmBits(0x3F800000) },
        PtxInstruction::DivRnF32 { destination: f32(5), left: F32Value::ImmBits(0x3F800000), right: F32Value::Reg(f32(5)) },
        PtxInstruction::MoveF32Imm { destination: f32(6), bits: 0x3F87DC22 },
        PtxInstruction::FmaRnF32 {
            destination: f32(6),
            a: F32Value::Reg(f32(6)),
            b: F32Value::Reg(f32(5)),
            c: F32Value::ImmBits(0xBFBA00E3),
        },
        PtxInstruction::FmaRnF32 {
            destination: f32(6),
            a: F32Value::Reg(f32(6)),
            b: F32Value::Reg(f32(5)),
            c: F32Value::ImmBits(0x3FB5F0E3),
        },
        PtxInstruction::FmaRnF32 {
            destination: f32(6),
            a: F32Value::Reg(f32(6)),
            b: F32Value::Reg(f32(5)),
            c: F32Value::ImmBits(0xBE91A98E),
        },
        PtxInstruction::FmaRnF32 {
            destination: f32(6),
            a: F32Value::Reg(f32(6)),
            b: F32Value::Reg(f32(5)),
            c: F32Value::ImmBits(0x3E827906),
        },
        PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(5)) },
        PtxInstruction::MulRnF32 { destination: f32(7), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(3)) },
        PtxInstruction::SubRnF32 { destination: f32(7), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(7)) },
        PtxInstruction::MulRnF32 { destination: f32(7), left: F32Value::Reg(f32(7)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(7), source: F32Value::Reg(f32(7)) },
        PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::SubRnF32 { destination: f32(6), left: F32Value::ImmBits(0x3F800000), right: F32Value::Reg(f32(6)) },
        PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(4)) },
        PtxInstruction::AddRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::ImmBits(0x3F800000) },
        PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::ImmBits(0x3F000000) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(6) },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(3, 6, 6, 10), instructions }
}
