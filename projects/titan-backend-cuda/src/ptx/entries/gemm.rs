use super::{
    super::ast::{Entry, F32Value, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{bounds_guard, done_label, entry_tail, linear_f32_ptrs, linear_tid},
    regs::{b32, b64, f32, predicate},
};

pub(super) fn gemm_f32(name: Identifier) -> Entry {
    let names = named_params::<6>(&name);
    let parameters = buffer_u32_params(&names, 3);
    let done = done_label(&name);
    let done_store = Label(name.suffix("_done_store"));
    let loop_label = Label(name.suffix("_k_loop"));
    let mut instructions = load_params(
        &names,
        &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::Ptr(3), ParamLoad::U32(1), ParamLoad::U32(2), ParamLoad::U32(3)],
    );
    instructions.extend(linear_tid(4, 5, 6, 7, false));
    instructions.push(PtxInstruction::MulLoU32 { destination: b32(8), left: b32(1), right: U32Value::Reg(b32(2)) });
    instructions.extend(bounds_guard(7, U32Value::Reg(b32(8)), 1, &done));
    instructions.extend([
        PtxInstruction::DivU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(2)) },
        PtxInstruction::RemU32 { destination: b32(10), left: b32(7), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MoveU32 { destination: b32(11), value: U32Value::Imm(0) },
        PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
        PtxInstruction::DefineLabel(loop_label.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(11), right: U32Value::Reg(b32(3)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: done_store.clone() },
        PtxInstruction::MadLoU32 { destination: b32(12), left: b32(9), right: b32(3), addend: b32(11) },
        PtxInstruction::MadLoU32 { destination: b32(13), left: b32(11), right: b32(2), addend: b32(10) },
        PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(12), right: 4 },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(13), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(4) },
        PtxInstruction::AddS64 { destination: b64(7), left: b64(2), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::LoadGlobalF32 { destination: f32(3), pointer: b64(7) },
        PtxInstruction::FmaRnF32 {
            destination: f32(1),
            a: F32Value::Reg(f32(2)),
            b: F32Value::Reg(f32(3)),
            c: F32Value::Reg(f32(1)),
        },
        PtxInstruction::AddU32 { destination: b32(11), left: b32(11), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: loop_label },
        PtxInstruction::DefineLabel(done_store),
    ]);
    instructions.extend(linear_f32_ptrs(7, 8, &[(3, 9)]));
    instructions.extend([
        PtxInstruction::StoreGlobalF32 { pointer: b64(9), value: f32(1) },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(3, 14, 10, 4), instructions }
}
