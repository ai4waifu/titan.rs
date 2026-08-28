use super::{
    super::ast::{Entry, F32Value, Identifier, ParameterKind, PtxInstruction, U32Value},
    params::{ParamLoad, declare_params, load_params, named_params, regs},
    prologue::{bounds_guard, done_label, entry_tail, kernel_label, linear_tid},
    regs::{b32, b64, f32, predicate},
};

pub(super) fn group_norm_f32(name: Identifier) -> Entry {
    let names = named_params::<12>(&name);
    let parameters = declare_params(
        &names,
        &[
            ParameterKind::GlobalF32Pointer,
            ParameterKind::GlobalF32Pointer,
            ParameterKind::GlobalF32Pointer,
            ParameterKind::GlobalF32Pointer,
            ParameterKind::U32,
            ParameterKind::U32,
            ParameterKind::U32,
            ParameterKind::U32,
            ParameterKind::U32,
            ParameterKind::F32,
            ParameterKind::U32,
            ParameterKind::U32,
        ],
    );
    let done = done_label(&name);
    let mean_loop = kernel_label(&name, "_mean_loop");
    let mean_done = kernel_label(&name, "_mean_done");
    let var_loop = kernel_label(&name, "_var_loop");
    let var_done = kernel_label(&name, "_var_done");
    let store_loop = kernel_label(&name, "_store_loop");
    let no_gamma = kernel_label(&name, "_no_gamma");
    let no_beta = kernel_label(&name, "_no_beta");
    let mut instructions = load_params(
        &names,
        &[
            ParamLoad::Ptr(1),
            ParamLoad::Ptr(2),
            ParamLoad::Ptr(3),
            ParamLoad::Ptr(4),
            ParamLoad::U32(1),
            ParamLoad::U32(2),
            ParamLoad::U32(3),
            ParamLoad::U32(4),
            ParamLoad::U32(5),
            ParamLoad::F32(6),
            ParamLoad::U32(14),
            ParamLoad::U32(15),
        ],
    );
    instructions.extend(linear_tid(6, 7, 8, 9, false));
    instructions.extend([PtxInstruction::MulLoU32 { destination: b32(10), left: b32(1), right: U32Value::Reg(b32(5)) }]);
    instructions.extend(bounds_guard(9, U32Value::Reg(b32(10)), 1, &done));
    instructions.extend([
        PtxInstruction::DivU32 { destination: b32(11), left: b32(9), right: U32Value::Reg(b32(5)) },
        PtxInstruction::RemU32 { destination: b32(12), left: b32(9), right: U32Value::Reg(b32(5)) },
        PtxInstruction::DivU32 { destination: b32(13), left: b32(2), right: U32Value::Reg(b32(5)) },
        PtxInstruction::MulLoU32 { destination: b32(16), left: b32(3), right: U32Value::Reg(b32(4)) },
        PtxInstruction::MulLoU32 { destination: b32(17), left: b32(13), right: U32Value::Reg(b32(16)) },
        PtxInstruction::MulLoU32 { destination: b32(18), left: b32(9), right: U32Value::Reg(b32(17)) },
        PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(mean_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(19), right: U32Value::Reg(b32(17)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: mean_done.clone() },
        PtxInstruction::AddU32 { destination: b32(20), left: b32(18), right: U32Value::Reg(b32(19)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(20), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::AddRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: mean_loop.clone() },
        PtxInstruction::DefineLabel(mean_done.clone()),
        PtxInstruction::CvtRnF32U32 { destination: f32(7), source: b32(17) },
        PtxInstruction::DivRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::MoveF32Imm { destination: f32(3), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(var_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(19), right: U32Value::Reg(b32(17)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: var_done.clone() },
        PtxInstruction::AddU32 { destination: b32(20), left: b32(18), right: U32Value::Reg(b32(19)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(20), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: var_loop.clone() },
        PtxInstruction::DefineLabel(var_done.clone()),
        PtxInstruction::DivRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(6)) },
        PtxInstruction::RsqrtApproxF32 { destination: f32(4), source: F32Value::Reg(f32(3)) },
        PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(store_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(19), right: U32Value::Reg(b32(17)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: done.clone() },
        PtxInstruction::AddU32 { destination: b32(20), left: b32(18), right: U32Value::Reg(b32(19)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(20), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
        PtxInstruction::AddS64 { destination: b64(7), left: b64(4), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::SubRnF32 { destination: f32(5), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(4)) },
        PtxInstruction::DivU32 { destination: b32(21), left: b32(19), right: U32Value::Reg(b32(16)) },
        PtxInstruction::MadLoU32 { destination: b32(21), left: b32(12), right: b32(13), addend: b32(21) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(3), left: b32(14), right: U32Value::Imm(0) },
        PtxInstruction::BranchIf { predicate: predicate(3), target: no_gamma.clone() },
        PtxInstruction::MultiplyWideU32 { destination: b64(8), left: b32(21), right: 4 },
        PtxInstruction::AddS64 { destination: b64(9), left: b64(2), right: b64(8) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(9) },
        PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::DefineLabel(no_gamma.clone()),
        PtxInstruction::SetPredicateEqU32 { destination: predicate(3), left: b32(15), right: U32Value::Imm(0) },
        PtxInstruction::BranchIf { predicate: predicate(3), target: no_beta.clone() },
        PtxInstruction::MultiplyWideU32 { destination: b64(8), left: b32(21), right: 4 },
        PtxInstruction::AddS64 { destination: b64(9), left: b64(3), right: b64(8) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(9) },
        PtxInstruction::AddRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::DefineLabel(no_beta.clone()),
        PtxInstruction::StoreGlobalF32 { pointer: b64(7), value: f32(5) },
        PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: store_loop.clone() },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(4, 22, 10, 8), instructions }
}
