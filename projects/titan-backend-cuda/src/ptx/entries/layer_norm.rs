use super::{
    super::ast::{Entry, F32Value, Identifier, ParameterKind, PtxInstruction, U32Value},
    params::{ParamLoad, declare_params, load_params, named_params, regs},
    prologue::{done_label, entry_tail, kernel_label, linear_index_guard},
    regs::{b32, b64, f32, predicate},
};

pub(super) fn layer_norm_f32(name: Identifier) -> Entry {
    let names = named_params::<9>(&name);
    let parameters = declare_params(
        &names,
        &[
            ParameterKind::GlobalF32Pointer,
            ParameterKind::GlobalF32Pointer,
            ParameterKind::GlobalF32Pointer,
            ParameterKind::GlobalF32Pointer,
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
            ParamLoad::F32(6),
            ParamLoad::U32(10),
            ParamLoad::U32(11),
        ],
    );
    instructions.extend(linear_index_guard(3, 4, 5, 6, U32Value::Reg(b32(1)), 1, &done, false));
    instructions.extend([
        PtxInstruction::MulLoU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(mean_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: mean_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::AddRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: mean_loop.clone() },
        PtxInstruction::DefineLabel(mean_done.clone()),
        PtxInstruction::CvtRnF32U32 { destination: f32(7), source: b32(2) },
        PtxInstruction::DivRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::MoveF32Imm { destination: f32(3), bits: 0x00000000 },
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(var_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: var_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: var_loop.clone() },
        PtxInstruction::DefineLabel(var_done.clone()),
        PtxInstruction::DivRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(7)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(6)) },
        PtxInstruction::RsqrtApproxF32 { destination: f32(4), source: F32Value::Reg(f32(3)) },
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::DefineLabel(store_loop.clone()),
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(9), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
        PtxInstruction::AddS64 { destination: b64(7), left: b64(4), right: b64(5) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
        PtxInstruction::SubRnF32 { destination: f32(5), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
        PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(4)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(3), left: b32(10), right: U32Value::Imm(0) },
        PtxInstruction::BranchIf { predicate: predicate(3), target: no_gamma.clone() },
        PtxInstruction::MultiplyWideU32 { destination: b64(8), left: b32(8), right: 4 },
        PtxInstruction::AddS64 { destination: b64(9), left: b64(2), right: b64(8) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(9) },
        PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::DefineLabel(no_gamma.clone()),
        PtxInstruction::SetPredicateEqU32 { destination: predicate(3), left: b32(11), right: U32Value::Imm(0) },
        PtxInstruction::BranchIf { predicate: predicate(3), target: no_beta.clone() },
        PtxInstruction::MultiplyWideU32 { destination: b64(8), left: b32(8), right: 4 },
        PtxInstruction::AddS64 { destination: b64(9), left: b64(3), right: b64(8) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(9) },
        PtxInstruction::AddRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::DefineLabel(no_beta.clone()),
        PtxInstruction::StoreGlobalF32 { pointer: b64(7), value: f32(5) },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
        PtxInstruction::Branch { target: store_loop.clone() },
    ]);
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(4, 12, 10, 8), instructions }
}
