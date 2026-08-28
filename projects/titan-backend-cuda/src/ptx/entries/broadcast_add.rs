use super::{
    super::ast::{Entry, F32Value, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::linear_index_guard,
    regs::{b32, b64, f32, predicate},
};

pub(super) fn broadcast_add_f32(name: Identifier) -> Entry {
    let names = named_params::<16>(&name);
    let parameters = buffer_u32_params(&names, 3);
    let done = Label(name.suffix("_done"));
    let lhs0_done = Label(name.suffix("_lhs_0_done"));
    let lhs1_done = Label(name.suffix("_lhs_1_done"));
    let lhs2_done = Label(name.suffix("_lhs_2_done"));
    let lhs3_done = Label(name.suffix("_lhs_3_done"));
    let rhs0_done = Label(name.suffix("_rhs_0_done"));
    let rhs1_done = Label(name.suffix("_rhs_1_done"));
    let rhs2_done = Label(name.suffix("_rhs_2_done"));
    let rhs3_done = Label(name.suffix("_rhs_3_done"));
    let mut instructions =
        load_params(&names[..4], &[ParamLoad::Ptr(1), ParamLoad::Ptr(2), ParamLoad::Ptr(3), ParamLoad::U32(1)]);
    instructions.extend(linear_index_guard(2, 3, 4, 5, U32Value::Reg(b32(1)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MoveU32 { destination: b32(6), value: U32Value::Reg(b32(5)) },
        PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: names[15].clone() },
        PtxInstruction::RemU32 { destination: b32(16), left: b32(6), right: U32Value::Reg(b32(12)) },
        PtxInstruction::DivU32 { destination: b32(6), left: b32(6), right: U32Value::Reg(b32(12)) },
        PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: names[14].clone() },
        PtxInstruction::RemU32 { destination: b32(15), left: b32(6), right: U32Value::Reg(b32(12)) },
        PtxInstruction::DivU32 { destination: b32(6), left: b32(6), right: U32Value::Reg(b32(12)) },
        PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: names[13].clone() },
        PtxInstruction::RemU32 { destination: b32(14), left: b32(6), right: U32Value::Reg(b32(12)) },
        PtxInstruction::DivU32 { destination: b32(6), left: b32(6), right: U32Value::Reg(b32(12)) },
        PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: names[12].clone() },
        PtxInstruction::RemU32 { destination: b32(13), left: b32(6), right: U32Value::Reg(b32(12)) },
        PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
        PtxInstruction::MoveU32 { destination: b32(9), value: U32Value::Imm(0) },
        PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: names[4].clone() },
        PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: lhs0_done.clone() },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(13)) },
        PtxInstruction::DefineLabel(lhs0_done.clone()),
        PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: names[8].clone() },
        PtxInstruction::MulLoU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(11)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(11), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: rhs0_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(13)) },
        PtxInstruction::DefineLabel(rhs0_done.clone()),
        PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: names[5].clone() },
        PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: lhs1_done.clone() },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(14)) },
        PtxInstruction::DefineLabel(lhs1_done.clone()),
        PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: names[9].clone() },
        PtxInstruction::MulLoU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(11)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(11), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: rhs1_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(14)) },
        PtxInstruction::DefineLabel(rhs1_done.clone()),
        PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: names[6].clone() },
        PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: lhs2_done.clone() },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(15)) },
        PtxInstruction::DefineLabel(lhs2_done.clone()),
        PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: names[10].clone() },
        PtxInstruction::MulLoU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(11)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(11), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: rhs2_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(15)) },
        PtxInstruction::DefineLabel(rhs2_done.clone()),
        PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: names[7].clone() },
        PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: lhs3_done.clone() },
        PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(16)) },
        PtxInstruction::DefineLabel(lhs3_done.clone()),
        PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: names[11].clone() },
        PtxInstruction::MulLoU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(11)) },
        PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(11), right: U32Value::Imm(1) },
        PtxInstruction::BranchIf { predicate: predicate(1), target: rhs3_done.clone() },
        PtxInstruction::AddU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(16)) },
        PtxInstruction::DefineLabel(rhs3_done.clone()),
        PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(8), right: 4 },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(9), right: 4 },
        PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(5), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(4) },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(5) },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(3), right: b64(6) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(5) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(3) },
        PtxInstruction::DefineLabel(done.clone()),
        PtxInstruction::Return,
    ]);
    Entry { name, parameters, registers: regs(2, 17, 7, 4), instructions }
}
