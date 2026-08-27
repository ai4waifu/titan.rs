use std::num::NonZeroU8;

use super::super::ast::{
    Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn broadcast_add_f32(name: Identifier) -> Entry {
        let parameter_names: [Identifier; 16] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 3 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();

    let predicate = |index| Register::new(RegisterClass::Predicate, index);
    let b32 = |index| Register::new(RegisterClass::B32, index);
    let b64 = |index| Register::new(RegisterClass::B64, index);
    let f32 = |index| Register::new(RegisterClass::F32, index);
        let done = Label(name.suffix("_done"));
        let lhs0_done = Label(name.suffix("_lhs_0_done"));
        let lhs1_done = Label(name.suffix("_lhs_1_done"));
        let lhs2_done = Label(name.suffix("_lhs_2_done"));
        let lhs3_done = Label(name.suffix("_lhs_3_done"));
        let rhs0_done = Label(name.suffix("_rhs_0_done"));
        let rhs1_done = Label(name.suffix("_rhs_1_done"));
        let rhs2_done = Label(name.suffix("_rhs_2_done"));
        let rhs3_done = Label(name.suffix("_rhs_3_done"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(17).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(3), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[3].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(2) },
            PtxInstruction::MoveNtidX { destination: b32(3) },
            PtxInstruction::MoveTidX { destination: b32(4) },
            PtxInstruction::MultiplyAddLoS32 { destination: b32(5), left: b32(2), right: b32(3), addend: b32(4) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(5), right: U32Value::Reg(b32(1)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::MoveU32 { destination: b32(6), value: U32Value::Reg(b32(5)) },
            PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: parameter_names[15].clone() },
            PtxInstruction::RemU32 { destination: b32(16), left: b32(6), right: U32Value::Reg(b32(12)) },
            PtxInstruction::DivU32 { destination: b32(6), left: b32(6), right: U32Value::Reg(b32(12)) },
            PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: parameter_names[14].clone() },
            PtxInstruction::RemU32 { destination: b32(15), left: b32(6), right: U32Value::Reg(b32(12)) },
            PtxInstruction::DivU32 { destination: b32(6), left: b32(6), right: U32Value::Reg(b32(12)) },
            PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: parameter_names[13].clone() },
            PtxInstruction::RemU32 { destination: b32(14), left: b32(6), right: U32Value::Reg(b32(12)) },
            PtxInstruction::DivU32 { destination: b32(6), left: b32(6), right: U32Value::Reg(b32(12)) },
            PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: parameter_names[12].clone() },
            PtxInstruction::RemU32 { destination: b32(13), left: b32(6), right: U32Value::Reg(b32(12)) },
            PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
            PtxInstruction::MoveU32 { destination: b32(9), value: U32Value::Imm(0) },
            PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: parameter_names[4].clone() },
            PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: lhs0_done.clone() },
            PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(13)) },
            PtxInstruction::DefineLabel(lhs0_done.clone()),
            PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: parameter_names[8].clone() },
            PtxInstruction::MulLoU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(11)) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(11), right: U32Value::Imm(1) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: rhs0_done.clone() },
            PtxInstruction::AddU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(13)) },
            PtxInstruction::DefineLabel(rhs0_done.clone()),
            PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: parameter_names[5].clone() },
            PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: lhs1_done.clone() },
            PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(14)) },
            PtxInstruction::DefineLabel(lhs1_done.clone()),
            PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: parameter_names[9].clone() },
            PtxInstruction::MulLoU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(11)) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(11), right: U32Value::Imm(1) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: rhs1_done.clone() },
            PtxInstruction::AddU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(14)) },
            PtxInstruction::DefineLabel(rhs1_done.clone()),
            PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: parameter_names[6].clone() },
            PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: lhs2_done.clone() },
            PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(15)) },
            PtxInstruction::DefineLabel(lhs2_done.clone()),
            PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: parameter_names[10].clone() },
            PtxInstruction::MulLoU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(11)) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(11), right: U32Value::Imm(1) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: rhs2_done.clone() },
            PtxInstruction::AddU32 { destination: b32(9), left: b32(9), right: U32Value::Reg(b32(15)) },
            PtxInstruction::DefineLabel(rhs2_done.clone()),
            PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: parameter_names[7].clone() },
            PtxInstruction::MulLoU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(10)) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(10), right: U32Value::Imm(1) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: lhs3_done.clone() },
            PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Reg(b32(16)) },
            PtxInstruction::DefineLabel(lhs3_done.clone()),
            PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: parameter_names[11].clone() },
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
        ],
        }
}
