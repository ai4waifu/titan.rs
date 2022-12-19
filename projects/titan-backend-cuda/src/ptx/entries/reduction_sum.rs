use std::num::NonZeroU8;

use super::{
    super::ast::{
        Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction, RegisterClass,
        RegisterDeclaration, U32Value,
    },
    prologue::linear_index_guard,
    regs::{b32, b64, f32, predicate},
};

pub(super) fn reduction_sum_f32(name: Identifier) -> Entry {
    let parameter_names: [Identifier; 4] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
    let parameters = parameter_names
        .iter()
        .enumerate()
        .map(|(index, parameter)| Parameter {
            name: parameter.clone(),
            kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
        })
        .collect();

    let done = Label(name.suffix("_done"));
    let done_store = Label(name.suffix("_done_store"));
    let loop_label = Label(name.suffix("_loop"));
    Entry {
        name,
        parameters,
        registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(10).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(3).unwrap() },
        ],
        instructions: {
            let mut instructions = vec![
                PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
                PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
                PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[2].clone() },
                PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[3].clone() },
            ];
            instructions.extend(linear_index_guard(3, 4, 5, 6, U32Value::Reg(b32(1)), 1, &done, true));
            instructions.extend([
                PtxInstruction::MulLoU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(2)) },
                PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
                PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
                PtxInstruction::DefineLabel(loop_label.clone()),
                PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
                PtxInstruction::BranchIf { predicate: predicate(2), target: done_store.clone() },
                PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
                PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
                PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
                PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(4) },
                PtxInstruction::AddRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
                PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
                PtxInstruction::Branch { target: loop_label.clone() },
                PtxInstruction::DefineLabel(done_store.clone()),
                PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(6), right: 4 },
                PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(5) },
                PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
                PtxInstruction::DefineLabel(done.clone()),
                PtxInstruction::Return,
            ]);
            instructions
        },
    }
}
