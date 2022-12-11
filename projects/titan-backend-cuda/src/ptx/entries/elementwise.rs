use std::num::NonZeroU8;

use super::super::ast::{
    ElementwiseOperation, Entry, Identifier, Label, Parameter, ParameterIndex, ParameterKind,
    PtxInstruction, Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn elementwise_f32(name: Identifier, operation: ElementwiseOperation) -> Entry {
    let parameter_names = [
        name.parameter(ParameterIndex(0)),
        name.parameter(ParameterIndex(1)),
        name.parameter(ParameterIndex(2)),
        name.parameter(ParameterIndex(3)),
    ];
    let parameters = vec![
        Parameter { name: parameter_names[0].clone(), kind: ParameterKind::GlobalF32Pointer },
        Parameter { name: parameter_names[1].clone(), kind: ParameterKind::GlobalF32Pointer },
        Parameter { name: parameter_names[2].clone(), kind: ParameterKind::GlobalF32Pointer },
        Parameter { name: parameter_names[3].clone(), kind: ParameterKind::U32 },
    ];
    let predicate = |index| Register::new(RegisterClass::Predicate, index);
    let b32 = |index| Register::new(RegisterClass::B32, index);
    let b64 = |index| Register::new(RegisterClass::B64, index);
    let f32 = |index| Register::new(RegisterClass::F32, index);
    let done = Label(name.suffix("_done"));
    Entry {
        name,
        parameters,
        registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(6).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(8).unwrap() },
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
            PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(5), right: 4 },
            PtxInstruction::AddS64 { destination: b64(5), left: b64(1), right: b64(4) },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(4) },
            PtxInstruction::AddS64 { destination: b64(7), left: b64(3), right: b64(4) },
            PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(5) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
            PtxInstruction::ArithmeticF32 { destination: f32(3), operation, left: f32(1), right: f32(2) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(7), value: f32(3) },
            PtxInstruction::DefineLabel(done),
            PtxInstruction::Return,
        ],
    }
}
