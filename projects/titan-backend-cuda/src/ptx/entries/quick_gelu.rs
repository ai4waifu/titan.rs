use std::num::NonZeroU8;

use super::{
    super::ast::{
        Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction, RegisterClass,
        RegisterDeclaration, U32Value,
    },
    prologue::linear_index_guard,
    regs::{b32, b64, f32},
};

pub(super) fn quick_gelu_f32(name: Identifier) -> Entry {
    let parameter_names: [Identifier; 4] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
    let parameters = parameter_names
        .iter()
        .enumerate()
        .map(|(index, parameter)| Parameter {
            name: parameter.clone(),
            kind: if index < 2 {
                ParameterKind::GlobalF32Pointer
            }
            else if matches!(index, 3) {
                ParameterKind::F32
            }
            else {
                ParameterKind::U32
            },
        })
        .collect();

    let done = Label(name.suffix("_done"));
    let mut instructions = vec![
        PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
        PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
        PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[2].clone() },
        PtxInstruction::LoadParameterF32 { destination: f32(2), parameter: parameter_names[3].clone() },
    ];
    instructions.extend(linear_index_guard(2, 3, 4, 5, U32Value::Reg(b32(1)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(5), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::MulRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
        PtxInstruction::SubRnF32 { destination: f32(3), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(3)) },
        PtxInstruction::MulRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3FB8AA3B) },
        PtxInstruction::Ex2ApproxF32 { destination: f32(3), source: F32Value::Reg(f32(3)) },
        PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3F800000) },
        PtxInstruction::DivRnF32 { destination: f32(3), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(3)) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(3) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry {
        name,
        parameters,
        registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(6).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(6).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
        ],
        instructions,
    }
}
