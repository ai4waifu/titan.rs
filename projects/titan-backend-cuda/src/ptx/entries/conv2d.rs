use std::num::NonZeroU8;

use super::super::ast::{
    Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn conv2d_f32(name: Identifier) -> Entry {
        let parameter_names: [Identifier; 21] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 4 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();

    let predicate = |index| Register::new(RegisterClass::Predicate, index);
    let b32 = |index| Register::new(RegisterClass::B32, index);
    let b64 = |index| Register::new(RegisterClass::B64, index);
    let f32 = |index| Register::new(RegisterClass::F32, index);
        let done = Label(name.suffix("_done"));
        let no_bias = Label(name.suffix("_no_bias"));
        let input_channel_loop = Label(name.suffix("_input_channel_loop"));
        let kernel_h_loop = Label(name.suffix("_kernel_h_loop"));
        let kernel_w_loop = Label(name.suffix("_kernel_w_loop"));
        let next_kernel_w = Label(name.suffix("_next_kernel_w"));
        let kernel_w_done = Label(name.suffix("_kernel_w_done"));
        let kernel_h_done = Label(name.suffix("_kernel_h_done"));
        let input_channel_done = Label(name.suffix("_input_channel_done"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(38).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(9).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(3), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(4), parameter: parameter_names[3].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[4].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[5].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(3), parameter: parameter_names[6].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(4), parameter: parameter_names[7].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(5), parameter: parameter_names[8].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(6), parameter: parameter_names[9].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(7), parameter: parameter_names[10].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(8), parameter: parameter_names[11].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(9), parameter: parameter_names[12].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(10), parameter: parameter_names[13].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(11), parameter: parameter_names[14].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(12), parameter: parameter_names[15].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(13), parameter: parameter_names[16].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(14), parameter: parameter_names[17].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(15), parameter: parameter_names[18].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(16), parameter: parameter_names[19].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(17), parameter: parameter_names[20].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(18) },
            PtxInstruction::MoveNtidX { destination: b32(19) },
            PtxInstruction::MoveTidX { destination: b32(20) },
            PtxInstruction::MadLoU32 { destination: b32(18), left: b32(18), right: b32(19), addend: b32(20) },
            PtxInstruction::MulLoU32 { destination: b32(19), left: b32(1), right: U32Value::Reg(b32(5)) },
            PtxInstruction::MulLoU32 { destination: b32(19), left: b32(19), right: U32Value::Reg(b32(8)) },
            PtxInstruction::MulLoU32 { destination: b32(19), left: b32(19), right: U32Value::Reg(b32(9)) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(18), right: U32Value::Reg(b32(19)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::DivU32 { destination: b32(20), left: b32(18), right: U32Value::Reg(b32(9)) },
            PtxInstruction::RemU32 { destination: b32(21), left: b32(18), right: U32Value::Reg(b32(9)) },
            PtxInstruction::DivU32 { destination: b32(22), left: b32(20), right: U32Value::Reg(b32(8)) },
            PtxInstruction::RemU32 { destination: b32(23), left: b32(20), right: U32Value::Reg(b32(8)) },
            PtxInstruction::DivU32 { destination: b32(24), left: b32(22), right: U32Value::Reg(b32(5)) },
            PtxInstruction::RemU32 { destination: b32(25), left: b32(22), right: U32Value::Reg(b32(5)) },
            PtxInstruction::DivU32 { destination: b32(26), left: b32(5), right: U32Value::Reg(b32(16)) },
            PtxInstruction::DivU32 { destination: b32(27), left: b32(25), right: U32Value::Reg(b32(26)) },
            PtxInstruction::DivU32 { destination: b32(28), left: b32(2), right: U32Value::Reg(b32(16)) },
            PtxInstruction::MulLoU32 { destination: b32(29), left: b32(27), right: U32Value::Reg(b32(28)) },
            PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(1), left: b32(17), right: U32Value::Imm(0) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: no_bias.clone() },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(25), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(3), right: b64(5) },
            PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(6) },
            PtxInstruction::DefineLabel(no_bias.clone()),
            PtxInstruction::MoveU32 { destination: b32(30), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(input_channel_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(30), right: U32Value::Reg(b32(28)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: input_channel_done.clone() },
            PtxInstruction::MoveU32 { destination: b32(31), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(kernel_h_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(31), right: U32Value::Reg(b32(6)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: kernel_h_done.clone() },
            PtxInstruction::MoveU32 { destination: b32(32), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(kernel_w_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(32), right: U32Value::Reg(b32(7)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: kernel_w_done.clone() },
            PtxInstruction::MulLoU32 { destination: b32(33), left: b32(31), right: U32Value::Reg(b32(14)) },
            PtxInstruction::MadLoU32 { destination: b32(33), left: b32(23), right: b32(10), addend: b32(33) },
            PtxInstruction::SubS32 { destination: b32(33), left: b32(33), right: U32Value::Reg(b32(12)) },
            PtxInstruction::MulLoU32 { destination: b32(34), left: b32(32), right: U32Value::Reg(b32(15)) },
            PtxInstruction::MadLoU32 { destination: b32(34), left: b32(21), right: b32(11), addend: b32(34) },
            PtxInstruction::SubS32 { destination: b32(34), left: b32(34), right: U32Value::Reg(b32(13)) },
            PtxInstruction::SetPredicateLtS32 { destination: predicate(2), left: b32(33), right: U32Value::Imm(0) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: next_kernel_w.clone() },
            PtxInstruction::SetPredicateLtS32 { destination: predicate(2), left: b32(34), right: U32Value::Imm(0) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: next_kernel_w.clone() },
            PtxInstruction::SetPredicateGeS32 { destination: predicate(2), left: b32(33), right: U32Value::Reg(b32(3)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: next_kernel_w.clone() },
            PtxInstruction::SetPredicateGeS32 { destination: predicate(2), left: b32(34), right: U32Value::Reg(b32(4)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: next_kernel_w.clone() },
            PtxInstruction::AddU32 { destination: b32(35), left: b32(29), right: U32Value::Reg(b32(30)) },
            PtxInstruction::MadLoU32 { destination: b32(36), left: b32(24), right: b32(2), addend: b32(35) },
            PtxInstruction::MadLoU32 { destination: b32(36), left: b32(36), right: b32(3), addend: b32(33) },
            PtxInstruction::MadLoU32 { destination: b32(36), left: b32(36), right: b32(4), addend: b32(34) },
            PtxInstruction::MadLoU32 { destination: b32(37), left: b32(25), right: b32(28), addend: b32(30) },
            PtxInstruction::MadLoU32 { destination: b32(37), left: b32(37), right: b32(6), addend: b32(31) },
            PtxInstruction::MadLoU32 { destination: b32(37), left: b32(37), right: b32(7), addend: b32(32) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(36), right: 4 },
            PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(37), right: 4 },
            PtxInstruction::AddS64 { destination: b64(7), left: b64(1), right: b64(5) },
            PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(6) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(7) },
            PtxInstruction::LoadGlobalF32 { destination: f32(3), pointer: b64(8) },
            PtxInstruction::FmaRnF32 { destination: f32(1), a: F32Value::Reg(f32(2)), b: F32Value::Reg(f32(3)), c: F32Value::Reg(f32(1)) },
            PtxInstruction::DefineLabel(next_kernel_w.clone()),
            PtxInstruction::AddU32 { destination: b32(32), left: b32(32), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: kernel_w_loop.clone() },
            PtxInstruction::DefineLabel(kernel_w_done.clone()),
            PtxInstruction::AddU32 { destination: b32(31), left: b32(31), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: kernel_h_loop.clone() },
            PtxInstruction::DefineLabel(kernel_h_done.clone()),
            PtxInstruction::AddU32 { destination: b32(30), left: b32(30), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: input_channel_loop.clone() },
            PtxInstruction::DefineLabel(input_channel_done.clone()),
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(18), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(4), right: b64(5) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
        }
}
