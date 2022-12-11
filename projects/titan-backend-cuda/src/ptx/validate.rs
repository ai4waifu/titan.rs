//! Light validation of typed PTX entries before stringify.

use std::collections::HashSet;

use titan_kernel::KernelError;

use super::ast::{Entry, PtxInstruction, Register, RegisterClass, U32Value};

pub(super) fn validate_entry(entry: &Entry) -> Result<(), KernelError> {
    let mut counts = [0u8; 4];
    for declaration in &entry.registers {
        let slot = match declaration.class {
            RegisterClass::Predicate => 0,
            RegisterClass::B32 => 1,
            RegisterClass::B64 => 2,
            RegisterClass::F32 => 3,
        };
        counts[slot] = declaration.count.get();
    }

    let mut labels = HashSet::new();
    for instruction in &entry.instructions {
        if let PtxInstruction::DefineLabel(label) = instruction {
            labels.insert(label.0.as_str().to_owned());
        }
    }

    for instruction in &entry.instructions {
        check_instruction(instruction, &counts, &labels)?;
    }
    Ok(())
}

fn class_slot(class: RegisterClass) -> usize {
    match class {
        RegisterClass::Predicate => 0,
        RegisterClass::B32 => 1,
        RegisterClass::B64 => 2,
        RegisterClass::F32 => 3,
    }
}

fn check_reg(register: Register, counts: &[u8; 4]) -> Result<(), KernelError> {
    let count = counts[class_slot(register.class)];
    if register.index.get() > count {
        return Err(KernelError::Compile(format!(
            "PTX register {} exceeds declared count {count}",
            register
        )));
    }
    Ok(())
}

fn check_u32(value: U32Value, counts: &[u8; 4]) -> Result<(), KernelError> {
    match value {
        U32Value::Reg(register) => check_reg(register, counts),
        U32Value::Imm(_) => Ok(()),
    }
}

fn check_instruction(
    instruction: &PtxInstruction,
    counts: &[u8; 4],
    labels: &HashSet<String>,
) -> Result<(), KernelError> {
    match instruction {
        PtxInstruction::LoadParameterU64 { destination, .. }
        | PtxInstruction::LoadParameterU32 { destination, .. }
        | PtxInstruction::LoadParameterF32 { destination, .. }
        | PtxInstruction::MoveCtaIdX { destination }
        | PtxInstruction::MoveNtidX { destination }
        | PtxInstruction::MoveTidX { destination }
        | PtxInstruction::MoveF32Imm { destination, .. }
        | PtxInstruction::CvtRnF32U32 { destination, .. } => check_reg(*destination, counts),
        PtxInstruction::MoveU32 { destination, value } => {
            check_reg(*destination, counts)?;
            check_u32(*value, counts)
        }
        PtxInstruction::MoveF32 { destination, source } => {
            check_reg(*destination, counts)?;
            check_reg(*source, counts)
        }
        PtxInstruction::MultiplyAddLoS32 { destination, left, right, addend }
        | PtxInstruction::MadLoU32 { destination, left, right, addend } => {
            check_reg(*destination, counts)?;
            check_reg(*left, counts)?;
            check_reg(*right, counts)?;
            check_reg(*addend, counts)
        }
        PtxInstruction::MulLoU32 { destination, left, right }
        | PtxInstruction::DivU32 { destination, left, right }
        | PtxInstruction::RemU32 { destination, left, right }
        | PtxInstruction::AddU32 { destination, left, right }
        | PtxInstruction::SubU32 { destination, left, right }
        | PtxInstruction::SubS32 { destination, left, right }
        | PtxInstruction::SetPredicateGeU32 { destination, left, right }
        | PtxInstruction::SetPredicateEqU32 { destination, left, right }
        | PtxInstruction::SetPredicateLtS32 { destination, left, right }
        | PtxInstruction::SetPredicateGeS32 { destination, left, right } => {
            check_reg(*destination, counts)?;
            check_reg(*left, counts)?;
            check_u32(*right, counts)
        }
        PtxInstruction::SetPredicateLtF32 { destination, left, right }
        | PtxInstruction::MaxF32 { destination, left, right }
        | PtxInstruction::AddRnF32 { destination, left, right }
        | PtxInstruction::SubRnF32 { destination, left, right }
        | PtxInstruction::MulRnF32 { destination, left, right }
        | PtxInstruction::DivRnF32 { destination, left, right } => {
            check_reg(*destination, counts)?;
            check_f32(*left, counts)?;
            check_f32(*right, counts)
        }
        PtxInstruction::BranchIf { predicate, target } => {
            check_reg(*predicate, counts)?;
            require_label(target.0.as_str(), labels)
        }
        PtxInstruction::Branch { target } => require_label(target.0.as_str(), labels),
        PtxInstruction::MultiplyWideU32 { destination, left, .. } => {
            check_reg(*destination, counts)?;
            check_reg(*left, counts)
        }
        PtxInstruction::AddS64 { destination, left, right } => {
            check_reg(*destination, counts)?;
            check_reg(*left, counts)?;
            check_reg(*right, counts)
        }
        PtxInstruction::LoadGlobalF32 { destination, pointer } => {
            check_reg(*destination, counts)?;
            check_reg(*pointer, counts)
        }
        PtxInstruction::ArithmeticF32 { destination, left, right, .. } => {
            check_reg(*destination, counts)?;
            check_reg(*left, counts)?;
            check_reg(*right, counts)
        }
        PtxInstruction::StoreGlobalF32 { pointer, value } => {
            check_reg(*pointer, counts)?;
            check_reg(*value, counts)
        }
        PtxInstruction::SqrtRnF32 { destination, source }
        | PtxInstruction::RsqrtApproxF32 { destination, source }
        | PtxInstruction::Ex2ApproxF32 { destination, source } => {
            check_reg(*destination, counts)?;
            check_f32(*source, counts)
        }
        PtxInstruction::FmaRnF32 { destination, a, b, c } => {
            check_reg(*destination, counts)?;
            check_f32(*a, counts)?;
            check_f32(*b, counts)?;
            check_f32(*c, counts)
        }
        PtxInstruction::DefineLabel(_) | PtxInstruction::Return => Ok(()),
    }
}

fn check_f32(value: super::ast::F32Value, counts: &[u8; 4]) -> Result<(), KernelError> {
    match value {
        super::ast::F32Value::Reg(register) => check_reg(register, counts),
        super::ast::F32Value::ImmBits(_) => Ok(()),
    }
}

fn require_label(name: &str, labels: &HashSet<String>) -> Result<(), KernelError> {
    if labels.contains(name) {
        Ok(())
    } else {
        Err(KernelError::Compile(format!("PTX branch target `{name}` has no DefineLabel")))
    }
}
