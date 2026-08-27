//! Atomic PTX instructions and typed operands. Each Display arm emits one PTX line.

use std::fmt;

use super::types::{ElementwiseOperation, FmaAddend, Identifier, Label, Register};

#[derive(Clone, Copy)]
pub(crate) enum U32Value {
    Reg(Register),
    Imm(u32),
}

impl fmt::Display for U32Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(register) => write!(formatter, "{register}"),
            Self::Imm(value) => write!(formatter, "{value}"),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum F32Value {
    Reg(Register),
    ImmBits(u32),
}

impl fmt::Display for F32Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(register) => write!(formatter, "{register}"),
            Self::ImmBits(bits) => write!(formatter, "0f{bits:08X}"),
        }
    }
}

pub(crate) enum PtxInstruction {
    LoadParameterU64 { destination: Register, parameter: Identifier },
    LoadParameterU32 { destination: Register, parameter: Identifier },
    LoadParameterF32 { destination: Register, parameter: Identifier },
    MoveCtaIdX { destination: Register },
    MoveNtidX { destination: Register },
    MoveTidX { destination: Register },
    MoveU32 { destination: Register, value: U32Value },
    MoveF32 { destination: Register, source: Register },
    MoveF32Imm { destination: Register, bits: u32 },
    MultiplyAddLoS32 { destination: Register, left: Register, right: Register, addend: Register },
    MadLoU32 { destination: Register, left: Register, right: Register, addend: Register },
    MulLoU32 { destination: Register, left: Register, right: U32Value },
    DivU32 { destination: Register, left: Register, right: U32Value },
    RemU32 { destination: Register, left: Register, right: U32Value },
    AddU32 { destination: Register, left: Register, right: U32Value },
    SubU32 { destination: Register, left: Register, right: U32Value },
    SubS32 { destination: Register, left: Register, right: U32Value },
    SetPredicateGeU32 { destination: Register, left: Register, right: U32Value },
    SetPredicateEqU32 { destination: Register, left: Register, right: U32Value },
    SetPredicateLtS32 { destination: Register, left: Register, right: U32Value },
    SetPredicateGeS32 { destination: Register, left: Register, right: U32Value },
    SetPredicateLtF32 { destination: Register, left: F32Value, right: F32Value },
    BranchIf { predicate: Register, target: Label },
    Branch { target: Label },
    MultiplyWideU32 { destination: Register, left: Register, right: u8 },
    AddS64 { destination: Register, left: Register, right: Register },
    LoadGlobalF32 { destination: Register, pointer: Register },
    ArithmeticF32 { destination: Register, operation: ElementwiseOperation, left: Register, right: Register },
    StoreGlobalF32 { pointer: Register, value: Register },
    CvtRnF32U32 { destination: Register, source: Register },
    SqrtRnF32 { destination: Register, source: F32Value },
    RsqrtApproxF32 { destination: Register, source: F32Value },
    MaxF32 { destination: Register, left: F32Value, right: F32Value },
    Ex2ApproxF32 { destination: Register, source: F32Value },
    AddRnF32 { destination: Register, left: F32Value, right: F32Value },
    SubRnF32 { destination: Register, left: F32Value, right: F32Value },
    MulRnF32 { destination: Register, left: F32Value, right: F32Value },
    DivRnF32 { destination: Register, left: F32Value, right: F32Value },
    FmaRnF32 { destination: Register, a: F32Value, b: F32Value, c: F32Value },
    DefineLabel(Label),
    Return,
}

impl fmt::Display for PtxInstruction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadParameterU64 { destination, parameter } => {
                write!(formatter, "ld.param.u64 {destination}, [{parameter}];")
            }
            Self::LoadParameterU32 { destination, parameter } => {
                write!(formatter, "ld.param.u32 {destination}, [{parameter}];")
            }
            Self::LoadParameterF32 { destination, parameter } => {
                write!(formatter, "ld.param.f32 {destination}, [{parameter}];")
            }
            Self::MoveCtaIdX { destination } => write!(formatter, "mov.u32 {destination}, %ctaid.x;"),
            Self::MoveNtidX { destination } => write!(formatter, "mov.u32 {destination}, %ntid.x;"),
            Self::MoveTidX { destination } => write!(formatter, "mov.u32 {destination}, %tid.x;"),
            Self::MoveU32 { destination, value } => write!(formatter, "mov.u32 {destination}, {value};"),
            Self::MoveF32 { destination, source } => write!(formatter, "mov.f32 {destination}, {source};"),
            Self::MoveF32Imm { destination, bits } => {
                write!(formatter, "mov.f32 {destination}, 0f{bits:08X};")
            }
            Self::MultiplyAddLoS32 { destination, left, right, addend } => {
                write!(formatter, "mad.lo.s32 {destination}, {left}, {right}, {addend};")
            }
            Self::MadLoU32 { destination, left, right, addend } => {
                write!(formatter, "mad.lo.u32 {destination}, {left}, {right}, {addend};")
            }
            Self::MulLoU32 { destination, left, right } => {
                write!(formatter, "mul.lo.u32 {destination}, {left}, {right};")
            }
            Self::DivU32 { destination, left, right } => {
                write!(formatter, "div.u32 {destination}, {left}, {right};")
            }
            Self::RemU32 { destination, left, right } => {
                write!(formatter, "rem.u32 {destination}, {left}, {right};")
            }
            Self::AddU32 { destination, left, right } => {
                write!(formatter, "add.u32 {destination}, {left}, {right};")
            }
            Self::SubU32 { destination, left, right } => {
                write!(formatter, "sub.u32 {destination}, {left}, {right};")
            }
            Self::SubS32 { destination, left, right } => {
                write!(formatter, "sub.s32 {destination}, {left}, {right};")
            }
            Self::SetPredicateGeU32 { destination, left, right } => {
                write!(formatter, "setp.ge.u32 {destination}, {left}, {right};")
            }
            Self::SetPredicateEqU32 { destination, left, right } => {
                write!(formatter, "setp.eq.u32 {destination}, {left}, {right};")
            }
            Self::SetPredicateLtS32 { destination, left, right } => {
                write!(formatter, "setp.lt.s32 {destination}, {left}, {right};")
            }
            Self::SetPredicateGeS32 { destination, left, right } => {
                write!(formatter, "setp.ge.s32 {destination}, {left}, {right};")
            }
            Self::SetPredicateLtF32 { destination, left, right } => {
                write!(formatter, "setp.lt.f32 {destination}, {left}, {right};")
            }
            Self::BranchIf { predicate, target } => write!(formatter, "@{predicate} bra {target};"),
            Self::Branch { target } => write!(formatter, "bra {target};"),
            Self::MultiplyWideU32 { destination, left, right } => {
                write!(formatter, "mul.wide.u32 {destination}, {left}, {right};")
            }
            Self::AddS64 { destination, left, right } => {
                write!(formatter, "add.s64 {destination}, {left}, {right};")
            }
            Self::LoadGlobalF32 { destination, pointer } => {
                write!(formatter, "ld.global.f32 {destination}, [{pointer}];")
            }
            Self::ArithmeticF32 { destination, operation, left, right } => match operation {
                ElementwiseOperation::Add => write!(formatter, "add.rn.f32 {destination}, {left}, {right};"),
                ElementwiseOperation::Mul => write!(formatter, "mul.rn.f32 {destination}, {left}, {right};"),
                ElementwiseOperation::Fma { addend } => {
                    let addend = match addend {
                        FmaAddend::Left => left,
                        FmaAddend::Right => right,
                    };
                    write!(formatter, "fma.rn.f32 {destination}, {left}, {right}, {addend};")
                }
            },
            Self::StoreGlobalF32 { pointer, value } => {
                write!(formatter, "st.global.f32 [{pointer}], {value};")
            }
            Self::CvtRnF32U32 { destination, source } => {
                write!(formatter, "cvt.rn.f32.u32 {destination}, {source};")
            }
            Self::SqrtRnF32 { destination, source } => {
                write!(formatter, "sqrt.rn.f32 {destination}, {source};")
            }
            Self::RsqrtApproxF32 { destination, source } => {
                write!(formatter, "rsqrt.approx.f32 {destination}, {source};")
            }
            Self::MaxF32 { destination, left, right } => {
                write!(formatter, "max.f32 {destination}, {left}, {right};")
            }
            Self::Ex2ApproxF32 { destination, source } => {
                write!(formatter, "ex2.approx.f32 {destination}, {source};")
            }
            Self::AddRnF32 { destination, left, right } => {
                write!(formatter, "add.rn.f32 {destination}, {left}, {right};")
            }
            Self::SubRnF32 { destination, left, right } => {
                write!(formatter, "sub.rn.f32 {destination}, {left}, {right};")
            }
            Self::MulRnF32 { destination, left, right } => {
                write!(formatter, "mul.rn.f32 {destination}, {left}, {right};")
            }
            Self::DivRnF32 { destination, left, right } => {
                write!(formatter, "div.rn.f32 {destination}, {left}, {right};")
            }
            Self::FmaRnF32 { destination, a, b, c } => {
                write!(formatter, "fma.rn.f32 {destination}, {a}, {b}, {c};")
            }
            Self::DefineLabel(label) => write!(formatter, "{label}:"),
            Self::Return => formatter.write_str("ret;"),
        }
    }
}
