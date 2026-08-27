//! Shared typed PTX AST: identifiers, registers, parameters, module/entry shells.

use std::{fmt, num::NonZeroU8};

use titan_kernel::KernelError;

use super::instruction::PtxInstruction;

#[derive(Clone, Copy)]
pub(crate) enum RegisterClass {
    Predicate,
    B32,
    B64,
    F32,
}

impl RegisterClass {
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Self::Predicate => "%p",
            Self::B32 => "%r",
            Self::B64 => "%rd",
            Self::F32 => "%f",
        }
    }

    pub(crate) fn ptx_type(self) -> &'static str {
        match self {
            Self::Predicate => ".pred",
            Self::B32 => ".b32",
            Self::B64 => ".b64",
            Self::F32 => ".f32",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ElementwiseOperation {
    Add,
    Mul,
    Fma { addend: FmaAddend },
}

#[derive(Clone, Copy)]
pub(crate) enum FmaAddend {
    Left,
    Right,
}

#[derive(Clone, Debug)]
pub(crate) struct Identifier(String);

impl Identifier {
    pub(crate) fn from_kernel_id(kernel_id: &str) -> Result<Self, KernelError> {
        let mut name = String::from("titan_");
        for character in kernel_id.bytes() {
            match character {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => name.push(character as char),
                b'.' | b'-' => name.push('_'),
                _ => return Err(KernelError::Unsupported("kernel ID cannot be represented as a PTX identifier".into())),
            }
        }
        if kernel_id.is_empty() {
            return Err(KernelError::Unsupported("kernel ID cannot be empty".into()));
        }
        Ok(Self(name))
    }

    pub(crate) fn parameter(&self, index: ParameterIndex) -> Self {
        Self(format!("{}_param_{}", self.0, index.0))
    }

    pub(crate) fn suffix(&self, suffix: &str) -> Self {
        debug_assert!(suffix.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'));
        Self(format!("{}{}", self.0, suffix))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum PtxVersion {
    V80,
}

impl fmt::Display for PtxVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::V80 => "8.0",
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Target(pub(crate) u16);

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "sm_{}", self.0)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum AddressSize {
    Bits64,
}

impl fmt::Display for AddressSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("64")
    }
}

pub(crate) struct PtxModule {
    pub(crate) version: PtxVersion,
    pub(crate) target: Target,
    pub(crate) address_size: AddressSize,
    pub(crate) entry: Entry,
}

impl fmt::Display for PtxModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, ".version {}", self.version)?;
        writeln!(formatter, ".target {}", self.target)?;
        writeln!(formatter, ".address_size {}", self.address_size)?;
        writeln!(formatter)?;
        write!(formatter, "{}", self.entry)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ParameterIndex(pub(crate) u8);

pub(crate) struct Parameter {
    pub(crate) name: Identifier,
    pub(crate) kind: ParameterKind,
}

#[derive(Clone, Copy)]
pub(crate) enum ParameterKind {
    GlobalF32Pointer,
    U32,
    F32,
}

impl fmt::Display for Parameter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = match self.kind {
            ParameterKind::GlobalF32Pointer => ".u64",
            ParameterKind::U32 => ".u32",
            ParameterKind::F32 => ".f32",
        };
        write!(formatter, ".param {ty} {}", self.name)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Register {
    pub(crate) class: RegisterClass,
    pub(crate) index: NonZeroU8,
}

impl Register {
    pub(crate) fn new(class: RegisterClass, index: u8) -> Self {
        Self { class, index: NonZeroU8::new(index).expect("PTX registers are one-indexed") }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", self.class.prefix(), self.index)
    }
}

pub(crate) struct RegisterDeclaration {
    pub(crate) class: RegisterClass,
    pub(crate) count: NonZeroU8,
}

impl fmt::Display for RegisterDeclaration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, ".reg {} {}<{}>;", self.class.ptx_type(), self.class.prefix(), self.count)
    }
}

#[derive(Clone)]
pub(crate) struct Label(pub(crate) Identifier);

impl fmt::Display for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

pub(crate) struct Entry {
    pub(crate) name: Identifier,
    pub(crate) parameters: Vec<Parameter>,
    pub(crate) registers: Vec<RegisterDeclaration>,
    pub(crate) instructions: Vec<PtxInstruction>,
}

impl fmt::Display for Entry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, ".visible .entry {}(", self.name)?;
        for (index, parameter) in self.parameters.iter().enumerate() {
            write!(formatter, "    {parameter}")?;
            if index + 1 != self.parameters.len() {
                writeln!(formatter, ",")?;
            } else {
                writeln!(formatter)?;
            }
        }
        writeln!(formatter, ")")?;
        writeln!(formatter, "{{")?;
        for register in &self.registers {
            writeln!(formatter, "    {register}")?;
        }
        writeln!(formatter)?;
        for instruction in &self.instructions {
            writeln!(formatter, "    {instruction}")?;
        }
        writeln!(formatter, "}}")
    }
}
