use super::{
    super::ast::{Entry, F32Value, Identifier, Label, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{bounds_guard, linear_tid},
    regs::{b32, b64, f32, predicate},
};

pub(super) fn conv2d_f32(name: Identifier) -> Entry {
    let names = named_params::<21>(&name);
    let parameters = buffer_u32_params(&names, 4);
    let done = Label(name.suffix("_done"));
    let no_bias = Label(name.suffix("_no_bias"));
    let input_channel_loop = Label(name.suffix("_input_channel_loop"));
    let kernel_h_loop = Label(name.suffix("_kernel_h_loop"));
    let kernel_w_loop = Label(name.suffix("_kernel_w_loop"));
    let next_kernel_w = Label(name.suffix("_next_kernel_w"));
    let kernel_w_done = Label(name.suffix("_kernel_w_done"));
    let kernel_h_done = Label(name.suffix("_kernel_h_done"));
    let input_channel_done = Label(name.suffix("_input_channel_done"));
    let mut instructions = load_params(
        &names,
        &[
            ParamLoad::Ptr(1),
            ParamLoad::Ptr(2),
            ParamLoad::Ptr(3),
            ParamLoad::Ptr(4),
            ParamLoad::U32(1),
            ParamLoad::U32(2),
            ParamLoad::U32(3),
            ParamLoad::U32(4),
            ParamLoad::U32(5),
            ParamLoad::U32(6),
            ParamLoad::U32(7),
            ParamLoad::U32(8),
            ParamLoad::U32(9),
            ParamLoad::U32(10),
            ParamLoad::U32(11),
            ParamLoad::U32(12),
            ParamLoad::U32(13),
            ParamLoad::U32(14),
            ParamLoad::U32(15),
            ParamLoad::U32(16),
            ParamLoad::U32(17),
        ],
    );
    instructions.extend(linear_tid(18, 19, 20, 18, false));
    instructions.extend([
        PtxInstruction::MulLoU32 { destination: b32(19), left: b32(1), right: U32Value::Reg(b32(5)) },
        PtxInstruction::MulLoU32 { destination: b32(19), left: b32(19), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MulLoU32 { destination: b32(19), left: b32(19), right: U32Value::Reg(b32(9)) },
    ]);
    instructions.extend(bounds_guard(18, U32Value::Reg(b32(19)), 1, &done));
    instructions.extend([
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
        PtxInstruction::FmaRnF32 {
            destination: f32(1),
            a: F32Value::Reg(f32(2)),
            b: F32Value::Reg(f32(3)),
            c: F32Value::Reg(f32(1)),
        },
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
    ]);
    Entry { name, parameters, registers: regs(3, 38, 9, 4), instructions }
}
