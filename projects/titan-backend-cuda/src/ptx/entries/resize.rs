use super::{
    super::ast::{Entry, Identifier, PtxInstruction, U32Value},
    params::{ParamLoad, buffer_u32_params, load_params, named_params, regs},
    prologue::{bounds_guard, done_label, entry_tail, f32_byte_offsets, linear_f32_load, linear_f32_store, linear_tid, ptr_plus_offset},
    regs::{b32, b64},
};

pub(super) fn resize_nearest2d_f32(name: Identifier) -> Entry {
    let names = named_params::<8>(&name);
    let parameters = buffer_u32_params(&names, 2);
    let done = done_label(&name);
    let mut instructions = load_params(
        &names,
        &[
            ParamLoad::Ptr(1),
            ParamLoad::Ptr(2),
            ParamLoad::U32(1),
            ParamLoad::U32(2),
            ParamLoad::U32(3),
            ParamLoad::U32(4),
            ParamLoad::U32(5),
            ParamLoad::U32(6),
        ],
    );
    instructions.extend(linear_tid(7, 8, 9, 10, true));
    instructions.extend([
        PtxInstruction::MulLoU32 { destination: b32(11), left: b32(1), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MulLoU32 { destination: b32(11), left: b32(11), right: U32Value::Reg(b32(5)) },
        PtxInstruction::MulLoU32 { destination: b32(11), left: b32(11), right: U32Value::Reg(b32(6)) },
    ]);
    instructions.extend(bounds_guard(10, U32Value::Reg(b32(11)), 1, &done));
    instructions.extend([
        PtxInstruction::RemU32 { destination: b32(12), left: b32(10), right: U32Value::Reg(b32(6)) },
        PtxInstruction::DivU32 { destination: b32(13), left: b32(10), right: U32Value::Reg(b32(6)) },
        PtxInstruction::RemU32 { destination: b32(14), left: b32(13), right: U32Value::Reg(b32(5)) },
        PtxInstruction::DivU32 { destination: b32(15), left: b32(13), right: U32Value::Reg(b32(5)) },
        PtxInstruction::RemU32 { destination: b32(16), left: b32(15), right: U32Value::Reg(b32(2)) },
        PtxInstruction::DivU32 { destination: b32(17), left: b32(15), right: U32Value::Reg(b32(2)) },
        PtxInstruction::MulLoU32 { destination: b32(18), left: b32(14), right: U32Value::Reg(b32(3)) },
        PtxInstruction::DivU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
        PtxInstruction::MulLoU32 { destination: b32(19), left: b32(12), right: U32Value::Reg(b32(4)) },
        PtxInstruction::DivU32 { destination: b32(19), left: b32(19), right: U32Value::Reg(b32(6)) },
        PtxInstruction::MulLoU32 { destination: b32(20), left: b32(17), right: U32Value::Reg(b32(2)) },
        PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(16)) },
        PtxInstruction::MulLoU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(3)) },
        PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(18)) },
        PtxInstruction::MulLoU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(4)) },
        PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(19)) },
    ]);
    instructions.extend(f32_byte_offsets(&[(20, 3), (10, 4)]));
    instructions.extend(ptr_plus_offset(3, &[(1, 5)]));
    instructions.extend(ptr_plus_offset(4, &[(2, 6)]));
    instructions.push(linear_f32_load(5, 1));
    instructions.push(linear_f32_store(6, 1));
    instructions.extend(entry_tail(&done));
    Entry { name, parameters, registers: regs(2, 21, 7, 2), instructions }
}
