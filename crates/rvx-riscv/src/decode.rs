use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    Byte,
    Half,
    Word,
    Double,
    ByteUnsigned,
    HalfUnsigned,
    WordUnsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    Byte,
    Half,
    Word,
    Double,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatLoadKind {
    Word,
    Double,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatStoreKind {
    Word,
    Double,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchKind {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    LrW,
    ScW,
    AmoswapW,
    AmoaddW,
    AmoxorW,
    AmoandW,
    AmoorW,
    AmominW,
    AmomaxW,
    AmominuW,
    AmomaxuW,
    LrD,
    ScD,
    AmoswapD,
    AmoaddD,
    AmoxorD,
    AmoandD,
    AmoorD,
    AmominD,
    AmomaxD,
    AmominuD,
    AmomaxuD,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedInstruction {
    Lui {
        rd: u8,
        imm: i64,
    },
    Auipc {
        rd: u8,
        imm: i64,
    },
    Jal {
        rd: u8,
        imm: i64,
    },
    Jalr {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Branch {
        kind: BranchKind,
        rs1: u8,
        rs2: u8,
        imm: i64,
    },
    Load {
        kind: LoadKind,
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    FloatLoad {
        kind: FloatLoadKind,
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Store {
        kind: StoreKind,
        rs1: u8,
        rs2: u8,
        imm: i64,
    },
    FloatStore {
        kind: FloatStoreKind,
        rs1: u8,
        rs2: u8,
        imm: i64,
    },
    Addi {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Slti {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Sltiu {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Xori {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Ori {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Andi {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Slli {
        rd: u8,
        rs1: u8,
        shamt: u8,
    },
    Srli {
        rd: u8,
        rs1: u8,
        shamt: u8,
    },
    Srai {
        rd: u8,
        rs1: u8,
        shamt: u8,
    },
    Addiw {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Slliw {
        rd: u8,
        rs1: u8,
        shamt: u8,
    },
    Srliw {
        rd: u8,
        rs1: u8,
        shamt: u8,
    },
    Sraiw {
        rd: u8,
        rs1: u8,
        shamt: u8,
    },
    Add {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sub {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sll {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Slt {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sltu {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Xor {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Srl {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sra {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Or {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    And {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Addw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Subw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sllw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Srlw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sraw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mul {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mulh {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mulhsu {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mulhu {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Div {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Divu {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Rem {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Remu {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mulw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Divw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Divuw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Remw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Remuw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fence,
    FenceI,
    Ecall,
    Ebreak,
    Wfi,
    Mret,
    Sret,
    SfenceVma {
        rs1: u8,
        rs2: u8,
    },
    Csrrw {
        rd: u8,
        rs1: u8,
        csr: u16,
    },
    Csrrs {
        rd: u8,
        rs1: u8,
        csr: u16,
    },
    Csrrc {
        rd: u8,
        rs1: u8,
        csr: u16,
    },
    Csrrwi {
        rd: u8,
        zimm: u8,
        csr: u16,
    },
    Csrrsi {
        rd: u8,
        zimm: u8,
        csr: u16,
    },
    Csrrci {
        rd: u8,
        zimm: u8,
        csr: u16,
    },
    Atomic {
        op: AtomicOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    },
}

pub fn decode(word: u32) -> Result<DecodedInstruction> {
    let opcode = word & 0x7f;
    let rd = rd(word);
    let funct3 = funct3(word);
    let rs1 = rs1(word);
    let rs2 = rs2(word);
    let funct7 = funct7(word);

    let insn = match opcode {
        0x37 => DecodedInstruction::Lui {
            rd,
            imm: u_imm(word),
        },
        0x17 => DecodedInstruction::Auipc {
            rd,
            imm: u_imm(word),
        },
        0x6f => DecodedInstruction::Jal {
            rd,
            imm: j_imm(word),
        },
        0x67 if funct3 == 0 => DecodedInstruction::Jalr {
            rd,
            rs1,
            imm: i_imm(word),
        },
        0x63 => DecodedInstruction::Branch {
            kind: match funct3 {
                0 => BranchKind::Eq,
                1 => BranchKind::Ne,
                4 => BranchKind::Lt,
                5 => BranchKind::Ge,
                6 => BranchKind::Ltu,
                7 => BranchKind::Geu,
                _ => bail!("unsupported branch funct3 {funct3}"),
            },
            rs1,
            rs2,
            imm: b_imm(word),
        },
        0x03 => DecodedInstruction::Load {
            kind: match funct3 {
                0 => LoadKind::Byte,
                1 => LoadKind::Half,
                2 => LoadKind::Word,
                3 => LoadKind::Double,
                4 => LoadKind::ByteUnsigned,
                5 => LoadKind::HalfUnsigned,
                6 => LoadKind::WordUnsigned,
                _ => bail!("unsupported load funct3 {funct3}"),
            },
            rd,
            rs1,
            imm: i_imm(word),
        },
        0x07 => DecodedInstruction::FloatLoad {
            kind: match funct3 {
                2 => FloatLoadKind::Word,
                3 => FloatLoadKind::Double,
                _ => bail!("unsupported fp load funct3 {funct3}"),
            },
            rd,
            rs1,
            imm: i_imm(word),
        },
        0x23 => DecodedInstruction::Store {
            kind: match funct3 {
                0 => StoreKind::Byte,
                1 => StoreKind::Half,
                2 => StoreKind::Word,
                3 => StoreKind::Double,
                _ => bail!("unsupported store funct3 {funct3}"),
            },
            rs1,
            rs2,
            imm: s_imm(word),
        },
        0x27 => DecodedInstruction::FloatStore {
            kind: match funct3 {
                2 => FloatStoreKind::Word,
                3 => FloatStoreKind::Double,
                _ => bail!("unsupported fp store funct3 {funct3}"),
            },
            rs1,
            rs2,
            imm: s_imm(word),
        },
        0x13 => match funct3 {
            0 => DecodedInstruction::Addi {
                rd,
                rs1,
                imm: i_imm(word),
            },
            2 => DecodedInstruction::Slti {
                rd,
                rs1,
                imm: i_imm(word),
            },
            3 => DecodedInstruction::Sltiu {
                rd,
                rs1,
                imm: i_imm(word),
            },
            4 => DecodedInstruction::Xori {
                rd,
                rs1,
                imm: i_imm(word),
            },
            6 => DecodedInstruction::Ori {
                rd,
                rs1,
                imm: i_imm(word),
            },
            7 => DecodedInstruction::Andi {
                rd,
                rs1,
                imm: i_imm(word),
            },
            1 if ((word >> 26) & 0x3f) == 0x00 => DecodedInstruction::Slli {
                rd,
                rs1,
                shamt: ((word >> 20) & 0x3f) as u8,
            },
            5 if ((word >> 26) & 0x3f) == 0x00 => DecodedInstruction::Srli {
                rd,
                rs1,
                shamt: ((word >> 20) & 0x3f) as u8,
            },
            5 if ((word >> 26) & 0x3f) == 0x10 => DecodedInstruction::Srai {
                rd,
                rs1,
                shamt: ((word >> 20) & 0x3f) as u8,
            },
            _ => bail!("unsupported op-imm"),
        },
        0x1b => match funct3 {
            0 => DecodedInstruction::Addiw {
                rd,
                rs1,
                imm: i_imm(word),
            },
            1 => DecodedInstruction::Slliw {
                rd,
                rs1,
                shamt: ((word >> 20) & 0x1f) as u8,
            },
            5 if funct7 == 0x00 => DecodedInstruction::Srliw {
                rd,
                rs1,
                shamt: ((word >> 20) & 0x1f) as u8,
            },
            5 if funct7 == 0x20 => DecodedInstruction::Sraiw {
                rd,
                rs1,
                shamt: ((word >> 20) & 0x1f) as u8,
            },
            _ => bail!("unsupported op-imm-32"),
        },
        0x33 => match (funct7, funct3) {
            (0x00, 0x0) => DecodedInstruction::Add { rd, rs1, rs2 },
            (0x20, 0x0) => DecodedInstruction::Sub { rd, rs1, rs2 },
            (0x00, 0x1) => DecodedInstruction::Sll { rd, rs1, rs2 },
            (0x00, 0x2) => DecodedInstruction::Slt { rd, rs1, rs2 },
            (0x00, 0x3) => DecodedInstruction::Sltu { rd, rs1, rs2 },
            (0x00, 0x4) => DecodedInstruction::Xor { rd, rs1, rs2 },
            (0x00, 0x5) => DecodedInstruction::Srl { rd, rs1, rs2 },
            (0x20, 0x5) => DecodedInstruction::Sra { rd, rs1, rs2 },
            (0x00, 0x6) => DecodedInstruction::Or { rd, rs1, rs2 },
            (0x00, 0x7) => DecodedInstruction::And { rd, rs1, rs2 },
            (0x01, 0x0) => DecodedInstruction::Mul { rd, rs1, rs2 },
            (0x01, 0x1) => DecodedInstruction::Mulh { rd, rs1, rs2 },
            (0x01, 0x2) => DecodedInstruction::Mulhsu { rd, rs1, rs2 },
            (0x01, 0x3) => DecodedInstruction::Mulhu { rd, rs1, rs2 },
            (0x01, 0x4) => DecodedInstruction::Div { rd, rs1, rs2 },
            (0x01, 0x5) => DecodedInstruction::Divu { rd, rs1, rs2 },
            (0x01, 0x6) => DecodedInstruction::Rem { rd, rs1, rs2 },
            (0x01, 0x7) => DecodedInstruction::Remu { rd, rs1, rs2 },
            _ => bail!("unsupported op"),
        },
        0x3b => match (funct7, funct3) {
            (0x00, 0x0) => DecodedInstruction::Addw { rd, rs1, rs2 },
            (0x20, 0x0) => DecodedInstruction::Subw { rd, rs1, rs2 },
            (0x00, 0x1) => DecodedInstruction::Sllw { rd, rs1, rs2 },
            (0x00, 0x5) => DecodedInstruction::Srlw { rd, rs1, rs2 },
            (0x20, 0x5) => DecodedInstruction::Sraw { rd, rs1, rs2 },
            (0x01, 0x0) => DecodedInstruction::Mulw { rd, rs1, rs2 },
            (0x01, 0x4) => DecodedInstruction::Divw { rd, rs1, rs2 },
            (0x01, 0x5) => DecodedInstruction::Divuw { rd, rs1, rs2 },
            (0x01, 0x6) => DecodedInstruction::Remw { rd, rs1, rs2 },
            (0x01, 0x7) => DecodedInstruction::Remuw { rd, rs1, rs2 },
            _ => bail!("unsupported op-32"),
        },
        0x0f if funct3 == 0 => DecodedInstruction::Fence,
        0x0f if funct3 == 1 => DecodedInstruction::FenceI,
        0x73 => decode_system(word, rd, rs1, funct3)?,
        0x2f => decode_atomic(word, rd, rs1, rs2)?,
        _ => bail!("unsupported opcode 0x{opcode:02x}"),
    };
    Ok(insn)
}

pub fn decode_compressed(word: u16) -> Result<DecodedInstruction> {
    let quadrant = word & 0x3;
    let funct3 = (word >> 13) & 0x7;
    Ok(match (quadrant, funct3) {
        (0b00, 0b000) => {
            let imm = c_addi4spn_imm(word);
            if imm == 0 {
                bail!("reserved c.addi4spn");
            }
            DecodedInstruction::Addi {
                rd: creg(((word >> 2) & 0x7) as u8),
                rs1: 2,
                imm,
            }
        }
        (0b00, 0b001) => DecodedInstruction::FloatLoad {
            kind: FloatLoadKind::Double,
            rd: creg(((word >> 2) & 0x7) as u8),
            rs1: creg(((word >> 7) & 0x7) as u8),
            imm: c_ld_imm(word) as i64,
        },
        (0b00, 0b010) => DecodedInstruction::Load {
            kind: LoadKind::Word,
            rd: creg(((word >> 2) & 0x7) as u8),
            rs1: creg(((word >> 7) & 0x7) as u8),
            imm: c_lw_imm(word) as i64,
        },
        (0b00, 0b011) => DecodedInstruction::Load {
            kind: LoadKind::Double,
            rd: creg(((word >> 2) & 0x7) as u8),
            rs1: creg(((word >> 7) & 0x7) as u8),
            imm: c_ld_imm(word) as i64,
        },
        (0b00, 0b101) => DecodedInstruction::FloatStore {
            kind: FloatStoreKind::Double,
            rs1: creg(((word >> 7) & 0x7) as u8),
            rs2: creg(((word >> 2) & 0x7) as u8),
            imm: c_ld_imm(word) as i64,
        },
        (0b00, 0b110) => DecodedInstruction::Store {
            kind: StoreKind::Word,
            rs1: creg(((word >> 7) & 0x7) as u8),
            rs2: creg(((word >> 2) & 0x7) as u8),
            imm: c_lw_imm(word) as i64,
        },
        (0b00, 0b111) => DecodedInstruction::Store {
            kind: StoreKind::Double,
            rs1: creg(((word >> 7) & 0x7) as u8),
            rs2: creg(((word >> 2) & 0x7) as u8),
            imm: c_ld_imm(word) as i64,
        },
        (0b01, 0b000) => DecodedInstruction::Addi {
            rd: ((word >> 7) & 0x1f) as u8,
            rs1: ((word >> 7) & 0x1f) as u8,
            imm: c_nzimm_6(word),
        },
        (0b01, 0b001) => {
            let rd = ((word >> 7) & 0x1f) as u8;
            if rd == 0 {
                bail!("reserved c.addiw");
            }
            DecodedInstruction::Addiw {
                rd,
                rs1: rd,
                imm: c_nzimm_6(word),
            }
        }
        (0b01, 0b010) => DecodedInstruction::Addi {
            rd: ((word >> 7) & 0x1f) as u8,
            rs1: 0,
            imm: c_nzimm_6(word),
        },
        (0b01, 0b011) => {
            let rd = ((word >> 7) & 0x1f) as u8;
            if rd == 2 {
                let imm = c_addi16sp_imm(word);
                if imm == 0 {
                    bail!("reserved c.addi16sp");
                }
                DecodedInstruction::Addi { rd: 2, rs1: 2, imm }
            } else if rd != 0 {
                DecodedInstruction::Lui {
                    rd,
                    imm: c_lui_imm(word),
                }
            } else {
                bail!("reserved c.lui");
            }
        }
        (0b01, 0b100) => {
            let rd = creg(((word >> 7) & 0x7) as u8);
            match ((word >> 10) & 0x3, (word >> 12) & 0x1, (word >> 5) & 0x3) {
                (0b00, _, _) => DecodedInstruction::Srli {
                    rd,
                    rs1: rd,
                    shamt: c_shamt(word),
                },
                (0b01, _, _) => DecodedInstruction::Srai {
                    rd,
                    rs1: rd,
                    shamt: c_shamt(word),
                },
                (0b10, _, _) => DecodedInstruction::Andi {
                    rd,
                    rs1: rd,
                    imm: c_nzimm_6(word),
                },
                (0b11, 0, 0b00) => DecodedInstruction::Sub {
                    rd,
                    rs1: rd,
                    rs2: creg(((word >> 2) & 0x7) as u8),
                },
                (0b11, 0, 0b01) => DecodedInstruction::Xor {
                    rd,
                    rs1: rd,
                    rs2: creg(((word >> 2) & 0x7) as u8),
                },
                (0b11, 0, 0b10) => DecodedInstruction::Or {
                    rd,
                    rs1: rd,
                    rs2: creg(((word >> 2) & 0x7) as u8),
                },
                (0b11, 0, 0b11) => DecodedInstruction::And {
                    rd,
                    rs1: rd,
                    rs2: creg(((word >> 2) & 0x7) as u8),
                },
                (0b11, 1, 0b00) => DecodedInstruction::Subw {
                    rd,
                    rs1: rd,
                    rs2: creg(((word >> 2) & 0x7) as u8),
                },
                (0b11, 1, 0b01) => DecodedInstruction::Addw {
                    rd,
                    rs1: rd,
                    rs2: creg(((word >> 2) & 0x7) as u8),
                },
                _ => bail!("unsupported c.op"),
            }
        }
        (0b01, 0b101) => DecodedInstruction::Jal {
            rd: 0,
            imm: c_jump_imm(word),
        },
        (0b01, 0b110) => DecodedInstruction::Branch {
            kind: BranchKind::Eq,
            rs1: creg(((word >> 7) & 0x7) as u8),
            rs2: 0,
            imm: c_branch_imm(word),
        },
        (0b01, 0b111) => DecodedInstruction::Branch {
            kind: BranchKind::Ne,
            rs1: creg(((word >> 7) & 0x7) as u8),
            rs2: 0,
            imm: c_branch_imm(word),
        },
        (0b10, 0b000) => {
            let rd = ((word >> 7) & 0x1f) as u8;
            if rd == 0 {
                bail!("reserved c.slli");
            }
            DecodedInstruction::Slli {
                rd,
                rs1: rd,
                shamt: c_shamt(word),
            }
        }
        (0b10, 0b001) => DecodedInstruction::FloatLoad {
            kind: FloatLoadKind::Double,
            rd: ((word >> 7) & 0x1f) as u8,
            rs1: 2,
            imm: c_ldsp_imm(word) as i64,
        },
        (0b10, 0b010) => {
            let rd = ((word >> 7) & 0x1f) as u8;
            if rd == 0 {
                bail!("reserved c.lwsp");
            }
            DecodedInstruction::Load {
                kind: LoadKind::Word,
                rd,
                rs1: 2,
                imm: c_lwsp_imm(word) as i64,
            }
        }
        (0b10, 0b011) => {
            let rd = ((word >> 7) & 0x1f) as u8;
            if rd == 0 {
                bail!("reserved c.ldsp");
            }
            DecodedInstruction::Load {
                kind: LoadKind::Double,
                rd,
                rs1: 2,
                imm: c_ldsp_imm(word) as i64,
            }
        }
        (0b10, 0b100) => {
            let rd = ((word >> 7) & 0x1f) as u8;
            let rs2 = ((word >> 2) & 0x1f) as u8;
            match (((word >> 12) & 1) != 0, rs2 == 0, rd == 0) {
                (false, true, false) => DecodedInstruction::Jalr {
                    rd: 0,
                    rs1: rd,
                    imm: 0,
                },
                (false, false, false) => DecodedInstruction::Add { rd, rs1: 0, rs2 },
                (true, true, true) => DecodedInstruction::Ebreak,
                (true, true, false) => DecodedInstruction::Jalr {
                    rd: 1,
                    rs1: rd,
                    imm: 0,
                },
                (true, false, false) => DecodedInstruction::Add { rd, rs1: rd, rs2 },
                _ => bail!("reserved c.jr/c.mv/c.add"),
            }
        }
        (0b10, 0b101) => DecodedInstruction::FloatStore {
            kind: FloatStoreKind::Double,
            rs1: 2,
            rs2: ((word >> 2) & 0x1f) as u8,
            imm: c_sdsp_imm(word) as i64,
        },
        (0b10, 0b110) => DecodedInstruction::Store {
            kind: StoreKind::Word,
            rs1: 2,
            rs2: ((word >> 2) & 0x1f) as u8,
            imm: c_swsp_imm(word) as i64,
        },
        (0b10, 0b111) => DecodedInstruction::Store {
            kind: StoreKind::Double,
            rs1: 2,
            rs2: ((word >> 2) & 0x1f) as u8,
            imm: c_sdsp_imm(word) as i64,
        },
        _ => bail!("unsupported compressed opcode 0b{quadrant:02b}/0b{funct3:03b}"),
    })
}

fn decode_system(word: u32, rd: u8, rs1: u8, funct3: u32) -> Result<DecodedInstruction> {
    let csr = ((word >> 20) & 0xfff) as u16;
    let imm12 = word >> 20;
    Ok(match funct3 {
        0 => match imm12 {
            0x000 => DecodedInstruction::Ecall,
            0x001 => DecodedInstruction::Ebreak,
            0x105 => DecodedInstruction::Wfi,
            0x102 => DecodedInstruction::Sret,
            0x302 => DecodedInstruction::Mret,
            _ if (imm12 & 0xfe0) == 0x120 => DecodedInstruction::SfenceVma {
                rs1,
                rs2: rs2(word),
            },
            _ => bail!("unsupported system imm12 0x{imm12:03x}"),
        },
        1 => DecodedInstruction::Csrrw { rd, rs1, csr },
        2 => DecodedInstruction::Csrrs { rd, rs1, csr },
        3 => DecodedInstruction::Csrrc { rd, rs1, csr },
        5 => DecodedInstruction::Csrrwi { rd, zimm: rs1, csr },
        6 => DecodedInstruction::Csrrsi { rd, zimm: rs1, csr },
        7 => DecodedInstruction::Csrrci { rd, zimm: rs1, csr },
        _ => bail!("unsupported system funct3 {funct3}"),
    })
}

fn decode_atomic(word: u32, rd: u8, rs1: u8, rs2: u8) -> Result<DecodedInstruction> {
    let funct5 = (word >> 27) & 0x1f;
    let aq = ((word >> 26) & 1) != 0;
    let rl = ((word >> 25) & 1) != 0;
    let funct3 = funct3(word);
    let op = match (funct5, funct3) {
        (0x02, 0x2) => AtomicOp::LrW,
        (0x03, 0x2) => AtomicOp::ScW,
        (0x01, 0x2) => AtomicOp::AmoswapW,
        (0x00, 0x2) => AtomicOp::AmoaddW,
        (0x04, 0x2) => AtomicOp::AmoxorW,
        (0x0c, 0x2) => AtomicOp::AmoandW,
        (0x08, 0x2) => AtomicOp::AmoorW,
        (0x10, 0x2) => AtomicOp::AmominW,
        (0x14, 0x2) => AtomicOp::AmomaxW,
        (0x18, 0x2) => AtomicOp::AmominuW,
        (0x1c, 0x2) => AtomicOp::AmomaxuW,
        (0x02, 0x3) => AtomicOp::LrD,
        (0x03, 0x3) => AtomicOp::ScD,
        (0x01, 0x3) => AtomicOp::AmoswapD,
        (0x00, 0x3) => AtomicOp::AmoaddD,
        (0x04, 0x3) => AtomicOp::AmoxorD,
        (0x0c, 0x3) => AtomicOp::AmoandD,
        (0x08, 0x3) => AtomicOp::AmoorD,
        (0x10, 0x3) => AtomicOp::AmominD,
        (0x14, 0x3) => AtomicOp::AmomaxD,
        (0x18, 0x3) => AtomicOp::AmominuD,
        (0x1c, 0x3) => AtomicOp::AmomaxuD,
        _ => bail!("unsupported atomic op"),
    };
    Ok(DecodedInstruction::Atomic {
        op,
        rd,
        rs1,
        rs2,
        aq,
        rl,
    })
}

fn rd(word: u32) -> u8 {
    ((word >> 7) & 0x1f) as u8
}

fn creg(value: u8) -> u8 {
    value + 8
}

fn funct3(word: u32) -> u32 {
    (word >> 12) & 0x7
}

fn rs1(word: u32) -> u8 {
    ((word >> 15) & 0x1f) as u8
}

fn rs2(word: u32) -> u8 {
    ((word >> 20) & 0x1f) as u8
}

fn funct7(word: u32) -> u32 {
    (word >> 25) & 0x7f
}

fn sign_extend(value: u64, bits: u8) -> i64 {
    let shift = 64 - bits as u32;
    ((value << shift) as i64) >> shift
}

fn c_nzimm_6(word: u16) -> i64 {
    sign_extend(
        (((word >> 12) & 1) as u64) << 5 | ((word >> 2) & 0x1f) as u64,
        6,
    )
}

fn c_shamt(word: u16) -> u8 {
    ((((word >> 12) & 1) << 5) | ((word >> 2) & 0x1f)) as u8
}

fn c_addi4spn_imm(word: u16) -> i64 {
    ((((word >> 6) & 0x1) << 2)
        | (((word >> 5) & 0x1) << 3)
        | (((word >> 11) & 0x3) << 4)
        | (((word >> 7) & 0xf) << 6)) as i64
}

fn c_lw_imm(word: u16) -> u64 {
    (((word >> 6) as u64 & 0x1) << 2)
        | (((word >> 10) as u64 & 0x7) << 3)
        | (((word >> 5) as u64 & 0x1) << 6)
}

fn c_ld_imm(word: u16) -> u64 {
    (((word >> 10) as u64 & 0x7) << 3) | (((word >> 5) as u64 & 0x3) << 6)
}

fn c_addi16sp_imm(word: u16) -> i64 {
    sign_extend(
        ((((word >> 12) & 1) as u64) << 9)
            | ((((word >> 6) & 1) as u64) << 4)
            | ((((word >> 5) & 1) as u64) << 6)
            | ((((word >> 3) & 0x3) as u64) << 7)
            | ((((word >> 2) & 1) as u64) << 5),
        10,
    )
}

fn c_lui_imm(word: u16) -> i64 {
    c_nzimm_6(word) << 12
}

fn c_jump_imm(word: u16) -> i64 {
    sign_extend(
        ((((word >> 12) & 1) as u64) << 11)
            | ((((word >> 11) & 1) as u64) << 4)
            | ((((word >> 9) & 0x3) as u64) << 8)
            | ((((word >> 8) & 1) as u64) << 10)
            | ((((word >> 7) & 1) as u64) << 6)
            | ((((word >> 6) & 1) as u64) << 7)
            | ((((word >> 3) & 0x7) as u64) << 1)
            | ((((word >> 2) & 1) as u64) << 5),
        12,
    )
}

fn c_branch_imm(word: u16) -> i64 {
    sign_extend(
        ((((word >> 12) & 1) as u64) << 8)
            | ((((word >> 10) & 0x3) as u64) << 3)
            | ((((word >> 5) & 0x3) as u64) << 6)
            | ((((word >> 3) & 0x3) as u64) << 1)
            | ((((word >> 2) & 1) as u64) << 5),
        9,
    )
}

fn c_lwsp_imm(word: u16) -> u64 {
    ((((word >> 4) & 0x7) as u64) << 2)
        | ((((word >> 12) & 1) as u64) << 5)
        | ((((word >> 2) & 0x3) as u64) << 6)
}

fn c_ldsp_imm(word: u16) -> u64 {
    ((((word >> 5) & 0x3) as u64) << 3)
        | ((((word >> 12) & 1) as u64) << 5)
        | ((((word >> 2) & 0x7) as u64) << 6)
}

fn c_swsp_imm(word: u16) -> u64 {
    ((((word >> 9) & 0xf) as u64) << 2) | ((((word >> 7) & 0x3) as u64) << 6)
}

fn c_sdsp_imm(word: u16) -> u64 {
    ((((word >> 10) & 0x7) as u64) << 3) | ((((word >> 7) & 0x7) as u64) << 6)
}

fn i_imm(word: u32) -> i64 {
    sign_extend((word >> 20) as u64, 12)
}

fn s_imm(word: u32) -> i64 {
    let value = (((word >> 25) << 5) | ((word >> 7) & 0x1f)) as u64;
    sign_extend(value, 12)
}

fn b_imm(word: u32) -> i64 {
    let value = (((word >> 31) << 12)
        | (((word >> 7) & 1) << 11)
        | (((word >> 25) & 0x3f) << 5)
        | (((word >> 8) & 0x0f) << 1)) as u64;
    sign_extend(value, 13)
}

fn u_imm(word: u32) -> i64 {
    sign_extend((word & 0xfffff000) as u64, 32)
}

fn j_imm(word: u32) -> i64 {
    let value = (((word >> 31) << 20)
        | (((word >> 12) & 0xff) << 12)
        | (((word >> 20) & 1) << 11)
        | (((word >> 21) & 0x3ff) << 1)) as u64;
    sign_extend(value, 21)
}

#[cfg(test)]
mod tests {
    use super::{DecodedInstruction, decode};

    #[test]
    fn decode_addi() {
        let insn = decode(0x0010_8093).unwrap();
        assert_eq!(
            insn,
            DecodedInstruction::Addi {
                rd: 1,
                rs1: 1,
                imm: 1
            }
        );
    }

    #[test]
    fn decode_mret() {
        let insn = decode(0x3020_0073).unwrap();
        assert_eq!(insn, DecodedInstruction::Mret);
    }
}
