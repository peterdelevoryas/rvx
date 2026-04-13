use anyhow::{Context, Result, bail};

use crate::Ram;

#[derive(Debug, Clone, Copy)]
pub struct ElfLoadInfo {
    pub entry: u64,
    pub start: u64,
    pub end: u64,
}

pub fn load_binary(bytes: &[u8], addr: u64, ram: &Ram) -> Result<ElfLoadInfo> {
    ram.load(addr, bytes)?;
    Ok(ElfLoadInfo {
        entry: addr,
        start: addr,
        end: addr + bytes.len() as u64,
    })
}

pub fn load_elf64(bytes: &[u8], ram: &Ram) -> Result<ElfLoadInfo> {
    if bytes.len() < 64 {
        bail!("elf image too small");
    }
    if &bytes[0..4] != b"\x7fELF" {
        bail!("not an elf image");
    }
    if bytes[4] != 2 || bytes[5] != 1 {
        bail!("expected 64-bit little-endian elf");
    }

    let entry = read_u64(bytes, 24)?;
    let phoff = read_u64(bytes, 32)? as usize;
    let phentsize = read_u16(bytes, 54)? as usize;
    let phnum = read_u16(bytes, 56)? as usize;

    let mut start = u64::MAX;
    let mut end = 0u64;
    for index in 0..phnum {
        let offset = phoff + index * phentsize;
        let p_type = read_u32(bytes, offset)?;
        if p_type != 1 {
            continue;
        }
        let p_offset = read_u64(bytes, offset + 8)? as usize;
        let p_vaddr = read_u64(bytes, offset + 16)?;
        let p_filesz = read_u64(bytes, offset + 32)? as usize;
        let p_memsz = read_u64(bytes, offset + 40)? as usize;
        let segment = bytes
            .get(p_offset..p_offset + p_filesz)
            .context("elf segment out of bounds")?;
        ram.load(p_vaddr, segment)?;
        if p_memsz > p_filesz {
            let zero_len = p_memsz - p_filesz;
            let zeros = vec![0; zero_len];
            ram.load(p_vaddr + p_filesz as u64, &zeros)?;
        }
        start = start.min(p_vaddr);
        end = end.max(p_vaddr + p_memsz as u64);
    }

    if start == u64::MAX {
        bail!("elf image had no PT_LOAD segments");
    }

    Ok(ElfLoadInfo { entry, start, end })
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let data = bytes
        .get(offset..offset + 2)
        .context("short read while parsing elf u16")?;
    Ok(u16::from_le_bytes([data[0], data[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let data = bytes
        .get(offset..offset + 4)
        .context("short read while parsing elf u32")?;
    Ok(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let data = bytes
        .get(offset..offset + 8)
        .context("short read while parsing elf u64")?;
    Ok(u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]))
}
