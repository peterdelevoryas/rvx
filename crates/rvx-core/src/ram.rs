use std::sync::RwLock;

use anyhow::{Result, bail};

#[derive(Debug)]
pub struct Ram {
    base: u64,
    data: RwLock<Vec<u8>>,
}

impl Ram {
    pub fn new(base: u64, size: u64) -> Result<Self> {
        let size: usize = size
            .try_into()
            .map_err(|_| anyhow::anyhow!("ram size does not fit host usize"))?;
        Ok(Self {
            base,
            data: RwLock::new(vec![0; size]),
        })
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn len(&self) -> u64 {
        self.data.read().unwrap().len() as u64
    }

    pub fn contains(&self, addr: u64, size: usize) -> bool {
        let end = match addr.checked_add(size as u64) {
            Some(end) => end,
            None => return false,
        };
        addr >= self.base && end <= self.base + self.len()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.data.read().unwrap().as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.data.read().unwrap().as_ptr() as *mut u8
    }

    pub fn load(&self, guest_addr: u64, bytes: &[u8]) -> Result<()> {
        if !self.contains(guest_addr, bytes.len()) {
            bail!("ram load out of range at 0x{guest_addr:016x}");
        }
        let offset = (guest_addr - self.base) as usize;
        let mut data = self.data.write().unwrap();
        data[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
        if !self.contains(addr, dst.len()) {
            bail!("ram read out of range at 0x{addr:016x}");
        }
        let offset = (addr - self.base) as usize;
        let data = self.data.read().unwrap();
        dst.copy_from_slice(&data[offset..offset + dst.len()]);
        Ok(())
    }

    pub fn write(&self, addr: u64, src: &[u8]) -> Result<()> {
        if !self.contains(addr, src.len()) {
            bail!("ram write out of range at 0x{addr:016x}");
        }
        let offset = (addr - self.base) as usize;
        let mut data = self.data.write().unwrap();
        data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    pub fn read_u8(&self, addr: u64) -> Result<u8> {
        let mut buf = [0; 1];
        self.read(addr, &mut buf)?;
        Ok(buf[0])
    }

    pub fn read_u16(&self, addr: u64) -> Result<u16> {
        let mut buf = [0; 2];
        self.read(addr, &mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    pub fn read_u32(&self, addr: u64) -> Result<u32> {
        let mut buf = [0; 4];
        self.read(addr, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn read_u64(&self, addr: u64) -> Result<u64> {
        let mut buf = [0; 8];
        self.read(addr, &mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    pub fn write_u8(&self, addr: u64, value: u8) -> Result<()> {
        self.write(addr, &[value])
    }

    pub fn write_u16(&self, addr: u64, value: u16) -> Result<()> {
        self.write(addr, &value.to_le_bytes())
    }

    pub fn write_u32(&self, addr: u64, value: u32) -> Result<()> {
        self.write(addr, &value.to_le_bytes())
    }

    pub fn write_u64(&self, addr: u64, value: u64) -> Result<()> {
        self.write(addr, &value.to_le_bytes())
    }
}
