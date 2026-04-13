use std::ops::Range;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use anyhow::{Result, bail};

use crate::Ram;

pub trait MmioDevice: Send + Sync {
    fn read(&mut self, offset: u64, size: u8) -> u64;
    fn write(&mut self, offset: u64, size: u8, value: u64);
}

trait MmioHandle: Send + Sync {
    fn read(&self, offset: u64, size: u8) -> u64;
    fn write(&self, offset: u64, size: u8, value: u64);
}

impl<T> MmioHandle for Mutex<T>
where
    T: MmioDevice + Send,
{
    fn read(&self, offset: u64, size: u8) -> u64 {
        self.lock().unwrap().read(offset, size)
    }

    fn write(&self, offset: u64, size: u8, value: u64) {
        self.lock().unwrap().write(offset, size, value);
    }
}

#[derive(Clone)]
pub struct MmioMap {
    pub base: u64,
    pub size: u64,
    device: Arc<dyn MmioHandle>,
}

impl MmioMap {
    fn range(&self) -> Range<u64> {
        self.base..self.base + self.size
    }
}

pub struct Bus {
    ram: Ram,
    regions: Vec<MmioMap>,
    reservation_epoch: AtomicU64,
    atomic_lock: Mutex<()>,
    concurrent_access: AtomicBool,
}

impl Bus {
    pub fn new(ram: Ram) -> Self {
        Self {
            ram,
            regions: Vec::new(),
            reservation_epoch: AtomicU64::new(1),
            atomic_lock: Mutex::new(()),
            concurrent_access: AtomicBool::new(false),
        }
    }

    pub fn ram(&self) -> &Ram {
        &self.ram
    }

    pub fn reservation_epoch(&self) -> u64 {
        self.reservation_epoch.load(Ordering::SeqCst)
    }

    pub fn invalidate_reservations(&self) -> u64 {
        self.reservation_epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub fn atomic_lock(&self) -> std::sync::MutexGuard<'_, ()> {
        self.atomic_lock.lock().unwrap()
    }

    pub fn set_concurrent_access(&self, enabled: bool) {
        self.concurrent_access.store(enabled, Ordering::Relaxed);
    }

    pub fn memory_barrier(&self) {
        if !self.concurrent_access.load(Ordering::Relaxed) {
            return;
        }
        std::sync::atomic::fence(Ordering::SeqCst);
        drop(self.atomic_lock());
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    pub fn map_mmio<T>(&mut self, base: u64, size: u64, device: Arc<Mutex<T>>)
    where
        T: MmioDevice + Send + 'static,
    {
        let device: Arc<dyn MmioHandle> = device;
        self.regions.push(MmioMap { base, size, device });
    }

    pub fn read_u8(&self, addr: u64) -> Result<u8> {
        self.read_le(addr, 1).map(|v| v as u8)
    }

    pub fn read_u16(&self, addr: u64) -> Result<u16> {
        self.read_le(addr, 2).map(|v| v as u16)
    }

    pub fn read_u32(&self, addr: u64) -> Result<u32> {
        self.read_le(addr, 4).map(|v| v as u32)
    }

    pub fn read_u64(&self, addr: u64) -> Result<u64> {
        self.read_le(addr, 8)
    }

    pub fn write_u8(&self, addr: u64, value: u8) -> Result<()> {
        let _guard = self.atomic_lock();
        self.write_le_locked(addr, 1, value as u64)
    }

    pub fn write_u16(&self, addr: u64, value: u16) -> Result<()> {
        let _guard = self.atomic_lock();
        self.write_le_locked(addr, 2, value as u64)
    }

    pub fn write_u32(&self, addr: u64, value: u32) -> Result<()> {
        let _guard = self.atomic_lock();
        self.write_le_locked(addr, 4, value as u64)
    }

    pub fn write_u64(&self, addr: u64, value: u64) -> Result<()> {
        let _guard = self.atomic_lock();
        self.write_le_locked(addr, 8, value)
    }

    pub fn read_u32_locked(
        &self,
        addr: u64,
        _guard: &std::sync::MutexGuard<'_, ()>,
    ) -> Result<u32> {
        self.read_u32(addr)
    }

    pub fn read_u64_locked(
        &self,
        addr: u64,
        _guard: &std::sync::MutexGuard<'_, ()>,
    ) -> Result<u64> {
        self.read_u64(addr)
    }

    pub fn write_u32_locked(
        &self,
        addr: u64,
        value: u32,
        _guard: &std::sync::MutexGuard<'_, ()>,
    ) -> Result<()> {
        self.write_le_locked(addr, 4, value as u64)
    }

    pub fn write_u64_locked(
        &self,
        addr: u64,
        value: u64,
        _guard: &std::sync::MutexGuard<'_, ()>,
    ) -> Result<()> {
        self.write_le_locked(addr, 8, value)
    }

    pub fn load(&self, addr: u64, bytes: &[u8]) -> Result<()> {
        self.ram.load(addr, bytes)
    }

    fn read_le(&self, addr: u64, size: u8) -> Result<u64> {
        if self.ram.contains(addr, size as usize) {
            return Ok(match size {
                1 => self.ram.read_u8(addr)? as u64,
                2 => self.ram.read_u16(addr)? as u64,
                4 => self.ram.read_u32(addr)? as u64,
                8 => self.ram.read_u64(addr)?,
                _ => bail!("unsupported bus read size {size}"),
            });
        }
        let region = self
            .regions
            .iter()
            .find(|region| region.range().contains(&addr))
            .ok_or_else(|| anyhow::anyhow!("unmapped read at 0x{addr:016x}"))?;
        Ok(region.device.read(addr - region.base, size))
    }

    fn write_le_locked(&self, addr: u64, size: u8, value: u64) -> Result<()> {
        if self.ram.contains(addr, size as usize) {
            match size {
                1 => self.ram.write_u8(addr, value as u8)?,
                2 => self.ram.write_u16(addr, value as u16)?,
                4 => self.ram.write_u32(addr, value as u32)?,
                8 => self.ram.write_u64(addr, value)?,
                _ => bail!("unsupported bus write size {size}"),
            }
            return Ok(());
        }
        let region = self
            .regions
            .iter()
            .find(|region| region.range().contains(&addr))
            .ok_or_else(|| anyhow::anyhow!("unmapped write at 0x{addr:016x}"))?;
        region.device.write(addr - region.base, size, value);
        Ok(())
    }
}
