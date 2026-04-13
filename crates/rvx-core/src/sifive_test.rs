#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownReason {
    Pass,
    Reset,
    Fail(u32),
}

#[derive(Debug, Default)]
pub struct SifiveTest {
    reason: Option<ShutdownReason>,
}

impl SifiveTest {
    pub fn new() -> Self {
        Self { reason: None }
    }

    pub fn read(&self, _offset: u64, _size: u8) -> u64 {
        0
    }

    pub fn write(&mut self, _offset: u64, _size: u8, value: u64) {
        let code = (value >> 16) as u32;
        self.reason = Some(match value & 0xffff {
            0x5555 => ShutdownReason::Pass,
            0x7777 => ShutdownReason::Reset,
            _ => ShutdownReason::Fail(code),
        });
    }

    pub fn reason(&self) -> Option<ShutdownReason> {
        self.reason
    }
}
