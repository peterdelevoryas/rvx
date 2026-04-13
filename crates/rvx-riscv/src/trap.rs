#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum Exception {
    InstructionAddressMisaligned = 0,
    InstructionAccessFault = 1,
    IllegalInstruction = 2,
    Breakpoint = 3,
    LoadAddressMisaligned = 4,
    LoadAccessFault = 5,
    StoreAddressMisaligned = 6,
    StoreAccessFault = 7,
    UserEnvCall = 8,
    SupervisorEnvCall = 9,
    MachineEnvCall = 11,
    InstructionPageFault = 12,
    LoadPageFault = 13,
    StorePageFault = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum Interrupt {
    SupervisorSoft = 1,
    MachineSoft = 3,
    SupervisorTimer = 5,
    MachineTimer = 7,
    SupervisorExternal = 9,
    MachineExternal = 11,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trap {
    pub exception: Option<Exception>,
    pub interrupt: Option<Interrupt>,
    pub tval: u64,
}

impl Trap {
    pub fn exception(exception: Exception, tval: u64) -> Self {
        Self {
            exception: Some(exception),
            interrupt: None,
            tval,
        }
    }

    pub fn interrupt(interrupt: Interrupt) -> Self {
        Self {
            exception: None,
            interrupt: Some(interrupt),
            tval: 0,
        }
    }

    pub fn cause(self) -> u64 {
        if let Some(interrupt) = self.interrupt {
            (1u64 << 63) | interrupt as u64
        } else {
            self.exception.expect("trap must carry cause") as u64
        }
    }

    pub fn from_cause(cause: u64, tval: u64) -> Option<Self> {
        if (cause & (1u64 << 63)) != 0 {
            let interrupt = match cause & !(1u64 << 63) {
                1 => Interrupt::SupervisorSoft,
                3 => Interrupt::MachineSoft,
                5 => Interrupt::SupervisorTimer,
                7 => Interrupt::MachineTimer,
                9 => Interrupt::SupervisorExternal,
                11 => Interrupt::MachineExternal,
                _ => return None,
            };
            return Some(Self::interrupt(interrupt));
        }

        let exception = match cause {
            0 => Exception::InstructionAddressMisaligned,
            1 => Exception::InstructionAccessFault,
            2 => Exception::IllegalInstruction,
            3 => Exception::Breakpoint,
            4 => Exception::LoadAddressMisaligned,
            5 => Exception::LoadAccessFault,
            6 => Exception::StoreAddressMisaligned,
            7 => Exception::StoreAccessFault,
            8 => Exception::UserEnvCall,
            9 => Exception::SupervisorEnvCall,
            11 => Exception::MachineEnvCall,
            12 => Exception::InstructionPageFault,
            13 => Exception::LoadPageFault,
            15 => Exception::StorePageFault,
            _ => return None,
        };
        Some(Self::exception(exception, tval))
    }
}
