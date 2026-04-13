use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use rvx_core::{
    Aclint, Bus, FdtBuilder, IrqLine, IrqSink, MmioDevice, Plic, Ram, ShutdownReason, SifiveTest,
    Uart16550, load_binary, load_elf64,
};

pub const MROM_BASE: u64 = 0x0000_1000;
pub const MROM_SIZE: u64 = 0x0000_f000;
pub const RAM_BASE: u64 = 0x8000_0000;
pub const ACLINT_BASE: u64 = 0x0200_0000;
pub const ACLINT_SIZE: u64 = 0x0001_0000;
pub const PLIC_BASE: u64 = 0x0c00_0000;
pub const PLIC_SIZE: u64 = 0x0400_0000;
pub const UART0_BASE: u64 = 0x1000_0000;
pub const UART0_SIZE: u64 = 0x100;
pub const SIFIVE_TEST_BASE: u64 = 0x0010_0000;
pub const SIFIVE_TEST_SIZE: u64 = 0x1000;
const PLIC_NUM_SOURCES: u32 = 96;
const UART_IRQ: u32 = 10;
const IRQ_MSI: u32 = 3;
const IRQ_MTI: u32 = 7;
const IRQ_SEI: u32 = 9;
const IRQ_MEI: u32 = 11;
const KERNEL_OFFSET: u64 = 0x20_0000;
const DTB_SLACK_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone)]
pub struct TestBundle {
    pub firmware: PathBuf,
    pub kernel: PathBuf,
    pub initrd: Option<PathBuf>,
    pub append: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct MachineLayout {
    pub ram_base: u64,
    pub ram_size: u64,
    pub fdt_addr: u64,
    pub initrd_range: Option<(u64, u64)>,
}

pub struct MachineModel {
    bus: Bus,
    mrom: Ram,
    pub uart: Arc<Mutex<UartDevice>>,
    pub aclint: Arc<Mutex<AclintDevice>>,
    pub plic: Arc<Mutex<PlicDevice>>,
    pub sifive_test: Arc<Mutex<SifiveTestDevice>>,
    pub hart_count: usize,
}

impl MachineModel {
    pub fn new(hart_count: usize, ram_bytes: u64) -> Result<Self> {
        let mut bus = Bus::new(Ram::new(RAM_BASE, ram_bytes)?);
        let mrom = Ram::new(MROM_BASE, MROM_SIZE)?;

        let uart = Arc::new(Mutex::new(UartDevice(Uart16550::new())));
        let aclint = Arc::new(Mutex::new(AclintDevice(Aclint::new(
            hart_count, 10_000_000,
        ))));
        let plic = Arc::new(Mutex::new(PlicDevice(Plic::new(
            hart_count * 2,
            PLIC_NUM_SOURCES,
        ))));
        let sifive_test = Arc::new(Mutex::new(SifiveTestDevice(SifiveTest::new())));
        let plic_sink = Arc::new(PlicIrqSink(plic.clone())) as Arc<dyn IrqSink>;
        uart.lock()
            .unwrap()
            .0
            .attach_irq(IrqLine::new(plic_sink, UART_IRQ));

        bus.map_mmio(UART0_BASE, UART0_SIZE, uart.clone());
        bus.map_mmio(ACLINT_BASE, ACLINT_SIZE, aclint.clone());
        bus.map_mmio(PLIC_BASE, PLIC_SIZE, plic.clone());
        bus.map_mmio(SIFIVE_TEST_BASE, SIFIVE_TEST_SIZE, sifive_test.clone());

        Ok(Self {
            bus,
            mrom,
            uart,
            aclint,
            plic,
            sifive_test,
            hart_count,
        })
    }

    pub fn bus(&self) -> &Bus {
        &self.bus
    }

    pub fn mrom(&self) -> &Ram {
        &self.mrom
    }

    pub fn mrom_mut(&mut self) -> &mut Ram {
        &mut self.mrom
    }

    pub fn layout(&self) -> MachineLayout {
        MachineLayout {
            ram_base: RAM_BASE,
            ram_size: self.bus.ram().len(),
            fdt_addr: 0,
            initrd_range: None,
        }
    }

    pub fn generate_fdt(
        &self,
        bootargs: Option<&str>,
        initrd_range: Option<(u64, u64)>,
    ) -> Vec<u8> {
        let mut fdt = FdtBuilder::new();
        let cpu_phandle_base = self.hart_count as u32 * 2 + 1;
        let plic_phandle = cpu_phandle_base + self.hart_count as u32;

        fdt.begin_node("");
        fdt.property_string("compatible", "rvx,riscv64-virt");
        fdt.property_string("model", "rvx,riscv64-virt");
        fdt.property_u32("#address-cells", 2);
        fdt.property_u32("#size-cells", 2);

        fdt.begin_node("cpus");
        fdt.property_u32("#address-cells", 1);
        fdt.property_u32("#size-cells", 0);
        fdt.property_u32("timebase-frequency", 10_000_000);
        for hart in 0..self.hart_count as u32 {
            let intc_phandle = hart * 2 + 1;
            let cpu_phandle = cpu_phandle_base + hart;
            fdt.begin_node(&format!("cpu@{hart}"));
            fdt.property_string("device_type", "cpu");
            fdt.property_u32("reg", hart);
            fdt.property_string("compatible", "riscv");
            fdt.property_string("riscv,isa", "rv64imafd_zicsr_zifencei");
            fdt.property_string("mmu-type", "riscv,sv57");
            fdt.property_string("enable-method", "riscv,sbi-hsm");
            fdt.property_string("status", "okay");
            fdt.property_u32("phandle", cpu_phandle);
            fdt.begin_node("interrupt-controller");
            fdt.property_u32("#interrupt-cells", 1);
            fdt.property_bytes("interrupt-controller", &[]);
            fdt.property_string("compatible", "riscv,cpu-intc");
            fdt.property_u32("phandle", intc_phandle);
            fdt.end_node();
            fdt.end_node();
        }
        fdt.end_node();

        fdt.begin_node(&format!("memory@{RAM_BASE:x}"));
        fdt.property_string("device_type", "memory");
        fdt.property_u32_list(
            "reg",
            &[
                0,
                RAM_BASE as u32,
                (self.bus.ram().len() >> 32) as u32,
                self.bus.ram().len() as u32,
            ],
        );
        fdt.end_node();

        fdt.begin_node("soc");
        fdt.property_u32("#address-cells", 2);
        fdt.property_u32("#size-cells", 2);
        fdt.property_string("compatible", "simple-bus");
        fdt.property_bytes("ranges", &[]);

        let mut plic_ext = Vec::with_capacity(self.hart_count * 4);
        for hart in 0..self.hart_count as u32 {
            let intc_phandle = hart * 2 + 1;
            plic_ext.extend([intc_phandle, IRQ_MEI, intc_phandle, IRQ_SEI]);
        }
        fdt.begin_node("plic@c000000");
        fdt.property_string("compatible", "sifive,plic-1.0.0");
        fdt.property_u32("#interrupt-cells", 1);
        fdt.property_bytes("interrupt-controller", &[]);
        fdt.property_u32("phandle", plic_phandle);
        fdt.property_u32("riscv,ndev", PLIC_NUM_SOURCES);
        fdt.property_u32_list("interrupts-extended", &plic_ext);
        fdt.property_u32_list("reg", &[0, PLIC_BASE as u32, 0, PLIC_SIZE as u32]);
        fdt.end_node();

        fdt.begin_node("serial@10000000");
        fdt.property_string("compatible", "ns16550a");
        fdt.property_u32("clock-frequency", 3_686_400);
        fdt.property_u32("current-speed", 115200);
        fdt.property_u32("interrupts", UART_IRQ);
        fdt.property_u32("interrupt-parent", plic_phandle);
        fdt.property_u32_list("reg", &[0, UART0_BASE as u32, 0, UART0_SIZE as u32]);
        fdt.end_node();

        let mut clint_ext = Vec::with_capacity(self.hart_count * 4);
        for hart in 0..self.hart_count as u32 {
            let intc_phandle = hart * 2 + 1;
            clint_ext.extend([intc_phandle, IRQ_MTI, intc_phandle, IRQ_MSI]);
        }
        fdt.begin_node("clint@2000000");
        fdt.property_string("compatible", "riscv,clint0");
        fdt.property_u32_list("interrupts-extended", &clint_ext);
        fdt.property_u32_list("reg", &[0, ACLINT_BASE as u32, 0, ACLINT_SIZE as u32]);
        fdt.end_node();

        fdt.begin_node("test@100000");
        fdt.property_string("compatible", "sifive,test0");
        fdt.property_u32_list(
            "reg",
            &[0, SIFIVE_TEST_BASE as u32, 0, SIFIVE_TEST_SIZE as u32],
        );
        fdt.end_node();

        fdt.end_node();

        fdt.begin_node("chosen");
        fdt.property_string("stdout-path", "/soc/serial@10000000");
        if let Some(bootargs) = bootargs {
            fdt.property_string("bootargs", bootargs);
        }
        if let Some((start, end)) = initrd_range {
            fdt.property_u64("linux,initrd-start", start);
            fdt.property_u64("linux,initrd-end", end);
        }
        fdt.end_node();

        fdt.end_node();
        fdt.finish()
    }

    pub fn load_bundle(&mut self, bundle: &TestBundle) -> Result<BootLayout> {
        let firmware_bytes = std::fs::read(&bundle.firmware)
            .with_context(|| format!("failed to read {}", bundle.firmware.display()))?;
        let kernel_bytes = std::fs::read(&bundle.kernel)
            .with_context(|| format!("failed to read {}", bundle.kernel.display()))?;
        let initrd_bytes = bundle
            .initrd
            .as_ref()
            .map(std::fs::read)
            .transpose()
            .context("failed to read initrd")?;

        let firmware = if is_elf(&firmware_bytes) {
            load_elf64(&firmware_bytes, self.bus.ram())?
        } else {
            load_binary(&firmware_bytes, RAM_BASE, self.bus.ram())?
        };
        let kernel = if is_elf(&kernel_bytes) {
            load_elf64(&kernel_bytes, self.bus.ram())?
        } else {
            load_binary(&kernel_bytes, RAM_BASE + KERNEL_OFFSET, self.bus.ram())?
        };

        let layout = compute_layout(self.layout(), initrd_bytes.as_deref(), 0)?;
        let fdt = self.generate_fdt(bundle.append.as_deref(), layout.initrd_range);
        let layout = compute_layout(self.layout(), initrd_bytes.as_deref(), fdt.len() as u64)?;
        let fdt = self.generate_fdt(bundle.append.as_deref(), layout.initrd_range);
        self.bus.load(layout.fdt_addr, &fdt)?;

        if let Some(initrd) = initrd_bytes.as_deref()
            && let Some((start, _)) = layout.initrd_range
        {
            self.bus.load(start, initrd)?;
        }

        Ok(BootLayout {
            firmware_entry: firmware.entry,
            kernel_entry: kernel.entry,
            fdt_addr: layout.fdt_addr,
            initrd_range: layout.initrd_range,
        })
    }

    pub fn shutdown_reason(&self) -> Option<ShutdownReason> {
        self.sifive_test.lock().unwrap().0.reason()
    }

    pub fn supervisor_external_pending(&self, hart_id: usize) -> bool {
        self.plic.lock().unwrap().0.context_pending(hart_id * 2 + 1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BootLayout {
    pub firmware_entry: u64,
    pub kernel_entry: u64,
    pub fdt_addr: u64,
    pub initrd_range: Option<(u64, u64)>,
}

pub struct UartDevice(pub Uart16550);
pub struct AclintDevice(pub Aclint);
pub struct PlicDevice(pub Plic);
pub struct SifiveTestDevice(pub SifiveTest);
struct PlicIrqSink(Arc<Mutex<PlicDevice>>);

impl MmioDevice for UartDevice {
    fn read(&mut self, offset: u64, _size: u8) -> u64 {
        self.0.read(offset) as u64
    }

    fn write(&mut self, offset: u64, _size: u8, value: u64) {
        self.0.write(offset, value as u8);
    }
}

impl MmioDevice for AclintDevice {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.0.read(offset, size)
    }

    fn write(&mut self, offset: u64, size: u8, value: u64) {
        self.0.write(offset, size, value);
    }
}

impl MmioDevice for PlicDevice {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.0.read(offset, size)
    }

    fn write(&mut self, offset: u64, size: u8, value: u64) {
        self.0.write(offset, size, value);
    }
}

impl MmioDevice for SifiveTestDevice {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.0.read(offset, size)
    }

    fn write(&mut self, offset: u64, size: u8, value: u64) {
        self.0.write(offset, size, value);
    }
}

impl IrqSink for PlicIrqSink {
    fn set_irq(&self, irq: u32, level: bool) {
        self.0.lock().unwrap().0.set_irq(irq, level);
    }
}

fn compute_layout(
    layout: MachineLayout,
    initrd: Option<&[u8]>,
    dtb_size: u64,
) -> Result<MachineLayout> {
    let ram_top = layout.ram_base + layout.ram_size;
    let dtb_reserved = align_up(
        dtb_size
            .checked_add(DTB_SLACK_BYTES)
            .context("dtb reservation overflow")?,
        0x1000,
    );
    let fdt_addr = align_down(
        ram_top
            .checked_sub(dtb_reserved)
            .context("ram too small for dtb")?,
        0x1000,
    );
    let initrd_range = if let Some(initrd) = initrd {
        let end = align_down(
            fdt_addr
                .checked_sub(0x1000)
                .context("ram too small for initrd gap")?,
            0x1000,
        );
        let start = align_down(
            end.checked_sub(initrd.len() as u64)
                .context("ram too small for initrd")?,
            0x1000,
        );
        if start < layout.ram_base + KERNEL_OFFSET + 0x20_0000 {
            bail!("initrd overlaps kernel window");
        }
        Some((start, start + initrd.len() as u64))
    } else {
        None
    };
    Ok(MachineLayout {
        ram_base: layout.ram_base,
        ram_size: layout.ram_size,
        fdt_addr,
        initrd_range,
    })
}

fn align_up(value: u64, align: u64) -> u64 {
    if value % align == 0 {
        value
    } else {
        value + (align - value % align)
    }
}

fn align_down(value: u64, align: u64) -> u64 {
    value - value % align
}

fn is_elf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x7fELF")
}
