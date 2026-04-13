# RVX

`rvx` is a Rust RISC-V emulator focused on Linux bring-up on a QEMU `virt`-style
machine with a throughput-first SMP runtime.

Current scope:

- First-party RV64 guest CPU, MMU, SBI, and device model in Rust
- `virt`-style DRAM, UART, PLIC, ACLINT, SiFive test shutdown, and optional
  VirtIO block
- External OpenSBI `fw_dynamic.bin` boot flow
- Linux `Image` + BusyBox initramfs boot path, including an automated
  `hello world` smoke test
- 4-hart machine model with SMP boot support

## Build

```bash
cargo build --release
```

## Prepare guest artifacts

This builds a pinned OpenSBI + Linux + BusyBox bundle under `artifacts/`.

```bash
cargo run -p xtask -- build-bundle
```

Shortcut:

```bash
cargo rvx-bundle
```

Artifacts produced:

- `artifacts/out/opensbi/fw_dynamic.bin`
- `artifacts/out/linux/Image`
- `artifacts/out/rootfs/rootfs.cpio.gz`

## Boot Linux

Fastest way to test the current setup:

```bash
cargo rvx-run
```

This alias boots the bundled OpenSBI + Linux + BusyBox image with `1` hart and
`128` MiB of RAM.

```bash
cargo run --release -p rvx-cli -- \
  --firmware artifacts/out/opensbi/fw_dynamic.bin \
  --kernel artifacts/out/linux/Image \
  --initrd artifacts/out/rootfs/rootfs.cpio.gz \
  --append "console=ttyS0 earlycon=sbi root=/dev/ram0 rdinit=/init loglevel=7" \
  --smp 4 \
  --mem 1024
```

Use `--perf-json /tmp/rvx.json` to emit a JSON boot profile after exit.
