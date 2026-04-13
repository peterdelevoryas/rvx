use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs as unix_fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use xshell::{Shell, cmd};

const OPENSBI_VERSION: &str = "1.5.1";
const LINUX_VERSION: &str = "6.12.25";
const BUSYBOX_VERSION: &str = "1.36.1";
const RVX_GUEST_ARCH: &str = "rv64imafd_zicsr_zifencei";
const RVX_GUEST_ABI: &str = "lp64d";
const RVX_BUSYBOX_TARGET: &str = "riscv64-linux-musl";
const RVX_BUSYBOX_CPU: &str = "baseline_rv64-c-f-d";
const RVX_BUSYBOX_ABI: &str = "lp64";
const RVX_BUSYBOX_TOOLCHAIN: &str = "zig-musl-v3";
const ZIG_CC: &str = "/home/linuxbrew/.linuxbrew/bin/zig";

fn main() {
    if let Err(err) = run() {
        eprintln!("xtask: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("build-bundle") | None => build_bundle(),
        Some("build-test-bundle") => build_test_bundle(),
        Some(other) => bail!("unknown xtask command: {other}"),
    }
}

fn build_bundle() -> Result<()> {
    let sources = prepare_sources()?;
    let sh = Shell::new()?;
    let rootfs_dir = sources.out_dir.join("rootfs");
    build_busybox_rootfs(&sh, &sources.busybox_src, &rootfs_dir)?;
    build_linux(&sh, &sources.linux_src, &rootfs_dir)?;
    build_opensbi(&sh, &sources.opensbi_src, &sources.out_dir.join("opensbi"))?;
    Ok(())
}

fn build_test_bundle() -> Result<()> {
    let sources = prepare_sources()?;
    let sh = Shell::new()?;
    let out_dir = sources.out_dir.join("test");
    let rootfs_dir = out_dir.join("rootfs");
    build_busybox_rootfs_with_init(&sh, &sources.busybox_src, &rootfs_dir, &test_init_script())?;
    build_linux(&sh, &sources.linux_src, &rootfs_dir)?;
    build_opensbi(&sh, &sources.opensbi_src, &out_dir.join("opensbi"))?;
    Ok(())
}

struct Sources {
    opensbi_src: PathBuf,
    linux_src: PathBuf,
    busybox_src: PathBuf,
    out_dir: PathBuf,
}

fn prepare_sources() -> Result<Sources> {
    let repo_root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?)
        .parent()
        .unwrap()
        .to_path_buf();
    let artifacts = repo_root.join("artifacts");
    let download_dir = artifacts.join("downloads");
    let source_dir = artifacts.join("src");
    let out_dir = artifacts.join("out");
    fs::create_dir_all(&download_dir)?;
    fs::create_dir_all(&source_dir)?;
    fs::create_dir_all(&out_dir)?;

    let opensbi_src = unpack_tarball(
        &download_dir,
        &source_dir,
        &format!(
            "https://github.com/riscv-software-src/opensbi/archive/refs/tags/v{OPENSBI_VERSION}.tar.gz"
        ),
        &format!("opensbi-{OPENSBI_VERSION}.tar.gz"),
    )?;
    let linux_src = unpack_tarball(
        &download_dir,
        &source_dir,
        &format!("https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-{LINUX_VERSION}.tar.xz"),
        &format!("linux-{LINUX_VERSION}.tar.xz"),
    )?;
    let busybox_src = unpack_tarball(
        &download_dir,
        &source_dir,
        &format!("https://busybox.net/downloads/busybox-{BUSYBOX_VERSION}.tar.bz2"),
        &format!("busybox-{BUSYBOX_VERSION}.tar.bz2"),
    )?;

    Ok(Sources {
        opensbi_src,
        linux_src,
        busybox_src,
        out_dir,
    })
}

fn build_opensbi(sh: &Shell, source: &Path, out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let build_dir = source.join("build");
    if !build_dir
        .join("platform/generic/firmware/fw_dynamic.bin")
        .exists()
    {
        if build_dir.exists() {
            fs::remove_dir_all(&build_dir)
                .with_context(|| format!("failed to clear {}", build_dir.display()))?;
        }
        cmd!(
            sh,
            "make -C {source} CROSS_COMPILE=riscv64-linux-gnu- PLATFORM=generic FW_PIC=y FW_DYNAMIC=y platform-cflags-y=-std=gnu11 -j24"
        )
        .run()?;
    }
    fs::copy(
        build_dir.join("platform/generic/firmware/fw_dynamic.bin"),
        out_dir.join("fw_dynamic.bin"),
    )
    .context("failed to copy fw_dynamic.bin")?;
    Ok(())
}

fn build_linux(sh: &Shell, source: &Path, rootfs_dir: &Path) -> Result<()> {
    let image = source.join("arch/riscv/boot/Image");
    let initrd = rootfs_dir.join("rootfs.cpio.gz");
    let stamp = source.join(".rvx-kernel-build-stamp");
    let build_stamp = format!("arch={RVX_GUEST_ARCH}\nabi={RVX_GUEST_ABI}\n");
    let kernel_flags = format!("-march={RVX_GUEST_ARCH} -mabi={RVX_GUEST_ABI}");
    if !initrd.exists() {
        bail!("missing initrd {}", initrd.display());
    }
    if fs::read_to_string(&stamp).ok().as_deref() != Some(build_stamp.as_str()) {
        let _ = cmd!(
            sh,
            "make -C {source} ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- mrproper"
        )
        .run();
    }
    if !source.join(".config").exists() {
        cmd!(
            sh,
            "make -C {source} ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig"
        )
        .run()?;
    }
    rewrite_kconfig(
        &source.join(".config"),
        &[
            ("CONFIG_NONPORTABLE", "y"),
            ("CONFIG_SMP", "y"),
            ("CONFIG_NR_CPUS", "64"),
            ("CONFIG_SERIAL_8250", "y"),
            ("CONFIG_SERIAL_8250_CONSOLE", "y"),
            ("CONFIG_SERIAL_OF_PLATFORM", "y"),
            ("CONFIG_BLK_DEV_INITRD", "y"),
            ("CONFIG_DEVTMPFS", "y"),
            ("CONFIG_DEVTMPFS_MOUNT", "y"),
        ],
        &[
            "CONFIG_DEBUG_INFO",
            "CONFIG_PORTABLE",
            "CONFIG_EFI",
            "CONFIG_DMI",
            "CONFIG_RISCV_ISA_C",
            "CONFIG_RISCV_ISA_V",
            "CONFIG_RISCV_ISA_V_DEFAULT_ENABLE",
            "CONFIG_RISCV_ISA_VENDOR_EXT",
            "CONFIG_RISCV_ISA_VENDOR_EXT_ANDES",
        ],
    )?;
    cmd!(
        sh,
        "make -C {source} ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- olddefconfig"
    )
    .run()?;
    rewrite_kconfig(
        &source.join(".config"),
        &[],
        &[
            "CONFIG_PORTABLE",
            "CONFIG_EFI",
            "CONFIG_DMI",
            "CONFIG_RISCV_ISA_C",
            "CONFIG_RISCV_ISA_V",
            "CONFIG_RISCV_ISA_V_DEFAULT_ENABLE",
            "CONFIG_RISCV_ISA_VENDOR_EXT",
            "CONFIG_RISCV_ISA_VENDOR_EXT_ANDES",
        ],
    )?;
    if !image.exists() || fs::read_to_string(&stamp).ok().as_deref() != Some(build_stamp.as_str()) {
        let mut build = cmd!(
            sh,
            "make -C {source} ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- Image -j24"
        );
        build = build
            .env("KCFLAGS", &kernel_flags)
            .env("KAFLAGS", &kernel_flags);
        build.run()?;
        fs::write(&stamp, &build_stamp)?;
    }

    let out_dir = rootfs_dir.parent().unwrap().join("linux");
    fs::create_dir_all(&out_dir)?;
    fs::copy(image, out_dir.join("Image")).context("failed to copy kernel Image")?;
    Ok(())
}

fn build_busybox_rootfs(sh: &Shell, source: &Path, out_dir: &Path) -> Result<()> {
    build_busybox_rootfs_with_init(sh, source, out_dir, &default_init_script())
}

fn build_busybox_rootfs_with_init(
    sh: &Shell,
    source: &Path,
    out_dir: &Path,
    init_script: &str,
) -> Result<()> {
    let rootfs_dir = out_dir.join("staging");
    let install_dir = rootfs_dir.join("busybox-install");
    let archive = out_dir.join("rootfs.cpio.gz");
    let stamp = out_dir.join(".rvx-busybox-build-stamp");
    let wrappers = prepare_busybox_toolchain(out_dir)?;
    let busybox_config = [
        ("CONFIG_BUSYBOX", "y"),
        ("CONFIG_STATIC", "y"),
        ("CONFIG_FEATURE_INSTALLER", "y"),
        ("CONFIG_INSTALL_APPLET_SYMLINKS", "y"),
        ("CONFIG_SH_IS_ASH", "y"),
        ("CONFIG_ASH", "y"),
        ("CONFIG_FEATURE_SH_STANDALONE", "y"),
        ("CONFIG_FEATURE_PREFER_APPLETS", "y"),
        ("CONFIG_FEATURE_EDITING", "y"),
        ("CONFIG_FEATURE_EDITING_FANCY_PROMPT", "y"),
        ("CONFIG_FEATURE_EDITING_HISTORY", "128"),
        ("CONFIG_FEATURE_TAB_COMPLETION", "y"),
        ("CONFIG_CTTYHACK", "y"),
        ("CONFIG_SETSID", "y"),
        ("CONFIG_INIT", "y"),
        ("CONFIG_FEATURE_USE_INITTAB", "y"),
        ("CONFIG_GETTY", "y"),
        ("CONFIG_MOUNT", "y"),
        ("CONFIG_CLEAR", "y"),
        ("CONFIG_ECHO", "y"),
        ("CONFIG_ENV", "y"),
        ("CONFIG_PRINTENV", "y"),
        ("CONFIG_CAT", "y"),
        ("CONFIG_CHMOD", "y"),
        ("CONFIG_DMESG", "y"),
        ("CONFIG_ID", "y"),
        ("CONFIG_LS", "y"),
        ("CONFIG_MKDIR", "y"),
        ("CONFIG_PS", "y"),
        ("CONFIG_FEATURE_PS_WIDE", "y"),
        ("CONFIG_PWD", "y"),
        ("CONFIG_SLEEP", "y"),
        ("CONFIG_STTY", "y"),
        ("CONFIG_SYNC", "y"),
        ("CONFIG_TOUCH", "y"),
        ("CONFIG_TTY", "y"),
        ("CONFIG_HOSTNAME", "y"),
        ("CONFIG_POWEROFF", "y"),
        ("CONFIG_REBOOT", "y"),
        ("CONFIG_UNAME", "y"),
    ];
    let busybox_config_stamp = busybox_config
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("\n");
    let bin_compat_stamp = busybox_bin_compat_applets().join(",");
    let guest_poweroff_helper_stamp = guest_poweroff_helper_source();
    let build_stamp = format!(
        "target={RVX_BUSYBOX_TARGET}\ncpu={RVX_BUSYBOX_CPU}\nabi={RVX_BUSYBOX_ABI}\ntoolchain={RVX_BUSYBOX_TOOLCHAIN}\nconfig={busybox_config_stamp}\nbin_compat={bin_compat_stamp}\npoweroff_helper={guest_poweroff_helper_stamp}\ninit={init_script}\n"
    );
    if archive.exists() && fs::read_to_string(&stamp).ok().as_deref() == Some(build_stamp.as_str())
    {
        return Ok(());
    }
    if rootfs_dir.exists() {
        fs::remove_dir_all(&rootfs_dir)?;
    }
    if archive.exists() {
        fs::remove_file(&archive)?;
    }
    fs::create_dir_all(&rootfs_dir)?;

    let _ = cmd!(sh, "make -C {source} distclean").run();
    cmd!(sh, "make -C {source} allnoconfig").run()?;
    rewrite_kconfig(
        &source.join(".config"),
        &busybox_config,
        &["CONFIG_DEBUG", "CONFIG_HUSH"],
    )?;
    run_busybox_oldconfig(source)?;
    let cc = &wrappers.cc;
    let ld = &wrappers.ld;
    let ar = &wrappers.ar;
    let ranlib = &wrappers.ranlib;
    let nm = &wrappers.nm;
    let strip = &wrappers.strip;
    let objcopy = &wrappers.objcopy;
    let objdump = &wrappers.objdump;
    cmd!(
        sh,
        "make -C {source} -j24 CC={cc} LD={ld} AR={ar} RANLIB={ranlib} NM={nm} STRIP={strip} OBJCOPY={objcopy} OBJDUMP={objdump}"
    )
    .run()?;
    cmd!(
        sh,
        "make -C {source} CC={cc} LD={ld} AR={ar} RANLIB={ranlib} NM={nm} STRIP={strip} OBJCOPY={objcopy} OBJDUMP={objdump} CONFIG_PREFIX={install_dir} install"
    )
    .run()?;
    install_busybox_bin_compat_symlinks(&install_dir)?;

    for dir in ["proc", "sys", "dev", "tmp", "etc", "root"] {
        fs::create_dir_all(install_dir.join(dir))?;
    }

    install_guest_poweroff_helper(sh, out_dir, &install_dir, cc)?;

    let init_path = install_dir.join("init");
    fs::write(&init_path, init_script)?;
    cmd!(sh, "chmod +x {init_path}").run()?;

    create_initramfs_archive(&install_dir, &archive)?;
    fs::write(&stamp, &build_stamp)?;
    Ok(())
}

fn install_busybox_bin_compat_symlinks(install_dir: &Path) -> Result<()> {
    for applet in busybox_bin_compat_applets() {
        let link = install_dir.join("bin").join(applet);
        if link.exists() {
            continue;
        }
        unix_fs::symlink("busybox", &link)
            .with_context(|| format!("failed to create {}", link.display()))?;
    }
    Ok(())
}

fn busybox_bin_compat_applets() -> &'static [&'static str] {
    &["clear", "env", "id", "tty"]
}

struct BusyboxToolchain {
    cc: String,
    ld: String,
    ar: String,
    ranlib: String,
    nm: String,
    strip: String,
    objcopy: String,
    objdump: String,
}

fn prepare_busybox_toolchain(out_dir: &Path) -> Result<BusyboxToolchain> {
    let tools_dir = out_dir.join("zig-musl-tools");
    fs::create_dir_all(&tools_dir)?;

    let cc = tools_dir.join("riscv64-linux-musl-gcc");
    let ld = tools_dir.join("riscv64-linux-musl-ld");
    let ar = tools_dir.join("riscv64-linux-musl-ar");
    let ranlib = tools_dir.join("riscv64-linux-musl-ranlib");

    write_executable(&cc, &zig_cc_wrapper_script())?;
    write_executable(&ld, &zig_cc_wrapper_script())?;
    write_executable(&ar, &format!("#!/bin/sh\nexec {ZIG_CC} ar \"$@\"\n"))?;
    write_executable(
        &ranlib,
        &format!("#!/bin/sh\nexec {ZIG_CC} ranlib \"$@\"\n"),
    )?;

    Ok(BusyboxToolchain {
        cc: cc.display().to_string(),
        ld: ld.display().to_string(),
        ar: ar.display().to_string(),
        ranlib: ranlib.display().to_string(),
        nm: "riscv64-linux-gnu-nm".to_string(),
        strip: "riscv64-linux-gnu-strip".to_string(),
        objcopy: "riscv64-linux-gnu-objcopy".to_string(),
        objdump: "riscv64-linux-gnu-objdump".to_string(),
    })
}

fn zig_cc_wrapper_script() -> String {
    format!(
        r#"#!/usr/bin/env bash
set -euo pipefail
args=()
skip_next=0
for arg in "$@"; do
    if [[ "$skip_next" -eq 1 ]]; then
        skip_next=0
        continue
    fi
    case "$arg" in
        -Wl,--warn-common|--warn-common|-Wl,--verbose|--verbose|-Wl,-Map,*)
            ;;
        -Map)
            skip_next=1
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done
exec {ZIG_CC} cc -target {RVX_BUSYBOX_TARGET} -mcpu={RVX_BUSYBOX_CPU} -mabi={RVX_BUSYBOX_ABI} -static "${{args[@]}}"
"#
    )
}

fn write_executable(path: &Path, contents: &str) -> Result<()> {
    fs::write(path, contents)?;
    let status = Command::new("chmod")
        .args(["+x", path.to_str().unwrap()])
        .status()?;
    if !status.success() {
        bail!("chmod failed for {}", path.display());
    }
    Ok(())
}

fn unpack_tarball(
    download_dir: &Path,
    source_dir: &Path,
    url: &str,
    filename: &str,
) -> Result<PathBuf> {
    let archive = download_dir.join(filename);
    if !archive.exists() {
        let response = ureq::get(url)
            .call()
            .with_context(|| format!("download failed: {url}"))?;
        let mut reader = response.into_reader();
        let mut writer = std::io::BufWriter::new(fs::File::create(&archive)?);
        std::io::copy(&mut reader, &mut writer)?;
    }

    let marker = source_dir.join(format!("{filename}.unpacked"));
    if !marker.exists() {
        let sh = Shell::new()?;
        cmd!(sh, "tar -C {source_dir} -xf {archive}").run()?;
        fs::write(&marker, b"ok")?;
    }

    let root = archive_root_name(filename);
    Ok(source_dir.join(root))
}

fn archive_root_name(filename: &str) -> String {
    filename
        .trim_end_matches(".tar.gz")
        .trim_end_matches(".tar.xz")
        .trim_end_matches(".tar.bz2")
        .to_string()
}

fn run_busybox_oldconfig(source: &Path) -> Result<()> {
    let status = Command::new("bash")
        .arg("-lc")
        .arg(format!(
            "yes \"\" | make -C '{}' CROSS_COMPILE=riscv64-linux-gnu- oldconfig",
            source.display()
        ))
        .status()
        .context("failed to launch busybox oldconfig")?;
    if !status.success() {
        bail!("busybox oldconfig failed with status {status}");
    }
    Ok(())
}

fn create_initramfs_archive(install_dir: &Path, archive: &Path) -> Result<()> {
    let status = Command::new("bash")
        .current_dir(install_dir)
        .env("RVX_INITRAMFS_OUT", archive)
        .arg("-lc")
        .arg("find . -print0 | cpio --null -ov --format=newc | gzip -9 > \"$RVX_INITRAMFS_OUT\"")
        .status()
        .context("failed to launch initramfs archive builder")?;
    if !status.success() {
        bail!("initramfs archive builder failed with status {status}");
    }
    Ok(())
}

fn default_init_script() -> String {
    init_script_body("rvx")
}

fn test_init_script() -> String {
    init_script_body("rvx-test")
}

fn init_script_body(hostname: &str) -> String {
    format!(
        r#"#!/bin/sh
set -eu

PATH=/sbin:/bin:/usr/sbin:/usr/bin
export PATH

mount -t devtmpfs devtmpfs /dev
exec >/dev/console 2>&1

log() {{
    if uptime_line=$(cat /proc/uptime 2>/dev/null); then
        guest_uptime=${{uptime_line%% *}}
        echo "[init $guest_uptime] $*"
    else
        echo "[init] $*"
    fi
}}

start_step() {{
    STEP_LABEL="$1"
    log "$STEP_LABEL"
    (
        while :; do
            sleep 5
            log "$STEP_LABEL still running"
        done
    ) &
    STEP_HEARTBEAT_PID=$!
}}

finish_step() {{
    if [ -n "${{STEP_HEARTBEAT_PID:-}}" ]; then
        kill "$STEP_HEARTBEAT_PID" 2>/dev/null || true
        wait "$STEP_HEARTBEAT_PID" 2>/dev/null || true
        STEP_HEARTBEAT_PID=
    fi
    log "$STEP_LABEL complete"
    STEP_LABEL=
}}

run_step() {{
    start_step "$1"
    shift
    "$@"
    finish_step
}}

log "starting /init"
run_step "mounting /proc" mount -t proc proc /proc
run_step "mounting /sys" mount -t sysfs sysfs /sys
run_step "creating base directories" mkdir -p /tmp /root /etc
run_step "setting hostname to {hostname}" hostname {hostname}

start_step "writing /etc/profile"
cat >/etc/profile <<'EOF'
export PATH=/sbin:/bin:/usr/sbin:/usr/bin
export HOME=/root
export USER=root
export TERM=vt100
export PS1='rvx# '
if [ -r /tmp/init-shell-heartbeat.pid ]; then
    read -r init_shell_heartbeat_pid </tmp/init-shell-heartbeat.pid || true
    if [ -n "${{init_shell_heartbeat_pid:-}}" ]; then
        kill "$init_shell_heartbeat_pid" 2>/dev/null || true
    fi
    rm -f /tmp/init-shell-heartbeat.pid
fi
echo 'RVX BusyBox shell ready.'
echo 'Type `/bin/rvx-poweroff` to exit the guest.'
EOF
finish_step

run_step "switching to /root" cd /root

log "launching BusyBox login shell"
(
    while :; do
        sleep 5
        log "waiting for BusyBox shell prompt"
    done
) &
echo "$!" >/tmp/init-shell-heartbeat.pid
cd /root
exec /bin/busybox cttyhack /bin/sh -l </dev/console >/dev/console 2>&1
"#
    )
}

fn install_guest_poweroff_helper(
    sh: &Shell,
    out_dir: &Path,
    install_dir: &Path,
    cc: &str,
) -> Result<()> {
    let source = out_dir.join("rvx-poweroff.c");
    fs::write(&source, guest_poweroff_helper_source())?;

    let binary = install_dir.join("bin/rvx-poweroff");
    cmd!(sh, "{cc} -Os -o {binary} {source}").run()?;

    let poweroff = install_dir.join("sbin/poweroff");
    if poweroff.exists() {
        fs::remove_file(&poweroff)?;
    }
    write_executable(&poweroff, "#!/bin/sh\nexec /bin/rvx-poweroff \"$@\"\n")?;
    Ok(())
}

fn guest_poweroff_helper_source() -> &'static str {
    r#"#include <errno.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#define RVX_LINUX_REBOOT_MAGIC1 0xfee1dead
#define RVX_LINUX_REBOOT_MAGIC2 672274793
#define RVX_LINUX_REBOOT_CMD_POWER_OFF 0x4321fedc

int main(void) {
    static const char message[] = "rvx-poweroff: requesting kernel poweroff\n";
    (void)write(STDOUT_FILENO, message, sizeof(message) - 1);

    if (syscall(SYS_reboot, RVX_LINUX_REBOOT_MAGIC1, RVX_LINUX_REBOOT_MAGIC2,
                RVX_LINUX_REBOOT_CMD_POWER_OFF, 0) == 0) {
        return 0;
    }

    perror("reboot(LINUX_REBOOT_CMD_POWER_OFF)");
    return errno ? errno : 1;
}
"#
}

fn rewrite_kconfig(path: &Path, set: &[(&str, &str)], unset: &[&str]) -> Result<()> {
    let original = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    let mut lines = Vec::new();
    let mut seen = BTreeMap::<String, bool>::new();

    for line in original.lines() {
        let mut handled = false;
        for &(key, value) in set {
            if line.starts_with(&format!("{key}=")) || line == format!("# {key} is not set") {
                lines.push(format!("{key}={value}"));
                seen.insert(key.to_string(), true);
                handled = true;
                break;
            }
        }
        if handled {
            continue;
        }
        for &key in unset {
            if line.starts_with(&format!("{key}=")) || line == format!("# {key} is not set") {
                lines.push(format!("# {key} is not set"));
                seen.insert(key.to_string(), true);
                handled = true;
                break;
            }
        }
        if !handled {
            lines.push(line.to_string());
        }
    }

    for &(key, value) in set {
        if !seen.contains_key(key) {
            lines.push(format!("{key}={value}"));
        }
    }
    for &key in unset {
        if !seen.contains_key(key) {
            lines.push(format!("# {key} is not set"));
        }
    }

    fs::write(path, lines.join("\n") + "\n")
        .with_context(|| format!("failed to write config {}", path.display()))?;
    Ok(())
}
