use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn prepare_guest_artifacts(root: &PathBuf) {
    let perf_json = root.join("target/linux_boot_perf.json");
    let _ = std::fs::remove_file(&perf_json);

    let status = Command::new("cargo")
        .current_dir(&root)
        .args(["run", "-p", "xtask", "--", "build-test-bundle"])
        .status()
        .expect("failed to launch xtask");
    assert!(status.success(), "build-test-bundle failed");

    let status = Command::new("cargo")
        .current_dir(&root)
        .args(["build", "--release", "-p", "rvx-cli"])
        .status()
        .expect("failed to build rvx-cli");
    assert!(status.success(), "release rvx-cli build failed");
}

fn run_linux_boot(
    root: &PathBuf,
    perf_json: &PathBuf,
    smp: u32,
    time_limit_ms: u64,
    _write_rounds: usize,
    experimental_parallel: bool,
) -> Output {
    let _ = std::fs::remove_file(perf_json);

    let mut child = Command::new(root.join("target/release/rvx-cli"))
        .current_dir(&root)
        .envs(
            experimental_parallel
                .then_some(("RVX_EXPERIMENTAL_PARALLEL", "1"))
                .into_iter(),
        )
        .args([
            "--firmware",
            "artifacts/out/test/opensbi/fw_dynamic.bin",
            "--kernel",
            "artifacts/out/test/linux/Image",
            "--initrd",
            "artifacts/out/test/rootfs/rootfs.cpio.gz",
            "--append",
            "console=ttyS0 earlycon=sbi root=/dev/ram0 rdinit=/init quiet",
            "--smp",
            &smp.to_string(),
            "--mem",
            "128",
            "--nographic",
            "--time-limit-ms",
            &time_limit_ms.to_string(),
            "--perf-json",
            perf_json.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to launch rvx-cli");

    let stdout = child.stdout.take().expect("missing child stdout");
    let stderr = child.stderr.take().expect("missing child stderr");
    let mut child_stdin = child.stdin.take().expect("missing child stdin");
    let observed_output = Arc::new((Mutex::new(Vec::<u8>::new()), Condvar::new()));
    let stdout_observed = Arc::clone(&observed_output);
    let stdout_reader = thread::spawn(move || read_pipe(stdout, stdout_observed));
    let stderr_observed = Arc::clone(&observed_output);
    let stderr_reader = thread::spawn(move || read_pipe(stderr, stderr_observed));
    let writer = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_millis(time_limit_ms.saturating_sub(5_000));
        let mut next_poke = Instant::now();
        let mut warmup_pokes = 0usize;
        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            if warmup_pokes < 6 && now >= next_poke {
                if child_stdin.write_all(b"\r").is_err() {
                    break;
                }
                let _ = child_stdin.flush();
                warmup_pokes += 1;
                next_poke = now + Duration::from_secs(2);
            }

            let (observed, notify) = &*observed_output;
            let guard = observed.lock().unwrap();
            if contains_shell_prompt(&guard) {
                drop(guard);
                let payload = b"echo hello world;/bin/rvx-poweroff\r";
                let _ = child_stdin.write_all(payload);
                let _ = child_stdin.flush();
                break;
            }
            let wait_for = deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(250));
            let _ = notify.wait_timeout(guard, wait_for).unwrap();
        }
    });

    let status = child.wait().expect("failed to wait for rvx-cli");
    let stdout = stdout_reader.join().expect("stdout reader panicked");
    let stderr = stderr_reader.join().expect("stderr reader panicked");
    let _ = writer.join();
    Output {
        status,
        stdout,
        stderr,
    }
}

fn read_pipe<R: Read>(mut reader: R, observed_output: Arc<(Mutex<Vec<u8>>, Condvar)>) -> Vec<u8> {
    let mut collected = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(count) => {
                collected.extend_from_slice(&buf[..count]);
                let (observed, notify) = &*observed_output;
                let mut observed = observed.lock().unwrap();
                observed.extend_from_slice(&buf[..count]);
                notify.notify_all();
            }
            Err(_) => break,
        }
    }
    collected
}

fn contains_shell_prompt(output: &[u8]) -> bool {
    output
        .windows(b"rvx# ".len())
        .any(|window| window == b"rvx# ")
}

fn assert_successful_busybox_boot(output: &Output, perf_json: &PathBuf) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    assert!(
        output.status.success(),
        "emulator exited unsuccessfully:\n{combined}"
    );
    assert!(
        combined.contains("hello world"),
        "guest never printed hello world:\n{combined}"
    );
    assert!(
        combined.contains("Power down"),
        "guest did not power down cleanly:\n{combined}"
    );

    let perf = std::fs::read_to_string(perf_json).expect("missing perf json");
    let perf: serde_json::Value = serde_json::from_str(&perf).expect("invalid perf json");
    let runtime = &perf["runtime"];
    assert!(
        runtime["compiled_blocks"].as_u64().unwrap_or(0) > 0,
        "expected compiled blocks in runtime report:\n{perf}"
    );
    assert!(
        runtime["jitted_blocks_executed"].as_u64().unwrap_or(0) > 0,
        "expected jitted block executions in runtime report:\n{perf}"
    );
}

#[test]
fn linux_busybox_echo_hello_world() {
    let root = workspace_root();
    prepare_guest_artifacts(&root);
    let perf_json = root.join("target/linux_boot_perf.json");
    let output = run_linux_boot(&root, &perf_json, 1, 105_000, 32, false);
    assert_successful_busybox_boot(&output, &perf_json);
}

#[test]
#[ignore]
fn linux_busybox_echo_hello_world_threaded_smp() {
    let root = workspace_root();
    prepare_guest_artifacts(&root);
    let perf_json = root.join("target/linux_boot_perf_threaded_smp.json");
    let output = run_linux_boot(&root, &perf_json, 4, 240_000, 90, true);
    assert_successful_busybox_boot(&output, &perf_json);
}
