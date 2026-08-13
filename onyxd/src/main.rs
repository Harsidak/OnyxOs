//! # OnyxD — The OnyxOS Central Daemon
//!
//! The brain of OnyxOS. Runs as a long-lived daemon process
//! launched by onyx-init, orchestrating:
//!
//! - Thermal monitoring (reads sysfs thermal zones)
//! - Power monitoring (reads INA3221 via sysfs)
//! - Market scheduler (Lagrangian auction via embedded Python/Cython)
//! - Agent lifecycle management
//! - cgroup resource enforcement
//! - IPC server for agent communication
//!
//! For the MVP, the market scheduler runs in embedded Python.
//! In production, it will be rewritten in Rust with the same
//! Lagrangian dual-ascent algorithm.

use std::path::Path;
use std::time::Duration;

/// Read a temperature from a Jetson thermal zone (millidegrees → Celsius)
fn read_thermal_zone(zone: u32) -> Option<f64> {
    let path = format!("/sys/devices/virtual/thermal/thermal_zone{}/temp", zone);
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .map(|mc| mc as f64 / 1000.0)
}

/// Read power draw from INA3221 (milliwatts → Watts)
fn read_power_draw() -> Option<f64> {
    let path = "/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon3/in1_input";
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .map(|mw| mw as f64 / 1000.0)
}

/// Write a value to a cgroup control file
fn write_cgroup(agent_class: &str, control: &str, value: &str) -> std::io::Result<()> {
    let path = format!("/sys/fs/cgroup/onyx/{}/{}", agent_class, control);
    std::fs::write(&path, value)
}

/// OnyxD main loop
#[tokio::main]
async fn main() {
    println!();
    println!("  ◆ OnyxD v0.1.0 — OnyxOS Central Daemon");
    println!("  ────────────────────────────────────────");
    println!();

    // Check if running on Jetson
    let is_jetson = Path::new("/sys/devices/virtual/thermal/thermal_zone0/temp").exists();
    if is_jetson {
        println!("[onyxd] Running on NVIDIA Jetson — live hardware mode");
    } else {
        println!("[onyxd] Not on Jetson — simulation mode");
        println!("[onyxd] Delegating to Python market scheduler...");

        // In non-Jetson mode, delegate to the Python scheduler
        let status = std::process::Command::new("python3")
            .args(["-m", "onyx_hud.dashboard"])
            .status();

        match status {
            Ok(s) => println!("[onyxd] Python scheduler exited: {}", s),
            Err(e) => eprintln!("[onyxd] Failed to launch Python scheduler: {}", e),
        }
        return;
    }

    // ── Production mode: Jetson hardware ──

    // Read initial thermal state
    let temp = read_thermal_zone(0).unwrap_or(25.0);
    println!("[onyxd] Initial CPU temp: {:.1}°C", temp);

    let power = read_power_draw().unwrap_or(0.0);
    println!("[onyxd] Initial power draw: {:.1}W", power);

    // Main loop: read sensors, run auction, enforce decisions
    println!("[onyxd] Starting market ticker (1Hz)...");

    let mut tick: u64 = 0;
    loop {
        tick += 1;

        // Read thermal zones
        let cpu_temp = read_thermal_zone(0).unwrap_or(25.0);
        let gpu_temp = read_thermal_zone(1).unwrap_or(25.0);
        let soc_temp = read_thermal_zone(6).unwrap_or(25.0);
        let power_w = read_power_draw().unwrap_or(0.0);

        // Log status
        if tick % 10 == 0 {
            println!(
                "[onyxd] tick={} cpu={:.1}°C gpu={:.1}°C soc={:.1}°C power={:.1}W",
                tick, cpu_temp, gpu_temp, soc_temp, power_w
            );
        }

        // TODO: Call into Python market scheduler via PyO3
        // TODO: Apply auction results via ioctl to onyx_sched.ko
        // TODO: Enforce cgroup limits based on market decisions

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
