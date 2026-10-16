// guardian.rs - Adaptive two-tier resource guardian
// Tier 1 FAST (5s): raw /proc/meminfo + /proc/stat reads — ~0 CPU, no sysinfo.
//   Instantly catches 95% RAM/CPU and fires emergency mitigation.
// Tier 2 ADAPTIVE (300s): full sysinfo scan + HwCache update + top-process analysis.
//   Adjusts performance mode (low/balanced/turbo) so Momentum uses more when idle,
//   and tells the user what's hogging the machine.
//
// Follows recent lightweight practice: raw procfs for hot path, sysinfo only on
// deep scan/emergency — avoids refresh_all() on 15W i7-8550U / 8GB + MX150.
//
// HwCache: Arc<Mutex<HwSnapshot>> updated on both tiers. Zero extra threads
// beyond the single guardian thread (two interleaved sleeps).

use crate::config::*;
use crossbeam_channel::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::{Components, System};

#[derive(Debug, Clone)]
pub struct TopProcess {
    pub pid: u32,
    pub name: String,
    pub mem_mb: f32,
    pub cpu_pct: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerfMode {
    Low,      // RAM/CPU high — keep CTX 1024, throttle reflection
    Balanced, // normal — standard settings
    Turbo,    // idle, RAM <50% & CPU <30% — can use larger CTX/budget
}

impl PerfMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "LOW-POWER",
            Self::Balanced => "BALANCED",
            Self::Turbo => "TURBO",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuardianAlert {
    pub ram_percent: f32,
    pub used_gb: f32,
    pub total_gb: f32,
    pub cpu_percent: f32,
    pub temp_c: f32,
    pub is_critical: bool,
    pub is_emergency: bool,
    pub perf_mode: PerfMode,
    pub top_processes: Vec<TopProcess>,
    pub action_taken: Option<String>,
    /// Human-readable hog report — shown in UI/chat
    pub hog_report: Option<String>,
}

/// Lightweight snapshot of current RAM — safe to call on every LLM turn.
/// Uses refresh_memory() only (reads /proc/meminfo), never refresh_all().
#[derive(Debug, Clone)]
pub struct RamSnapshot {
    pub ram_percent: f32,
    pub used_gb: f32,
    pub total_gb: f32,
}

/// Full HW snapshot cached by the guardian thread on every 30s tick.
/// Reading this is a Mutex lock + struct clone — zero sysfs / proc I/O.
#[derive(Debug, Clone, Default)]
pub struct HwSnapshot {
    pub ram_pct: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub cpu_pct: f32,
    pub temp_c: f32,
}

/// Shared cache: written by guardian, read by prompt formatter + main.rs.
pub type HwCache = Arc<Mutex<HwSnapshot>>;

pub fn new_hw_cache() -> HwCache {
    let initial = read_meminfo_fast()
        .map(|(ram_pct, ram_used_gb, ram_total_gb)| HwSnapshot {
            ram_pct,
            ram_used_gb,
            ram_total_gb,
            ..HwSnapshot::default()
        })
        .unwrap_or_default();
    Arc::new(Mutex::new(initial))
}

#[derive(Debug, Clone)]
pub struct KillRequest {
    pub pid: u32,
    pub name: String,
}

/// Lightweight pending emergency kill — UI shows it and asks user to confirm.
/// Main.rs polls this via new channel; never kills without explicit user click.
pub type KillRequestSender = Sender<KillRequest>;
pub type KillRequestReceiver = crossbeam_channel::Receiver<KillRequest>;

/// Read CPU package temperature in °C using sysinfo::Components (one-shot hwmon
/// scan — no polling interval, effectively a single file read under /sys/class/hwmon).
///
/// Looks for the coretemp "Package id 0" sensor which is the authoritative
/// die-level reading on Intel mobile chips. Falls back to reading the raw
/// hwmon sysfs file directly if sysinfo returns nothing or NaN.
pub fn read_cpu_temp() -> Option<f32> {
    // Primary: sysinfo Components — enumerates /sys/class/hwmon once
    let components = Components::new_with_refreshed_list();
    for c in &components {
        let label = c.label();
        // sysinfo formats the label as "coretemp Package id 0 temp1" on this chip
        if label.contains("Package") || label.contains("package") {
            let t = c.temperature();
            if t.is_finite() && t > 0.0 {
                return Some(t);
            }
        }
    }

    // Fallback: direct sysfs read of /sys/class/hwmon/hwmon4/temp1_input
    // Value is in millidegrees Celsius.
    // hwmon4 = coretemp on this machine; scan a small range in case the index
    // shifts between boots.
    for idx in 0..8u8 {
        let path = format!("/sys/class/hwmon/hwmon{idx}/name");
        let Ok(name) = std::fs::read_to_string(&path) else {
            continue;
        };
        if name.trim() != "coretemp" {
            continue;
        }
        let temp_path = format!("/sys/class/hwmon/hwmon{idx}/temp1_input");
        if let Ok(raw) = std::fs::read_to_string(&temp_path) {
            if let Ok(millic) = raw.trim().parse::<i32>() {
                return Some(millic as f32 / 1000.0);
            }
        }
    }

    None
}

// ── Raw procfs fast path — zero sysinfo allocation, used by 5s emergency poll ──

fn read_meminfo_fast() -> Option<(f32, f32, f32)> {
    // Returns (ram_pct, used_gb, total_gb) from /proc/meminfo directly.
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb: u64 = 0;
    let mut avail_kb: u64 = 0;
    for line in s.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = line.split_whitespace().nth(1)?.parse().ok()?;
        } else if line.starts_with("MemAvailable:") {
            avail_kb = line.split_whitespace().nth(1)?.parse().ok()?;
        }
    }
    if total_kb == 0 {
        return None;
    }
    let used_kb = total_kb.saturating_sub(avail_kb);
    let total_gb = total_kb as f32 / (1024.0 * 1024.0);
    let used_gb = used_kb as f32 / (1024.0 * 1024.0);
    let pct = (used_kb as f32 / total_kb as f32) * 100.0;
    Some((pct, used_gb, total_gb))
}

fn read_cpu_fast(prev: &mut (u64, u64)) -> f32 {
    // Simple /proc/stat delta — returns 0..100, needs two samples.
    let s = match std::fs::read_to_string("/proc/stat") {
        Ok(v) => v,
        Err(_) => return 0.0,
    };
    let line = match s.lines().next() {
        Some(l) => l,
        None => return 0.0,
    };
    let nums: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    if nums.len() < 4 {
        return 0.0;
    }
    let idle = nums[3] + nums.get(4).copied().unwrap_or(0);
    let total: u64 = nums.iter().sum();
    let prev_total = prev.0;
    let prev_idle = prev.1;
    *prev = (total, idle);
    if prev_total == 0 {
        return 0.0;
    }
    let dt = total.saturating_sub(prev_total) as f32;
    let di = idle.saturating_sub(prev_idle) as f32;
    if dt == 0.0 {
        0.0
    } else {
        ((dt - di) / dt * 100.0).clamp(0.0, 100.0)
    }
}

fn top_processes(n: usize) -> Vec<TopProcess> {
    // Only call on emergency/adaptive scan — does process scan.
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    sys.refresh_cpu_usage();
    // Give cpu_usage a moment to be non-zero — but don't sleep; use second refresh if available elsewhere.
    let mut procs: Vec<TopProcess> = sys
        .processes()
        .values()
        .map(|p| TopProcess {
            pid: p.pid().as_u32(),
            name: p.name().to_string_lossy().into_owned(),
            mem_mb: p.memory() as f32 / (1024.0 * 1024.0),
            cpu_pct: p.cpu_usage(),
        })
        .collect();
    procs.sort_by(|a, b| {
        b.mem_mb
            .partial_cmp(&a.mem_mb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Merge memory-top and cpu-top to surface both kinds of hogs
    let mut cpu_sorted = procs.clone();
    cpu_sorted.sort_by(|a, b| {
        b.cpu_pct
            .partial_cmp(&a.cpu_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for p in procs.into_iter().chain(cpu_sorted.into_iter()) {
        if seen.insert(p.pid) {
            out.push(p);
            if out.len() >= n {
                break;
            }
        }
    }
    out.sort_by(|a, b| {
        b.mem_mb
            .partial_cmp(&a.mem_mb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(n);
    out
}

fn format_hog_report(top: &[TopProcess], ram_pct: f32, cpu_pct: f32, temp_c: f32) -> String {
    let mut s = format!(
        "🧠 TOP HOGS — RAM {:.0}% CPU {:.0}% Temp {:.0}°C\n",
        ram_pct, cpu_pct, temp_c
    );
    for tp in top {
        s.push_str(&format!(
            "  • {} (pid {}) — {:.0} MB  CPU {:.0}%\n",
            tp.name, tp.pid, tp.mem_mb, tp.cpu_pct
        ));
    }
    s.push_str("Ask me to kill any pid, or type 'kill <pid>' — I will confirm before doing it.");
    s
}

fn perf_mode_for(ram_pct: f32, cpu_pct: f32) -> PerfMode {
    if ram_pct < 50.0 && cpu_pct < 30.0 {
        PerfMode::Turbo
    } else if ram_pct >= RAM_CRITICAL_PERCENT || cpu_pct >= CPU_EMERGENCY_PERCENT {
        PerfMode::Low
    } else {
        PerfMode::Balanced
    }
}

/// Read current RAM usage. Returns None only if sysinfo reports total=0
/// (shouldn't happen on Linux, but guard anyway).
pub fn read_ram() -> Option<RamSnapshot> {
    if let Some((pct, used, total)) = read_meminfo_fast() {
        return Some(RamSnapshot {
            ram_percent: pct,
            used_gb: used,
            total_gb: total,
        });
    }
    let mut sys = System::new();
    sys.refresh_memory();
    let total_b = sys.total_memory();
    if total_b == 0 {
        return None;
    }
    let used_b = sys.used_memory();
    Some(RamSnapshot {
        ram_percent: (used_b as f32 / total_b as f32) * 100.0,
        used_gb: used_b as f32 / (1u64 << 30) as f32,
        total_gb: total_b as f32 / (1u64 << 30) as f32,
    })
}

/// One-line hardware state string for injection into the LLM system prompt.
/// Reads from the shared HwCache — zero sysfs/proc reads at inference time.
///
/// Example:
///   [Hardware: RAM 5.2/7.6 GB (68%) | CPU 34% 71°C | risk 0.12 | warn ...]
pub fn format_hw_state_for_prompt(cache: &HwCache) -> String {
    let snap = cache.lock().map(|g| g.clone()).unwrap_or_default();

    let ram_part = if snap.ram_total_gb > 0.0 {
        format!(
            "RAM {:.1}/{:.1} GB ({:.0}%)",
            snap.ram_used_gb, snap.ram_total_gb, snap.ram_pct
        )
    } else {
        format!("RAM {:.0}%", snap.ram_pct)
    };

    let cpu_part = if snap.cpu_pct > 0.0 || snap.temp_c > 0.0 {
        format!("CPU {:.0}% {:.0}°C", snap.cpu_pct, snap.temp_c)
    } else {
        "CPU unknown".to_string()
    };

    format!(
        "[Hardware: {} | {} | warn RAM>{:.0}% temp>{:.0}°C critical RAM>{:.0}% temp>{:.0}°C — stay brief if high]",
        ram_part, cpu_part,
        RAM_WARN_PERCENT, CPU_TEMP_WARN_C,
        RAM_CRITICAL_PERCENT, CPU_TEMP_CRITICAL_C,
    )
}

pub fn start_guardian(alert_tx: Sender<GuardianAlert>, hw_cache: HwCache) {
    start_guardian_with_kill(alert_tx, hw_cache, None);
}
pub fn start_guardian_with_kill(
    alert_tx: Sender<GuardianAlert>,
    hw_cache: HwCache,
    _kill_tx: Option<KillRequestSender>,
) {
    guardian_loop(alert_tx, hw_cache);
}
fn guardian_loop(alert_tx: Sender<GuardianAlert>, hw_cache: HwCache) {
    std::thread::spawn(move || {
        let mut sys = System::new();
        let mut smooth_count: u32 = 0;
        let mut cpu_prev: (u64, u64) = (0, 0);
        let mut last_adaptive = Instant::now() - Duration::from_secs(GUARDIAN_ADAPTIVE_SECS);
        let _ = read_cpu_fast(&mut cpu_prev);
        let mut last_perf = PerfMode::Balanced;
        loop {
            std::thread::sleep(Duration::from_secs(GUARDIAN_FAST_SECS));
            let (ram_pct, used_gb, total_gb) = read_meminfo_fast().unwrap_or((0.0, 0.0, 0.0));
            if total_gb == 0.0 {
                continue;
            }
            let cpu_pct = read_cpu_fast(&mut cpu_prev);
            let is_emergency = ram_pct >= RAM_EMERGENCY_PERCENT || cpu_pct >= CPU_EMERGENCY_PERCENT;
            if let Ok(mut g) = hw_cache.lock() {
                g.ram_pct = ram_pct;
                g.ram_used_gb = used_gb;
                g.ram_total_gb = total_gb;
                g.cpu_pct = cpu_pct;
            }
            if is_emergency {
                let action = std::fs::write("/proc/sys/vm/drop_caches", "1")
                    .ok()
                    .map(|_| "Dropped OS page caches (level 1)".to_string());
                let top = top_processes(GUARDIAN_TOP_N);
                let temp_c = read_cpu_temp().unwrap_or(0.0);
                if let Ok(mut g) = hw_cache.lock() {
                    g.temp_c = temp_c;
                }
                let hog = format_hog_report(&top, ram_pct, cpu_pct, temp_c);
                let _ = alert_tx.send(GuardianAlert {
                    ram_percent: ram_pct,
                    used_gb,
                    total_gb,
                    cpu_percent: cpu_pct,
                    temp_c,
                    is_critical: true,
                    is_emergency: true,
                    perf_mode: PerfMode::Low,
                    top_processes: top,
                    action_taken: action,
                    hog_report: Some(hog),
                });
                smooth_count = 0;
                last_perf = PerfMode::Low;
                continue;
            }
            let do_adaptive = last_adaptive.elapsed().as_secs() >= GUARDIAN_ADAPTIVE_SECS;
            if do_adaptive {
                last_adaptive = Instant::now();
                sys.refresh_memory();
                sys.refresh_cpu_usage();
                let sys_cpu = {
                    let c = sys.cpus();
                    if c.is_empty() {
                        cpu_pct
                    } else {
                        c.iter().map(|x| x.cpu_usage()).sum::<f32>() / c.len() as f32
                    }
                };
                let temp_c = read_cpu_temp().unwrap_or(0.0);
                if let Ok(mut g) = hw_cache.lock() {
                    let tb = sys.total_memory();
                    let ub = sys.used_memory();
                    if tb > 0 {
                        g.ram_pct = (ub as f32 / tb as f32) * 100.0;
                        g.ram_used_gb = ub as f32 / (1u64 << 30) as f32;
                        g.ram_total_gb = tb as f32 / (1u64 << 30) as f32;
                    }
                    g.cpu_pct = sys_cpu;
                    g.temp_c = temp_c;
                }
                let (e_ram, e_cpu, e_temp) = hw_cache
                    .lock()
                    .map(|g| (g.ram_pct, g.cpu_pct, g.temp_c))
                    .unwrap_or((ram_pct, sys_cpu, temp_c));
                let perf = perf_mode_for(e_ram, e_cpu);
                let is_crit = e_ram >= RAM_CRITICAL_PERCENT || e_temp >= CPU_TEMP_CRITICAL_C;
                let is_warn = e_ram >= RAM_WARN_PERCENT || e_temp >= CPU_TEMP_WARN_C;
                let is_smooth =
                    e_ram < RAM_WARN_PERCENT && (e_temp == 0.0 || e_temp < CPU_TEMP_WARN_C);
                if is_smooth {
                    smooth_count += 1;
                    if smooth_count >= SMOOTH_STREAK {
                        smooth_count = 0;
                        if let Ok(m) = crate::memory::Memory::open() {
                            let _ = m.label_hw_samples(i64::MAX, 0);
                        }
                    }
                } else {
                    smooth_count = 0;
                }
                if is_crit || is_warn || perf != last_perf {
                    let action = if is_crit {
                        std::fs::write("/proc/sys/vm/drop_caches", "1")
                            .ok()
                            .map(|_| "Dropped OS page caches (level 1)".to_string())
                    } else {
                        None
                    };
                    let top = if is_crit || perf != last_perf {
                        top_processes(GUARDIAN_TOP_N)
                    } else {
                        vec![]
                    };
                    let hog = if !top.is_empty() {
                        Some(format_hog_report(&top, e_ram, e_cpu, e_temp))
                    } else {
                        None
                    };
                    let _ = alert_tx.send(GuardianAlert {
                        ram_percent: e_ram,
                        used_gb,
                        total_gb,
                        cpu_percent: e_cpu,
                        temp_c: e_temp,
                        is_critical: is_crit,
                        is_emergency: false,
                        perf_mode: perf.clone(),
                        top_processes: top,
                        action_taken: action,
                        hog_report: hog,
                    });
                    last_perf = perf;
                }
            } else {
                if ram_pct >= RAM_CRITICAL_PERCENT {
                    let top = top_processes(GUARDIAN_TOP_N);
                    let hog = format_hog_report(&top, ram_pct, cpu_pct, 0.0);
                    let _ = alert_tx.send(GuardianAlert {
                        ram_percent: ram_pct,
                        used_gb,
                        total_gb,
                        cpu_percent: cpu_pct,
                        temp_c: 0.0,
                        is_critical: true,
                        is_emergency: false,
                        perf_mode: PerfMode::Low,
                        top_processes: top,
                        action_taken: None,
                        hog_report: Some(hog),
                    });
                } else if ram_pct >= RAM_WARN_PERCENT && last_perf != PerfMode::Low {
                    let _ = alert_tx.send(GuardianAlert {
                        ram_percent: ram_pct,
                        used_gb,
                        total_gb,
                        cpu_percent: cpu_pct,
                        temp_c: 0.0,
                        is_critical: false,
                        is_emergency: false,
                        perf_mode: PerfMode::Balanced,
                        top_processes: vec![],
                        action_taken: None,
                        hog_report: None,
                    });
                }
            }
        }
    });
}
