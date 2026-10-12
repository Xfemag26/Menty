// reflection.rs - Background idle reflection task
// Only runs when the user has been inactive for REFLECTION_IDLE_SECS (5 min).
// This prevents reflection from burning CPU/heat while the user is actively chatting.

use crate::config::*;
use crate::memory::Memory;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Call this every time the user sends a message so reflection backs off.
/// Stored as a shared atomic timestamp.
pub type LastActivityTs = Arc<std::sync::atomic::AtomicU64>;

pub fn new_activity_tracker() -> LastActivityTs {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Arc::new(std::sync::atomic::AtomicU64::new(ts))
}

pub fn record_activity(ts: &LastActivityTs) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    ts.store(now, std::sync::atomic::Ordering::Relaxed);
}

fn seconds_since_activity(ts: &LastActivityTs) -> u64 {
    let last = ts.load(std::sync::atomic::Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    now.saturating_sub(last)
}

/// Detect hallucination signals in a raw session string using pattern matching only.
/// No LLM call — pure string scan. Each detected pattern writes a diary entry.
///
/// Signals detected:
///   • Negative feedback after AI turn  ("wrong", "no", "you made that up", etc.)
///   • Failed tool call (AI emitted a path that returned ❌)
///   • User-supplied correction ("it's actually", "the correct answer is")
async fn audit_raw_session(memory: &Arc<Mutex<Memory>>, raw_text: &str) {
    // Split into lines for turn-by-turn analysis
    let lines: Vec<&str> = raw_text.lines().collect();

    // Negative feedback keywords — user explicitly correcting Ghostwater
    let neg_triggers: &[&str] = &[
        "wrong",
        "no that's",
        "not right",
        "incorrect",
        "you're wrong",
        "you made that up",
        "that's false",
        "hallucinating",
        "that's not",
        "actually it's",
        "actually the",
        "the correct",
        "it's actually",
        "you missed",
        "that didn't work",
        "that failed",
        "bad path",
    ];

    // Scan for user lines following an assistant turn that contain corrections.
    // Track the most recent assistant line so we can attribute the claim correctly.
    let mut last_was_assistant = false;
    let mut last_assistant_line = "";
    for line in &lines {
        let lower = line.to_ascii_lowercase();

        // Blank lines between turns must not reset the pairing — skip them.
        if line.trim().is_empty() {
            continue;
        }

        if lower.starts_with("ghostwater:") || lower.starts_with("assistant:") {
            last_was_assistant = true;
            last_assistant_line = line;
            continue;
        }
        if (lower.starts_with("user:") || lower.starts_with("you:")) && last_was_assistant {
            let correction_found = neg_triggers.iter().any(|t| lower.contains(t));
            if correction_found {
                let entry = format!(
                    "[hallucination-signal] claim: {} | correction: {}",
                    last_assistant_line.chars().take(120).collect::<String>(),
                    line.chars().take(120).collect::<String>()
                );
                println!("🔍 [REFLECTION] {}", entry);
                let mem = memory.lock().await;
                let _ = mem.add_diary_entry(&entry);
            }
        }
        // Any non-blank, non-assistant line ends the assistant→user pairing window.
        last_was_assistant = false;
    }

    // Detect failed tool calls — any line containing ❌ after a [TOOL:] line
    let mut last_tool_line = String::new();
    for line in &lines {
        if line.contains("[TOOL:") {
            last_tool_line = line.chars().take(120).collect();
        } else if line.contains('❌') && !last_tool_line.is_empty() {
            let entry = format!(
                "[failed-tool] tool: {} | error: {}",
                last_tool_line,
                line.chars().take(120).collect::<String>()
            );
            println!("🔍 [REFLECTION] {}", entry);
            let mem = memory.lock().await;
            let _ = mem.add_diary_entry(&entry);
            last_tool_line.clear();
        } else if !line.contains("[TOOL:") {
            last_tool_line.clear();
        }
    }
}

pub fn start_reflection_loop(
    handle: &tokio::runtime::Handle,
    memory: Arc<Mutex<Memory>>,
    activity: LastActivityTs,
) {
    handle.spawn(async move {
        let mut last_prune = Instant::now();

        loop {
            // Poll every 60s but only do work when genuinely idle
            tokio::time::sleep(Duration::from_secs(60)).await;

            let idle_secs = seconds_since_activity(&activity);
            if idle_secs < REFLECTION_IDLE_SECS {
                // User active recently — skip this cycle entirely
                continue;
            }

            // Web access is intentionally user-initiated. Idle reflection does
            // not search the network or update puzzles in the background.

            // 1. Audit raw sessions — pure string analysis, no LLM call needed.
            //    Reads up to 5 unaudited raw sessions per idle cycle, marks them done,
            //    then deletes them. Storage goes back to zero after this pass.
            {
                let raw_sessions = {
                    let mem = memory.lock().await;
                    mem.pop_unaudited_raw_sessions(5).unwrap_or_default()
                };
                for (_session_id, raw_text) in raw_sessions {
                    if seconds_since_activity(&activity) < REFLECTION_IDLE_SECS {
                        break;
                    }
                    audit_raw_session(&memory, &raw_text).await;
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }

            // 3. Synaptic pruning once every 24h, only when idle
            if last_prune.elapsed().as_secs() >= PRUNE_INTERVAL_SECS
                && seconds_since_activity(&activity) >= REFLECTION_IDLE_SECS
            {
                println!("✂️ [REFLECTION] Synaptic pruning...");
                let mem = memory.lock().await;
                if let Ok((edges, nodes)) = mem.prune(0.2) {
                    println!("✂️ Pruned {} weak edges, {} orphaned nodes", edges, nodes);
                }

                // Dual-speed strength decay: fast for sessions, evidence-slowed for semantic
                let _ = mem.decay_strengths();

                // Graph compaction: enforce MAX_LIVE_NODES cap, evict weakest disconnected leaves
                if let Ok(evicted) = mem.compact_nodes() {
                    if evicted > 0 {
                        println!("✂️ [REFLECTION] Compacted {} excess leaf nodes", evicted);
                    }
                }

                // Prune hw_samples ring-buffer (24h window, ~100KB max)
                if let Ok(n) = mem.prune_hw_samples() {
                    if n > 0 {
                        println!("✂️ Pruned {} old hw_samples", n);
                    }
                }

                // Safety-valve: prune any leftover raw session rows
                if let Ok(n) = mem.prune_raw_sessions() {
                    if n > 0 {
                        println!("✂️ Pruned {} stale raw_sessions", n);
                    }
                }

                // If weights have drifted enough, write one diary entry
                if let Ok(Some(lesson)) = mem.summarize_hw_lessons() {
                    let _ = mem.add_diary_entry(&lesson);
                    println!("📓 [REFLECTION] {}", lesson);
                }

                last_prune = Instant::now();
            }
        }
    });
}
