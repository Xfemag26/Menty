// main.rs - egui UI + conversation loop + process lifecycle
// Replaces: harness.py, assistat.py, run_momentum.sh

mod action_log;
mod config;
mod guardian;
mod llm;
mod memory;
mod permissions;
mod reflection;
mod telemetry;
mod tools;

use chrono::Local;
use crossbeam_channel::{unbounded, Receiver};
use eframe::egui;
use std::sync::Arc;
use tokio::sync::Mutex;

use config::*;
use guardian::{
    format_hw_state_for_prompt, new_hw_cache, start_guardian, GuardianAlert, HwCache, PerfMode,
};
use llm::{complete_stream_with_profile, GpuStatus, InferenceRuntime};
use memory::Memory;
use reflection::{new_activity_tracker, record_activity, start_reflection_loop, LastActivityTs};
use tools::dispatch_with_memory;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppMode {
    AiChat,
    ServerConnect,
}

#[derive(Clone)]
struct ChatMessage {
    role: String, // "user" | "assistant" | "system" | "tool"
    content: String,
    time: String,
}

struct MomentumApp {
    app_mode: AppMode,
    remote_url: String,
    remote_api_key: String,
    messages: Vec<ChatMessage>,
    input_text: String,
    memory: Arc<Mutex<Memory>>,
    guardian_rx: Receiver<GuardianAlert>,
    last_alert: Option<GuardianAlert>,
    is_generating: bool,
    tokio_runtime: tokio::runtime::Handle,
    chat_tx: crossbeam_channel::Sender<ChatMessage>,
    chat_rx: Receiver<ChatMessage>,
    activity: LastActivityTs,
    hw_cache: HwCache,
    // Native Rust inference runtime — owns the active model and CUDA allocations.
    _inference_runtime: Option<InferenceRuntime>,
    gpu_status: Option<GpuStatus>,
    gpu_retry_prompt: bool,
    inference_error: Option<String>,
    // Model switcher: index into config::MODEL_PROFILES
    active_model_idx: usize,
    // Embed server intentionally NOT kept alive here; started on-demand in memory.rs
    // Voice input
    is_recording: bool,
    record_process: Option<std::process::Child>,
    mic_pressed_at: Option<std::time::Instant>,
    // NMOS Zero-Lag prefetch tracker
    last_prefetch: std::time::Instant,
}

impl MomentumApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        memory: Arc<Mutex<Memory>>,
        guardian_rx: Receiver<GuardianAlert>,
        tokio_runtime: tokio::runtime::Handle,
        inference_runtime: Option<InferenceRuntime>,
        inference_error: Option<String>,
        activity: LastActivityTs,
        hw_cache: HwCache,
    ) -> Self {
        let mut visuals = egui::Visuals::dark();
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(18, 18, 24);
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(30, 30, 40);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(45, 45, 60);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 60, 80);
        visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);
        visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);
        visuals.widgets.active.rounding = egui::Rounding::same(8.0);
        visuals.widgets.noninteractive.rounding = egui::Rounding::same(8.0);

        visuals.window_fill = egui::Color32::from_rgb(18, 18, 24);
        visuals.panel_fill = egui::Color32::from_rgb(18, 18, 24);
        visuals.selection.bg_fill = egui::Color32::from_rgb(100, 220, 180);
        cc.egui_ctx.set_visuals(visuals);
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(10.0, 10.0);
        style.spacing.window_margin = egui::Margin::same(12.0);
        cc.egui_ctx.set_style(style);

        let (chat_tx, chat_rx) = unbounded();

        let active_model_idx = 0usize; // default = first profile (Qwen 2.5 Coder 3B)
        let gpu_status = inference_runtime
            .as_ref()
            .map(|runtime| runtime.gpu_status.clone());
        let gpu_retry_prompt = gpu_status
            .as_ref()
            .map(GpuStatus::is_constrained)
            .unwrap_or(false);
        let mut app = Self {
            app_mode: AppMode::AiChat,
            remote_url: String::new(),
            remote_api_key: String::new(),
            messages: vec![ChatMessage {
                role: "system".into(),
                content: format!(
                    "🌊 Ghostwater initialized. Model: {}",
                    config::MODEL_PROFILES[active_model_idx].name
                ),
                time: Local::now().format("%H:%M").to_string(),
            }],
            input_text: String::new(),
            memory,
            guardian_rx,
            last_alert: None,
            is_generating: false,
            tokio_runtime,
            chat_tx,
            chat_rx,
            activity,
            hw_cache,
            _inference_runtime: inference_runtime,
            gpu_status,
            gpu_retry_prompt,
            inference_error,
            active_model_idx,
            is_recording: false,
            record_process: None,
            mic_pressed_at: None,
            last_prefetch: std::time::Instant::now(),
        };

        if let Some(error) = app.inference_error.take() {
            app.messages.push(ChatMessage {
                role: "system".into(),
                content: format!("❌ Native inference could not load a model: {error}"),
                time: Local::now().format("%H:%M").to_string(),
            });
        }
        if let Some(gpu) = app.gpu_status.as_ref() {
            if gpu.is_constrained() {
                app.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!("GPU allocation is adaptive: {} of {} target layers loaded. The next layer needs about {} MiB of additional free VRAM. You can close GPU applications and choose Retry; Momentum will not terminate them.", gpu.loaded_layers, gpu.target_layers, gpu.additional_mb_needed),
                    time: Local::now().format("%H:%M").to_string(),
                });
            }
        }
        if let Ok(mem) = app.memory.try_lock() {
            // Show boot crash notice if the previous session didn't exit cleanly
            match mem.mark_crash_if_unclean() {
                Ok(true) => {
                    app.messages.push(ChatMessage {
                        role: "system".into(),
                        content: "⚠️ [GUARDIAN] Previous session ended uncleanly (crash/OOM). HW risk model updated.".into(),
                        time: Local::now().format("%H:%M").to_string(),
                    });
                    // Label any open samples from the previous session as bad (outcome=1)
                    // Use i64::MAX as cutoff — all open samples belong to the dead session
                    let _ = mem.label_hw_samples(i64::MAX, 1);
                }
                Ok(false) => {
                    // Previous session was clean: label open samples as smooth (outcome=0)
                    let _ = mem.label_hw_samples(i64::MAX, 0);
                }
                Err(_) => {}
            }

            if let Ok(diary) = mem.get_diary(3) {
                for entry in diary.into_iter().rev() {
                    app.messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!("💾 [MEMORY]: {entry}"),
                        time: "".into(),
                    });
                }
            }
        }
        app
    }

    /// Load a different model into the in-process Rust inference engine.
    fn switch_model(&mut self, idx: usize) {
        if idx >= config::MODEL_PROFILES.len() {
            return;
        }
        let profile = &config::MODEL_PROFILES[idx];

        // Resource warning: check available RAM against profile threshold
        let ram_ok = guardian::read_ram()
            .map(|s| (s.total_gb - s.used_gb) >= profile.ram_warn_gb)
            .unwrap_or(true);

        let warn = if !ram_ok {
            format!(" ⚠️ Low RAM — {} may be slow or unstable.", profile.name)
        } else {
            String::new()
        };

        self.messages.push(ChatMessage {
            role: "system".into(),
            content: format!("🔄 Switching to {}…{}", profile.name, warn),
            time: Local::now().format("%H:%M").to_string(),
        });

        // Drop the previous engine first so its resident GPU layers are freed.
        self._inference_runtime = None;
        self.gpu_status = None;
        self.gpu_retry_prompt = false;

        match InferenceRuntime::start_with_profile(profile) {
            Ok(runtime) => {
                self.active_model_idx = idx;
                self.gpu_status = Some(runtime.gpu_status.clone());
                if runtime.gpu_status.is_constrained() {
                    self.gpu_retry_prompt = true;
                }
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!(
                        "✅ Now using {} (GPU layers {}/{}, context {}, output ≤{} tokens){}",
                        profile.name,
                        runtime.gpu_status.loaded_layers,
                        runtime.gpu_status.target_layers,
                        profile.ctx_size,
                        profile.max_tokens,
                        warn
                    ),
                    time: Local::now().format("%H:%M").to_string(),
                });
                if runtime.gpu_status.is_constrained() {
                    self.messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!(
                            "GPU memory is limited: {} layer(s) loaded, {} requested; about {} MiB more VRAM is needed for the next layer. Close GPU apps you recognize, then use Retry. Momentum will not close anything for you.",
                            runtime.gpu_status.loaded_layers,
                            runtime.gpu_status.target_layers,
                            runtime.gpu_status.additional_mb_needed
                        ),
                        time: Local::now().format("%H:%M").to_string(),
                    });
                }
                self._inference_runtime = Some(runtime);
                let memory = self.memory.clone();
                self.tokio_runtime.spawn(async move {
                    let mem = memory.lock().await;
                    if let Ok(count) = mem.backfill_missing_embeddings(24) {
                        if count > 0 {
                            println!("Re-embedded {count} nodes for the selected model");
                        }
                    }
                });
            }
            Err(e) => {
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!("❌ Model switch failed: {e}"),
                    time: Local::now().format("%H:%M").to_string(),
                });
            }
        }
    }

    fn send_message(&mut self, text: String) {
        if text.trim().is_empty() || self.is_generating {
            return;
        }

        // /model command — cheapest possible switcher, no inference needed
        if text.trim().starts_with("/model") {
            let arg = text
                .trim()
                .trim_start_matches("/model")
                .trim()
                .to_lowercase();
            if arg == "list" || arg.is_empty() {
                let list = config::MODEL_PROFILES
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let active = if i == self.active_model_idx {
                            " ← active"
                        } else {
                            ""
                        };
                        format!("[{}] {}{}\n    {}", i, p.name, active, p.description)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!(
                        "Available models:\n{list}\n\nType /model <number> to switch."
                    ),
                    time: Local::now().format("%H:%M").to_string(),
                });
            } else if let Ok(idx) = arg.parse::<usize>() {
                self.switch_model(idx);
            } else {
                // Match by partial name
                let found = config::MODEL_PROFILES
                    .iter()
                    .position(|p| p.name.to_lowercase().contains(&arg));
                match found {
                    Some(idx) => self.switch_model(idx),
                    None => {
                        self.messages.push(ChatMessage {
                            role: "system".into(),
                            content: format!(
                                "Unknown model '{}'. Type /model list to see options.",
                                arg
                            ),
                            time: Local::now().format("%H:%M").to_string(),
                        });
                    }
                }
            }
            return;
        }

        // Record activity so reflection loop knows not to run
        record_activity(&self.activity);

        let user_msg = ChatMessage {
            role: "user".into(),
            content: text.clone(),
            time: Local::now().format("%H:%M").to_string(),
        };
        self.messages.push(user_msg);
        self.input_text.clear();
        self.is_generating = true;

        let lower = text.to_lowercase();
        let is_positive =
            lower.contains("perfect") || lower.contains("nice") || lower.contains("lol");
        // Word-boundary check: avoid "no" matching "node", "know", "notation", etc.
        let is_negative = lower.contains("wrong")
            || lower.contains("bad")
            || lower
                .split_whitespace()
                .any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == "no");

        let mem_clone = self.memory.clone();
        let prompt_input = text.clone();
        let tx = self.chat_tx.clone();
        let hw_cache_clone = self.hw_cache.clone();
        let active_profile_idx = self.active_model_idx;
        let active_max_tokens = config::MODEL_PROFILES[active_profile_idx].max_tokens;
        let history: Vec<_> = self
            .messages
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .rev()
            .take(MAX_HISTORY_TURNS)
            .cloned()
            .collect();

        self.tokio_runtime.spawn(async move {
            // route_context: local-model vector recall with FTS5 fallback and
            // bounded one-hop graph neighbors; capped to keep prompts compact.
            let context = {
                let mem = mem_clone.lock().await;
                let routed = mem.route_context(&prompt_input);
                if routed.is_empty() {
                    mem.get_context(&prompt_input)
                } else {
                    routed
                }
            };

            // raw_snapshot built first (iter, no move) for store_session_with_raw.
            let raw_snapshot = history.iter()
                .rev()
                .map(|m| format!("{}: {}", if m.role == "user" { "User" } else { "Ghostwater" }, m.content))
                .collect::<Vec<_>>()
                .join("\n");

            // Build history as separate im_start turns so the model sees them as
            // distinct exchanges, not as part of the current user message.
            // Fix #3: history consumed here (into_iter) after raw_snapshot is done.
            let history_turns = history.into_iter().rev()
                .map(|m| {
                    let role = if m.role == "user" { "user" } else { "assistant" };
                    format!("<|im_start|>{}\n{}\n<|im_end|>", role, m.content)
                })
                .collect::<Vec<_>>()
                .join("\n");

            // HW state read from cache — zero /proc or sysfs I/O on the hot path.
            let ram_state = format_hw_state_for_prompt(&hw_cache_clone);

            // Sample current HW state before inference starts (generating=true).
            let sample_id: Option<i64> = {
                let snap = hw_cache_clone.lock().map(|g| g.clone()).unwrap_or_default();
                let mem = mem_clone.lock().await;
                mem.insert_hw_sample(snap.ram_pct, snap.cpu_pct, snap.temp_c, true).ok()
            };

            let home = config::home_dir().to_string_lossy().into_owned();
            let desktop = config::home_dir().join("Desktop").to_string_lossy().into_owned();
            let env_hint = format!("Absolute paths on this machine: HOME={} DESKTOP={} — NEVER use /home/user. Always use the real HOME above for file tools.", home, desktop);
            // Conditional tool-docs injection: plain conversation turns skip the ~200-token
            // PERSONA_TOOLS block. needs_tools() is a pure byte scan, zero allocations.
            let persona_block = if needs_tools(&prompt_input) {
                format!("{}\n{}", PERSONA_CORE, PERSONA_TOOLS)
            } else {
                PERSONA_CORE.to_string()
            };
            let profile = &config::MODEL_PROFILES[active_profile_idx];
            let system_part = format!("{}\n{}\n{}\n\nVerified Context:\n{}", persona_block, ram_state, env_hint, context);
            let prompt = config::format_prompt(profile, &system_part, &history_turns, &prompt_input);

            // Seed an empty assistant bubble so delta messages grow it.
            let _ = tx.send(ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                time: Local::now().format("%H:%M").to_string(),
            });
            let tx_stream = tx.clone();
            let raw_response = complete_stream_with_profile(profile, &prompt, active_max_tokens, |piece| {
                let _ = tx_stream.send(ChatMessage {
                    role: "delta".into(),
                    content: piece.to_string(),
                    time: String::new(),
                });
            }).unwrap_or_else(|e| format!("Error: {e}"));

            // Close the pre-inference sample as smooth (outcome=0).
            // Crash labelling happens at boot via mark_crash_if_unclean().
            if let Some(sid) = sample_id {
                let mem = mem_clone.lock().await;
                let _ = mem.label_hw_samples(sid, 0);
            }

            // ── Agentic tool loop ─────────────────────────────────────────────
            // Runs until the model stops emitting [TOOL:] tags or hits MAX_TOOL_DEPTH.
            // Each iteration: execute all tags → feed results back → re-infer.
            // This allows genuine multi-step autonomy (screen_read → find_and_click
            // → screen_read → shell → notify) without any artificial one-round limit.
            const MAX_TOOL_DEPTH: usize = 8; // prevents runaway loops
            // Cap per-tool result to 600 chars in the context accumulator.
            // Full output is shown in the UI; only a digest enters the prompt.
            const TOOL_CTX_PER_RESULT: usize = 600;

            let mut current_response = raw_response.clone();
            let mut accumulated_tool_ctx = String::new();
            let mut final_ans = String::new();
            let mut used_tools = false;

            for _depth in 0..MAX_TOOL_DEPTH {
                let tool_lines: Vec<String> = current_response.lines()
                    .filter(|l| l.contains("[TOOL:"))
                    .map(|l| l.trim().to_string())
                    .collect();

                if tool_lines.is_empty() {
                    // No more tool calls — this is the final answer
                    final_ans = current_response.clone();
                    break;
                }

                used_tools = true;

                // Show the model's tool-call turn in the chat (once per iteration)
                let _ = tx.send(ChatMessage {
                    role: "assistant".into(),
                    content: current_response.clone(),
                    time: Local::now().format("%H:%M").to_string(),
                });

                // Execute all tool tags in this response
                let mut round_ctx = String::new();
                for tool_str in &tool_lines {
                    let _ = tx.send(ChatMessage {
                        role: "tool".into(),
                        content: format!("→ {}", tool_str),
                        time: Local::now().format("%H:%M").to_string(),
                    });

                    if let Some(result) = dispatch_with_memory(tool_str, &mem_clone).await {
                        // Full output shown in UI
                        let _ = tx.send(ChatMessage {
                            role: "tool".into(),
                            content: result.output.chars().take(2000).collect::<String>(),
                            time: Local::now().format("%H:%M").to_string(),
                        });
                        // Digest for prompt — keeps context window sane
                        let digest: String = result.output.chars().take(TOOL_CTX_PER_RESULT).collect();
                        round_ctx.push_str(&format!("\n{}→ {}", tool_str, digest));
                    }
                }

                // Accumulate — but hard-cap total context to avoid blowing the budget
                accumulated_tool_ctx.push_str(&round_ctx);
                if accumulated_tool_ctx.len() > 2400 {
                    // Keep only the last 2400 chars (most recent results are most relevant)
                    let start = accumulated_tool_ctx.len() - 2400;
                    accumulated_tool_ctx = accumulated_tool_ctx[start..].to_string();
                }

                // Re-infer with results + history so the model keeps context across tool rounds
                let profile = &config::MODEL_PROFILES[active_profile_idx];
                let sys_follow = format!("{}\n{}\nTool results (latest):{}", PERSONA_CORE, ram_state, accumulated_tool_ctx);
                let follow_up = config::format_prompt(profile, &sys_follow, &history_turns, &prompt_input);
                current_response = llm::complete_with_profile(profile, &follow_up, active_max_tokens)
                    .unwrap_or_else(|e| format!("Error: {e}"));
            }

            // If we hit MAX_TOOL_DEPTH without a clean exit, use the last response as answer
            if final_ans.is_empty() {
                final_ans = current_response.clone();
                let _ = tx.send(ChatMessage {
                    role: "system".into(),
                    content: format!("⚠️ Tool depth limit ({MAX_TOOL_DEPTH}) reached."),
                    time: Local::now().format("%H:%M").to_string(),
                });
            }

            // Send the final answer (always — guarantees is_generating resets)
            let _ = tx.send(ChatMessage {
                role: "assistant".into(),
                content: final_ans.clone(),
                time: Local::now().format("%H:%M").to_string(),
            });

            // Store session — use final answer if tools ran, raw response if not
            let to_store = if used_tools { &final_ans } else { &raw_response };
            {
                let mem = mem_clone.lock().await;
                let _ = mem.store_session_with_raw(
                    &prompt_input, to_store, "neutral", &raw_snapshot,
                );
            }

            if is_positive || is_negative {
                let delta = if is_positive { 0.2 } else { -0.2 };
                let keywords: Vec<&str> = prompt_input.split_whitespace().take(3).collect();
                let mem = mem_clone.lock().await;
                let _ = mem.boost_recent_nodes(&keywords, delta);
            }
        });
    }

    fn start_recording(&mut self) {
        let child = std::process::Command::new("arecord")
            .args([
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "/tmp/momentum_voice.wav",
            ])
            .spawn();
        match child {
            Ok(c) => {
                self.record_process = Some(c);
                self.is_recording = true;
            }
            Err(e) => eprintln!("arecord spawn failed: {e}"),
        }
    }

    fn stop_recording_and_transcribe(&mut self) {
        if let Some(mut child) = self.record_process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.is_recording = false;
        let model = crate::config::whisper_model_path();
        let cli = crate::config::whisper_cli_path();
        let threads = crate::config::N_THREADS.to_string();
        std::thread::spawn(move || {
            let model_arg = model.to_string_lossy().into_owned();
            let out = std::process::Command::new(cli)
                .args([
                    "-m",
                    &model_arg,
                    "-nt",
                    "-t",
                    &threads,
                    "/tmp/momentum_voice.wav",
                ])
                .output();
            match out {
                Ok(o) => {
                    let text = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    if !text.is_empty() {
                        let _ = std::fs::write("/tmp/momentum_voice_result.txt", text);
                    }
                }
                Err(e) => eprintln!("whisper-cli failed: {e}"),
            }
        });
    }
}

impl eframe::App for MomentumApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll incoming chat messages from async inference task
        while let Ok(msg) = self.chat_rx.try_recv() {
            if msg.role == "delta" {
                // Append to the last assistant bubble (streamed token)
                if let Some(last) = self.messages.last_mut() {
                    if last.role == "assistant" {
                        last.content.push_str(&msg.content);
                        continue;
                    }
                }
                // Fallback: no assistant bubble yet — create one
                self.messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: msg.content,
                    time: Local::now().format("%H:%M").to_string(),
                });
            } else {
                if msg.role == "assistant" && msg.content.is_empty() {
                    // Seed bubble for streaming — only push if last isn't already
                    // an empty assistant bubble
                    if self
                        .messages
                        .last()
                        .map(|m| m.role == "assistant" && m.content.is_empty())
                        .unwrap_or(false)
                    {
                        continue;
                    }
                }
                if msg.role == "assistant" && !msg.content.is_empty() {
                    // Final assistant message: replace the streamed bubble content
                    // with the final trimmed version, then mark generation done.
                    if let Some(last) = self.messages.last_mut() {
                        if last.role == "assistant" {
                            last.content = msg.content;
                            last.time = msg.time;
                            self.is_generating = false;
                            continue;
                        }
                    }
                }
                if msg.role == "assistant" {
                    self.is_generating = false;
                }
                self.messages.push(msg);
            }
        }

        // Poll guardian alerts
        while let Ok(alert) = self.guardian_rx.try_recv() {
            if alert.is_emergency {
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!(
                        "🚨 [EMERGENCY {:.0}% RAM {:.0}% CPU] {} {}",
                        alert.ram_percent,
                        alert.cpu_percent,
                        alert.hog_report.as_deref().unwrap_or(""),
                        alert.action_taken.as_deref().unwrap_or("")
                    ),
                    time: Local::now().format("%H:%M").to_string(),
                });
                // Hog report with ask-before-kill hint already inside GuardianAlert
                if let Some(h) = alert.hog_report.clone() {
                    self.messages.push(ChatMessage {
                        role: "system".into(),
                        content: h,
                        time: Local::now().format("%H:%M").to_string(),
                    });
                }
            } else if alert.is_critical {
                let mut c = format!(
                    "⚠️ [GUARDIAN CRITICAL]: RAM {:.1}% CPU {:.0}% Mode {}",
                    alert.ram_percent,
                    alert.cpu_percent,
                    alert.perf_mode.label()
                );
                if let Some(a) = alert.action_taken.as_deref() {
                    c.push_str(&format!(" — {}", a));
                }
                if let Some(h) = alert.hog_report.as_deref() {
                    c.push_str("\n");
                    c.push_str(h);
                }
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: c,
                    time: Local::now().format("%H:%M").to_string(),
                });
            } else if alert.perf_mode != PerfMode::Balanced {
                // Turbo/low transition notice
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!(
                        "ℹ️ Guardian → {} (RAM {:.0}% CPU {:.0}%)",
                        alert.perf_mode.label(),
                        alert.ram_percent,
                        alert.cpu_percent
                    ),
                    time: Local::now().format("%H:%M").to_string(),
                });
            }
            self.last_alert = Some(alert);
        }

        // Top panel: Logo dropdown menu + status/model switcher or server connect mode
        egui::TopBottomPanel::top("top_panel")
            .frame(
                egui::Frame::none()
                    .inner_margin(egui::Margin::symmetric(16.0, 10.0))
                    .fill(egui::Color32::from_rgb(13, 17, 23)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let mode_text = match self.app_mode {
                        AppMode::AiChat => "🌊 Ghostwater (AI Chat)",
                        AppMode::ServerConnect => "🌐 Connect to Servers",
                    };
                    egui::ComboBox::from_id_salt("logo_mode_dropdown")
                        .selected_text(mode_text)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.app_mode, AppMode::AiChat, "🌊 Ghostwater (AI Chat)");
                            ui.selectable_value(&mut self.app_mode, AppMode::ServerConnect, "🌐 Connect to Servers");
                        });

                    ui.separator();

                    if self.app_mode == AppMode::AiChat {
                        if let Some(gpu) = &self.gpu_status {
                            if let Some(name) = gpu.device_name.as_deref() {
                                let tint = if gpu.loaded_layers < gpu.target_layers {
                                    egui::Color32::YELLOW
                                } else {
                                    egui::Color32::LIGHT_GREEN
                                };
                                ui.colored_label(
                                    tint,
                                    format!("GPU {} · {}/{} layers", name, gpu.loaded_layers, gpu.target_layers),
                                );
                            } else if let Some(reason) = gpu.failure_reason.as_deref() {
                                ui.weak(format!("CPU fallback · {reason}"));
                            }
                        }

                        let current_name = config::MODEL_PROFILES[self.active_model_idx].name;
                        egui::ComboBox::from_id_salt("model_selector")
                            .selected_text(current_name)
                            .show_ui(ui, |ui| {
                                let frame_free_gb = self
                                    .hw_cache
                                    .lock()
                                    .map(|g| (g.ram_total_gb - g.ram_used_gb).max(0.0))
                                    .unwrap_or(4.0);
                                for (i, profile) in config::MODEL_PROFILES.iter().enumerate() {
                                    let ram_ok = frame_free_gb >= profile.ram_warn_gb;
                                    let label = if ram_ok {
                                        profile.name.to_string()
                                    } else {
                                        format!("⚠️ {}", profile.name)
                                    };
                                    let is_active = i == self.active_model_idx;
                                    if ui
                                        .selectable_label(is_active, &label)
                                        .on_hover_text(profile.description)
                                        .clicked()
                                        && !is_active
                                    {
                                        self.input_text = format!("/model {}", i);
                                        ctx.request_repaint();
                                    }
                                }
                            });
                    } else {
                        ui.weak("Configure remote servers (Kaggle / Colab ngrok / Omniroute)");
                    }
                });
            });

        // Ask before suggesting that the user release VRAM; never stop another app.
        let mut retry_gpu_allocation = false;
        if self.gpu_retry_prompt {
            let status = self.gpu_status.clone();
            egui::Window::new("GPU memory is limited")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    if let Some(status) = status {
                        ui.label(format!(
                            "{}: using {} of {} requested GPU layers.",
                            status.device_name.as_deref().unwrap_or("NVIDIA GPU"),
                            status.loaded_layers,
                            status.target_layers
                        ));
                        ui.label(format!("About {} MiB more VRAM is needed for the remaining target.", status.additional_mb_needed));
                    }
                    ui.label("Close GPU applications you recognize, then retry. Momentum will not close applications or kill processes.");
                    ui.horizontal(|ui| {
                        if ui.button("I freed VRAM — retry").clicked() { retry_gpu_allocation = true; }
                        if ui.button("Keep current split").clicked() { self.gpu_retry_prompt = false; }
                    });
                });
        }
        if retry_gpu_allocation {
            let model_idx = self.active_model_idx;
            self.switch_model(model_idx);
        }

        // Bottom panel: Text input
        egui::TopBottomPanel::bottom("bottom_panel")
            .frame(
                egui::Frame::none()
                    .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                    .fill(egui::Color32::from_rgb(18, 18, 24)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let response = ui.add_sized(
                        [ui.available_width() - 120.0, 36.0],
                        egui::TextEdit::singleline(&mut self.input_text).hint_text(
                            if self.is_generating {
                                "Ghostwater is thinking..."
                            } else {
                                "Talk to Ghostwater... (Enter to send)"
                            },
                        ),
                    );
                    // NMOS Zero-Lag: Prefetch weights and prewarm engine while user types
                    if (response.changed() || (!self.input_text.is_empty() && response.has_focus()))
                        && self.last_prefetch.elapsed() > std::time::Duration::from_secs(20)
                    {
                        self.last_prefetch = std::time::Instant::now();
                        let profile = &config::MODEL_PROFILES[self.active_model_idx];
                        if let Some(blob_path) = config::resolve_blob(profile) {
                            crate::llm::async_prefetch_model(&blob_path);
                            crate::llm::async_prewarm_server();
                        }
                    }
                    if !self.is_generating {
                        if response.lost_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                            let msg = self.input_text.clone();
                            self.send_message(msg);
                        }
                        if ui.button("Send").clicked() {
                            let msg = self.input_text.clone();
                            self.send_message(msg);
                        }
                        let speak_btn = ui.button("🔊");
                        if speak_btn.clicked() {
                            if let Some(last) =
                                self.messages.iter().rev().find(|m| m.role == "assistant")
                            {
                                let text = last.content.chars().take(500).collect::<String>();
                                std::thread::spawn(move || {
                                    let _ = std::process::Command::new("spd-say")
                                        .args(["--wait", &text])
                                        .status();
                                });
                            }
                        }
                        speak_btn.on_hover_text("Speak last reply");

                        // 🎤 voice input button
                        let mic_label = if self.is_recording { "🔴" } else { "🎤" };
                        let mic_btn = ui.button(mic_label);
                        let mic_down = mic_btn.is_pointer_button_down_on();
                        let mic_clicked = mic_btn.clicked();

                        if mic_down && self.mic_pressed_at.is_none() {
                            // pointer held — start push-to-talk
                            self.mic_pressed_at = Some(std::time::Instant::now());
                            if !self.is_recording {
                                self.start_recording();
                            }
                        }

                        if !mic_down {
                            if let Some(pressed_at) = self.mic_pressed_at.take() {
                                if pressed_at.elapsed().as_millis() >= 400 {
                                    // was a hold — release stops recording
                                    self.stop_recording_and_transcribe();
                                }
                                // short press handled by clicked() below
                            }
                        }

                        if mic_clicked {
                            // only treat as toggle if it wasn't a hold
                            if self.mic_pressed_at.is_none() {
                                if self.is_recording {
                                    self.stop_recording_and_transcribe();
                                } else {
                                    self.start_recording();
                                }
                            }
                        }

                        if self.is_recording {
                            mic_btn.on_hover_text("Recording… click or release to stop");
                            ctx.request_repaint();
                        } else {
                            mic_btn.on_hover_text("Hold or click to record voice");
                        }

                        // poll for transcription result in /tmp/momentum_voice_result.txt
                        if !self.is_recording {
                            let result_path =
                                std::path::Path::new("/tmp/momentum_voice_result.txt");
                            if result_path.exists() {
                                if let Ok(text) = std::fs::read_to_string(result_path) {
                                    let trimmed = text.trim().to_string();
                                    if !trimmed.is_empty() {
                                        self.input_text = trimmed;
                                    }
                                    let _ = std::fs::remove_file(result_path);
                                }
                            }
                        }
                    } else {
                        ui.add_enabled(false, egui::Button::new("..."));
                    }
                });
            });

        // Central panel: Chat history or Server Connection panel
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.app_mode {
                AppMode::AiChat => {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for msg in &self.messages {
                                let (fill_color, border_color) = match msg.role.as_str() {
                                    "user" => (egui::Color32::from_rgb(22, 27, 34), egui::Color32::from_rgb(48, 54, 61)),
                                    "assistant" => (egui::Color32::from_rgb(22, 27, 34), egui::Color32::from_rgb(88, 166, 255)),
                                    _ => (egui::Color32::from_rgb(22, 27, 34), egui::Color32::from_rgb(110, 118, 129)),
                                };
                                egui::Frame::none()
                                    .fill(fill_color)
                                    .stroke(egui::Stroke::new(1.0_f32, border_color))
                                    .rounding(6.0)
                                    .inner_margin(12.0)
                                    .show(ui, |ui| {
                                        let (badge, color) = match msg.role.as_str() {
                                            "user" => ("👤 You", egui::Color32::from_rgb(88, 166, 255)),
                                            "assistant" => ("🌊 Ghostwater", egui::Color32::from_rgb(35, 134, 54)),
                                            "system" => ("⚙️ System", egui::Color32::from_rgb(139, 148, 158)),
                                            _ => ("🛠️ Tool", egui::Color32::from_rgb(210, 153, 34)),
                                        };
                                        ui.horizontal(|ui| {
                                            ui.colored_label(color, badge);
                                            if !msg.time.is_empty() { ui.weak(&msg.time); }
                                        });
                                        ui.add_space(4.0);
                                        ui.label(&msg.content);
                                    });
                                ui.add_space(8.0);
                            }

                            if self.is_generating {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(22, 27, 34))
                                    .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(88, 166, 255)))
                                    .rounding(6.0)
                                    .inner_margin(12.0)
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.weak("Ghostwater is generating...");
                                        });
                                    });
                            }
                        });
                }
                AppMode::ServerConnect => {
                    ui.add_space(20.0);
                    ui.heading("🌐 Connect Remote GPU Servers");
                    ui.weak("Connect Kaggle, Colab (ngrok), Omniroute, or OpenAI-compatible servers.");
                    ui.add_space(16.0);
                    ui.label("Endpoint URL:");
                    ui.add(egui::TextEdit::singleline(&mut self.remote_url).hint_text("https://xxxx.ngrok-free.app/v1"));
                    ui.add_space(10.0);
                    ui.label("API Key / Token:");
                    ui.add(egui::TextEdit::singleline(&mut self.remote_api_key).password(true));
                    ui.add_space(16.0);
                    if ui.button("Test Connection").clicked() {
                        let url = self.remote_url.clone();
                        let tx = self.chat_tx.clone();
                        self.tokio_runtime.spawn(async move {
                            let client = ureq::Agent::new_with_defaults();
                            match client.get(&format!("{}/models", url.trim_end_matches('/'))).call() {
                                Ok(_) => { let _ = tx.send(ChatMessage { role: "system".into(), content: format!("✅ Connected to {}", url), time: Local::now().format("%H:%M").to_string() }); }
                                Err(e) => { let _ = tx.send(ChatMessage { role: "system".into(), content: format!("❌ Connection failed: {}", e), time: Local::now().format("%H:%M").to_string() }); }
                            }
                        });
                    }
                }
            }
        });

        // Auto-dispatch model switch commands written by the ComboBox closure.
        // The closure can't call switch_model() directly due to borrow rules,
        // so it writes "/model N" to input_text and we execute it here, after
        // all panels have been rendered and borrows released.
        if self.input_text.starts_with("/model ") && !self.is_generating {
            let cmd = std::mem::take(&mut self.input_text);
            self.send_message(cmd);
        }

        // Request repaint so UI stays responsive to background messages
        let interval = if self.is_generating { 100 } else { 500 };
        ctx.request_repaint_after(std::time::Duration::from_millis(interval));
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--bench") {
        let mut profile_arg = "0";
        let mut prompt_file = "prompts/example1.txt";
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--profile" && i + 1 < args.len() {
                profile_arg = &args[i + 1];
            }
            if args[i] == "--prompt-file" && i + 1 < args.len() {
                prompt_file = &args[i + 1];
            }
            i += 1;
        }

        let _ = telemetry::init();
        let _runtime = if config::llama_cli_path().exists() {
            None
        } else if let Ok(idx) = profile_arg.parse::<usize>() {
            if idx < config::MODEL_PROFILES.len() {
                let profile = &config::MODEL_PROFILES[idx];
                Some(InferenceRuntime::start_with_profile(profile)?)
            } else {
                Some(InferenceRuntime::start()?)
            }
        } else {
            if let Some(pos) = config::MODEL_PROFILES.iter().position(|pf| pf.name.to_lowercase().contains(&profile_arg.to_lowercase())) {
                let profile = &config::MODEL_PROFILES[pos];
                Some(InferenceRuntime::start_with_profile(profile)?)
            } else {
                Some(InferenceRuntime::start()?)
            }
        };

        let prompt = std::fs::read_to_string(prompt_file).unwrap_or_else(|_| "Hello momentum".to_string());
        let start = std::time::Instant::now();
        let mut token_count = 0usize;
        let bench_profile = if let Ok(idx) = profile_arg.parse::<usize>() {
            if idx < config::MODEL_PROFILES.len() { &config::MODEL_PROFILES[idx] } else { &config::MODEL_PROFILES[0] }
        } else {
            if let Some(pos) = config::MODEL_PROFILES.iter().position(|pf| pf.name.to_lowercase().contains(&profile_arg.to_lowercase())) {
                &config::MODEL_PROFILES[pos]
            } else {
                &config::MODEL_PROFILES[0]
            }
        };
        let formatted_prompt = config::format_prompt(bench_profile, "You are a helpful assistant.", "", &prompt);
        let out = llm::complete_stream_with_profile(bench_profile, &formatted_prompt, 256, |piece| {
            token_count += piece.chars().count();
        })?;
        let elapsed = start.elapsed();
        let summary = serde_json::json!({
            "elapsed_ms": elapsed.as_millis(),
            "tokens": token_count,
            "output_len": out.len(),
            "profile": profile_arg,
        });
        std::fs::write("/tmp/bench_out.json", summary.to_string())?;
        println!("{}", summary.to_string());
        return Ok(());
    }

    std::env::set_var("RAYON_NUM_THREADS", "4");
    let _ = telemetry::init();
    if let Ok(log) = action_log::ActionLog::open() {
        let _ = log.remove_done();
    }
    // Keep async orchestration lean; the native inference engine is serialized
    // through its mutex and runs the CPU/GPU kernels synchronously.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?;

    let _rt_guard = rt.enter();

    let (alert_tx, alert_rx) = unbounded();
    let hw_cache = new_hw_cache();
    start_guardian(alert_tx, hw_cache.clone());

    let memory = Arc::new(Mutex::new(Memory::open()?));

    // Activity tracker — shared between send_message and the reflection loop
    let activity = new_activity_tracker();

    // Pass activity clone to reflection so it backs off while user is active
    start_reflection_loop(rt.handle(), memory.clone(), activity.clone());

    // Load the custom in-process Rust engine. CUDA is optional; unsupported
    // devices fall back to CPU inference, with profile-aware context limits.
    let (inference_runtime, inference_error) = match InferenceRuntime::start() {
        Ok(runtime) => (Some(runtime), None),
        Err(error) => (None, Some(error.to_string())),
    };
    if inference_runtime.is_some() {
        if let Ok(mem) = memory.try_lock() {
            match mem.backfill_missing_embeddings(24) {
                Ok(count) if count > 0 => println!("Rebuilt {count} legacy memory vectors"),
                Ok(_) => {}
                Err(error) => eprintln!("Memory vector backfill skipped: {error}"),
            }
        }
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 560.0])
            .with_min_inner_size([480.0, 360.0])
            .with_title("Ghostwater Momentum"),
        ..Default::default()
    };

    let rt_handle = rt.handle().clone();
    let mem_for_exit = memory.clone();
    let result = eframe::run_native(
        "Ghostwater Momentum",
        options,
        Box::new(move |cc| {
            Ok(Box::new(MomentumApp::new(
                cc,
                memory,
                alert_rx,
                rt_handle,
                inference_runtime,
                inference_error,
                activity,
                hw_cache,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("UI error: {e}"));

    // Write clean-exit sentinel so next boot knows we didn't crash.
    // Runs after eframe returns (window closed normally).
    if let Ok(mem) = mem_for_exit.try_lock() {
        let _ = mem.mark_clean_exit();
    }

    result
}
