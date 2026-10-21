// config.rs - All constants and paths in one place
// Tuned for i7-8550U (15W TDP, 8GB RAM) + GeForce MX150 (2GB VRAM, CUDA 6.1)

// Inference uses tuned local llama-cli with AVX2 OpenMP multithreading for generation.
// infer-rs remains available for mean token embeddings and fallback.

// ── Model profiles ────────────────────────────────────────────────────────────
// Each entry: (display_name, blob_sha_fragment, max_gpu_layers, ctx_size, ram_warn_gb, description)
// max_gpu_layers is a target, capped by live CUDA free memory at load time.
// Remaining layers execute on the CPU; Momentum never kills other processes.
//
// ram_warn_gb: threshold at which the UI shows an orange resource warning banner.
//
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProfile {
    pub name: &'static str,        // shown in UI dropdown
    pub blob_sha: &'static str,    // unique prefix of the sha256 blob filename
    pub gpu_layers: u32,           // upper bound; adaptive allocator may use fewer layers
    pub ctx_size: u32,             // context window tokens
    pub max_tokens: u32,           // max tokens per response
    pub ram_warn_gb: f32,          // show RAM warning if available RAM < this
    pub description: &'static str, // shown in tooltip / warning banner
}

/// All models available to Momentum. First entry is the startup default.
/// To add a new model: pull it with ollama, find its blob hash with
///   `ls -lh /media/.../ollama_models/blobs/sha256-*`
/// then add a row here.
pub static MODEL_PROFILES: &[ModelProfile] = &[
    ModelProfile {
        name:        "Qwen 2.5 Coder 3B (Tools)",
        blob_sha:    "sha256-4a1881",
        gpu_layers:  99,
        ctx_size:    2048,
        max_tokens:  512,
        ram_warn_gb: 2.0,
        description: "Structured output & tool tags. Uses as many CUDA layers as current free VRAM allows.",
    },
    ModelProfile {
        name:        "Hermes 3 8B (Primary)",
        blob_sha:    "sha256-c8985d",
        gpu_layers:  14,
        ctx_size:    2048,
        max_tokens:  256,
        ram_warn_gb: 3.5,
        description: "Best reasoning. Targets 14 CUDA layers; remaining layers stay on CPU if VRAM is scarce.",
    },
    ModelProfile {
        name:        "Qwen 2.5 3B (Fast)",
        blob_sha:    "sha256-5ee4f0",
        gpu_layers:  99,
        ctx_size:    2048,
        max_tokens:  512,
        ram_warn_gb: 2.0,
        description: "Quick fallback. Uses as many CUDA layers as current free VRAM allows.",
    },
    ModelProfile {
        name:        "TinyLlama 1.1B (Emergency)",
        blob_sha:    "sha256-2af3b8",
        gpu_layers:  99,
        ctx_size:    2048,
        max_tokens:  256,
        ram_warn_gb: 1.0,
        description: "⚠️ High hallucination rate. Use only when RAM is critical; CUDA placement adapts to free VRAM.",
    },
    ModelProfile {
        name:        "Mistral 7B Instruct (Smart)",
        blob_sha:    "sha256-faf975",
        gpu_layers:  14,
        ctx_size:    4096,
        max_tokens:  512,
        ram_warn_gb: 3.5,
        description: "Standard 7B instruct model. Targets 14 CUDA layers, with CPU fallback for other layers.",
    },

];

use std::path::PathBuf;

// Configuration overrides (all optional): MOMENTUM_DATA_DIR, MOMENTUM_DB_PATH,
// MOMENTUM_AI_DRIVE, MOMENTUM_AI_DRIVE_FALLBACK, MOMENTUM_MODELS_DIR,
// MOMENTUM_ARCHIVE_PATH, MOMENTUM_WHISPER_CLI, MOMENTUM_WHISPER_MODEL,
// MOMENTUM_BLENDER_BIN, MOMENTUM_OUTPUT_DIR, MOMENTUM_PIPER_BIN,
// MOMENTUM_PIPER_MODEL, MOMENTUM_EZSYNTH_ROOT and MOMENTUM_PYTHON.
pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn project_data_dir() -> PathBuf {
    std::env::var_os("MOMENTUM_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join("AI_project"))
}

fn login_name() -> String {
    std::env::var("USER")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            home_dir()
                .file_name()
                .map(|value| value.to_string_lossy().into_owned())
        })
        .unwrap_or_default()
}

pub fn ai_drive_root() -> PathBuf {
    if let Some(path) = std::env::var_os("MOMENTUM_AI_DRIVE") {
        return PathBuf::from(path);
    }
    PathBuf::from("/media").join(login_name()).join("AI_Drive")
}

pub fn ai_projects_root() -> PathBuf {
    ai_drive_root().join("AI_projects")
}

pub fn model_blob_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(configured) = std::env::var_os("MOMENTUM_MODELS_DIR") {
        let configured = PathBuf::from(configured);
        if configured.file_name().is_some_and(|name| name == "blobs") {
            roots.push(configured);
        } else {
            roots.push(configured.join("blobs"));
            roots.push(configured.join("ollama_models").join("blobs"));
        }
    }
    roots.push(ai_drive_root().join("ollama_models").join("blobs"));
    if let Some(fallback) = std::env::var_os("MOMENTUM_AI_DRIVE_FALLBACK") {
        roots.push(PathBuf::from(fallback).join("ollama_models").join("blobs"));
    } else {
        let user = std::env::var("USER").unwrap_or_default();
        roots.push(
            PathBuf::from("/media")
                .join(user)
                .join("AI_Drive1")
                .join("ollama_models")
                .join("blobs"),
        );
    }
    roots
}

/// Resolve a GGUF model blob from configured/local SSD Ollama blob directories.
pub fn resolve_blob(profile: &ModelProfile) -> Option<String> {
    for drive in model_blob_dirs() {
        let Ok(entries) = std::fs::read_dir(drive) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with(profile.blob_sha) {
                return Some(entry.path().to_string_lossy().into_owned());
            }
        }
    }
    None
}

pub fn database_path() -> PathBuf {
    std::env::var_os("MOMENTUM_DB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_data_dir().join("momentum.db"))
}

pub fn memory_archive_paths() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("MOMENTUM_ARCHIVE_PATH") {
        return vec![PathBuf::from(path)];
    }
    let mut paths = vec![ai_drive_root().join("momentum_archive.db")];
    if let Some(fallback) = std::env::var_os("MOMENTUM_AI_DRIVE_FALLBACK") {
        paths.push(PathBuf::from(fallback).join("momentum_archive.db"));
    } else {
        paths.push(
            PathBuf::from("/media")
                .join(login_name())
                .join("AI_Drive1")
                .join("momentum_archive.db"),
        );
    }
    paths
}

pub fn whisper_cli_path() -> PathBuf {
    std::env::var_os("MOMENTUM_WHISPER_CLI")
        .map(PathBuf::from)
        .unwrap_or_else(|| ai_projects_root().join("whisper.cpp/build/bin/whisper-cli"))
}

pub fn llama_bin_dir() -> PathBuf {
    std::env::var_os("MOMENTUM_LLAMA_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/mehtculos-xfemag/AI_project/llama-bin"))
}

pub fn llama_cli_path() -> PathBuf {
    llama_bin_dir().join("llama-cli")
}

pub fn llama_threads() -> u32 {
    std::env::var("MOMENTUM_LLAMA_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(4)
}

pub fn llama_ngl() -> u32 {
    0
}

pub fn llama_ctx(profile: &ModelProfile) -> u32 {
    let cap = if profile.ram_warn_gb >= 3.5 { 1024 } else { 2048 };
    profile.ctx_size.min(cap).max(1)
}

pub fn llama_batch() -> u32 {
    256
}

pub fn whisper_model_path() -> PathBuf {
    std::env::var_os("MOMENTUM_WHISPER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| ai_projects_root().join("whisper.cpp/models/ggml-base.en.bin"))
}

pub fn blender_path() -> PathBuf {
    std::env::var_os("MOMENTUM_BLENDER_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| ai_projects_root().join("apps/blender/blender-5.2.2-linux-x64/blender"))
}

pub fn creative_output_dir() -> PathBuf {
    std::env::var_os("MOMENTUM_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| ai_projects_root().join("blender_output"))
}

pub fn piper_bin_path() -> PathBuf {
    std::env::var_os("MOMENTUM_PIPER_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| ai_projects_root().join("apps/piper/piper"))
}

pub fn piper_model_path() -> PathBuf {
    std::env::var_os("MOMENTUM_PIPER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| ai_projects_root().join("apps/piper/voices/en_US-lessac-medium.onnx"))
}

pub fn ezsynth_root() -> PathBuf {
    std::env::var_os("MOMENTUM_EZSYNTH_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| ai_projects_root().join("Ezsynth"))
}

pub fn python_path() -> PathBuf {
    std::env::var_os("MOMENTUM_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"))
}
pub const N_THREADS: u32 = 3; // 4 physical cores on i7-8550U; leaves HTs for OS/GUI

// Guardian thresholds — adaptive two-tier
// Lightweight fast poll (5s) watches for 95% emergency, heavy adaptive poll (300s) tunes mode.
pub const RAM_WARN_PERCENT: f32 = 75.0;
pub const RAM_CRITICAL_PERCENT: f32 = 85.0;
pub const RAM_EMERGENCY_PERCENT: f32 = 95.0;
pub const CPU_EMERGENCY_PERCENT: f32 = 95.0;
pub const GUARDIAN_FAST_SECS: u64 = 5; // instant 95% detection (raw /proc read, ~0 CPU)
pub const GUARDIAN_ADAPTIVE_SECS: u64 = 300; // full 5-min adaptive scan + top-process analysis
pub const GUARDIAN_TOP_N: usize = 5; // how many hogs to report when emergency fires

// CPU temperature thresholds (°C) — i7-8550U Tjmax is 100°C per coretemp driver
// Warn at 80°C: sustained load; critical at 95°C: within 5° of throttle point
pub const CPU_TEMP_WARN_C: f32 = 80.0;
pub const CPU_TEMP_CRITICAL_C: f32 = 95.0;

// Memory / reflection
pub const MAX_HISTORY_TURNS: usize = 4; // fewer turns = smaller prompt = less RAM
// Reflection only runs when the machine is genuinely idle (user away 5 min)
pub const REFLECTION_IDLE_SECS: u64 = 300;
pub const PRUNE_INTERVAL_SECS: u64 = 86400;

// HW telemetry + gradient memory
// Keep raw hw_samples for 24h then prune; ~2800 rows * ~40B ≈ 110 KB max.
pub const HW_SAMPLE_KEEP_HOURS: i64 = 24;
// How many consecutive smooth samples before we label the period "safe".
pub const SMOOTH_STREAK: u32 = 3;
// SGD learning rate for the 5-float HW risk vector.  Small enough that one
// crash cannot dominate; large enough to learn in ~20 events.
pub const HW_LR: f32 = 0.05;
// Weight clip — prevents a single run of bad luck from saturating the predictor.
pub const HW_GRAD_CLIP: f32 = 3.0;
// Daily multiplicative decay on node strength (≈ 1-week half-life).
// Applied once per 24h prune cycle, idle-only, so 0 CPU during active use.
pub const STRENGTH_DECAY: f32 = 0.99;

// Episodic sessions decay 5× faster than semantic knowledge nodes.
// After ~1.5 days of no reinforcement a raw session drops below prune threshold.
pub const SESSION_DECAY_MULTIPLIER: f32 = 5.0;

// Hard cap on total live nodes. Once exceeded, the compaction pass during
// the 24h cycle evicts the weakest disconnected leaf nodes first.
// ~500 nodes * ~200 B avg content ≈ 100 KB max in-DB footprint.
pub const MAX_LIVE_NODES: usize = 500;

// Raw session ring-buffer cap.
// Raw text is kept only until reflection audits it, then deleted.
// Hard cap ensures storage stays bounded even if reflection never runs.
// 50 rows × ~6 KB avg ≈ 300 KB max. After reflection: 0 KB.
pub const RAW_SESSION_KEEP: usize = 50;

// 1-hop graph context: max neighbour nodes pulled per query seed.
// 3 seeds * 4 neighbours = 12 context snippets ≤ ~120 tokens.
pub const GRAPH_HOP_NEIGHBOURS: usize = 4;

// Minimum edge strength to follow during 1-hop expansion.
// Edges below this are considered too weak to be useful context.
pub const GRAPH_MIN_EDGE_STRENGTH: f32 = 0.35;
// Memory SSD pocket — overflow/archive + lean context window
pub const MEMORY_ARCHIVE_MAX_MB: u64 = 512; // cap archive DB size on SSD
pub const CONTEXT_MAX_CHARS: usize = 1600; // fits inside 2048 ctx with PERSONA+history+response
                                            // Ghostwater persona — split so tool docs are only injected when the query
                                            // looks tool-like, saving ~200 tokens on plain conversation turns.
                                            //
                                            // PERSONA_CORE  : always sent (~180 chars / ~45 tokens)
                                            // PERSONA_TOOLS : injected only when needs_tools() returns true (~780 chars / ~195 tokens)
                                            // PERSONA       : full combined string kept for the rare cases that need it (tests, etc.)
pub const PERSONA_CORE: &str = "You are Ghostwater, a direct and concise AI companion running locally on limited hardware.\nStay brief. No filler. Match the user's tone.\nA tool call is ONE line and must include every argument, especially file content. No text after the closing ].";

pub const PERSONA_TOOLS: &str = r#"TOOL SYNTAX — output ONE tag per line, nothing else on that line:

AUTONOMOUS DESKTOP (fully offline, X11):
[TOOL: shell "bash command here"] — run anything, full user permissions. For long tasks: "cmd &"
[TOOL: screen_read] — screenshot+OCR in one call → returns screen text + word positions
[TOOL: screen_read "x,y,w,h"] — capture only a region (faster)
[TOOL: find_and_click "Save"] — finds text on screen and clicks it. Preferred over pixel coords.
[TOOL: find_and_click "OK" "1"] — button: 1=left(default) 2=middle 3=right
[TOOL: window_list] — list all open windows by title
[TOOL: window_focus "Firefox"] — bring a window to front by title fragment
[TOOL: clipboard "read"] — read what's in the clipboard
[TOOL: clipboard "write" "text to copy"] — write to clipboard
[TOOL: notify "Task done" "render finished" "normal"] — desktop notification
[TOOL: desktop_action "key ctrl+c"] — raw xdotool: keys, type, mouse. Use find_and_click instead when possible.
[TOOL: desktop_action "type hello"] | [TOOL: desktop_action "mousemove 500 300 click 1"]

FILES:
[TOOL: read_file "/path"] | [TOOL: write_file "/path" "the COMPLETE text to save — a path alone writes nothing"] | [TOOL: append_to_file "/path" "content"]
[TOOL: list_directory "/path"] | [TOOL: search_in_file "/path" "pattern"] | [TOOL: find_files "/dir" "*.rs"]
To save a conversation, put the transcript inside the second quotes. Never call write_file with only a path.

WEB & LEARNING (user-initiated only, no background calls):
[TOOL: search_web "query"] | [TOOL: open_url "https://..."]
[TOOL: web_learn "topic"] — search + store as memory node in one step
[TOOL: download_sfx "https://..." "/output.wav"] — download audio via yt-dlp

MEDIA & CREATIVE:
[TOOL: blender_script "import bpy; ..."] — run Python in Blender headless
[TOOL: blender_render "/scene.blend" "/out/dir" "1" "250"] — render frame sequence
[TOOL: open_blender] | [TOOL: open_blender "/file.blend"] — launch Blender GUI
[TOOL: ffmpeg_compose "frames_to_video" "/frames/frame_%04d.png" "24" "/out.mp4" ""]
[TOOL: ffmpeg_compose "add_audio" "/video.mp4" "/audio.wav" "/final.mp4" ""]
[TOOL: ffmpeg_compose "trim" "/in.mp4" "5.0" "30.0" "/out.mp4"]
[TOOL: ffmpeg_compose "concat" "/list.txt" "/out.mp4" "" ""]
[TOOL: ffmpeg_compose "extract_audio" "/video.mp4" "/audio.wav" "" ""]
[TOOL: piper_tts "text to speak" "/output.wav"] — offline TTS, no network
[TOOL: ebsynth_run "/keyframe.png" "/frames_dir" "/output_dir" "1"] — style-transfer frames
[TOOL: transcribe "/audio.wav"] — local whisper.cpp STT
[TOOL: screenshot] | [TOOL: ocr] | [TOOL: ocr "/img.png"]
[TOOL: install_piper] | [TOOL: install_ebsynth] — one-time setup, downloads to AI Drive
[TOOL: kill_process "1234"]

MEMORY:
[TOOL: memory_search "query"] | [TOOL: memory_add "label" "type" "content"]
[TOOL: memory_update "id" "" "new_content"] | [TOOL: memory_hide "id"] | [TOOL: memory_get "query"]

AGENT LOOP PATTERN (read screen → think → act → verify):
  screen_read → decide what to do → find_and_click / shell / desktop_action → screen_read to verify"#;

/// Returns true when the query likely needs a tool or memory op.
/// Pure string scan — zero allocations on the hot path.
pub fn needs_tools(query: &str) -> bool {
    let q = query.as_bytes();
    // Check for common tool-trigger words without allocating a lowercase copy.
    // Covers: file ops, web, commands, memory, code, run, open, search, find, list, write, read
    let triggers: &[&[u8]] = &[
        b"file",
        b"web",
        b"search",
        b"find",
        b"open",
        b"run",
        b"list",
        b"write",
        b"read",
        b"create",
        b"delete",
        b"append",
        b"browse",
        b"memory",
        b"remember",
        b"forget",
        b"recall",
        b"blender",
        b"command",
        b"execute",
        b"shell",
        b"script",
        b"kill",
        b"transcri",
        b"whisper",
        b"audio",
        b".wav",
        b".mp3",
        b".flac",
        b".ogg",
        b"speak",
        b"say this",
        b"read aloud",
        b"tts",
        b"screenshot",
        b"ocr",
        b"click",
        b"mousemove",
        b"xdotool",
        b"desktop",
        b"http",
        b"www.",
        b".com",
        b".org",
        b".sh",
        b".py",
        b".rs",
        b"/home",
        b"/media",
        b"/tmp",
        b"~/",
        // creative pipeline
        b"render",
        b"sculpt",
        b"rig",
        b"animat",
        b"blend",
        b"ebsynth",
        b"piper",
        b"voice",
        b"narrat",
        b"ffmpeg",
        b"movie",
        b"video",
        b"frame",
        b"composit",
        b"sfx",
        b"sound effect",
        b"download audio",
        b"web_learn",
        b"learn about",
        b"look up",
        b"install_piper",
        b"install_ebsynth",
        // autonomous agent
        b"shell",
        b"click",
        b"screen",
        b"window",
        b"clipboard",
        b"notify",
        b"open ",
        b"close ",
        b"type ",
        b"press ",
        b"focus",
    ];
    // Case-insensitive: lowercase each byte only when comparing
    triggers.iter().any(|t| {
        q.windows(t.len()).any(|w| {
            w.iter()
                .zip(t.iter())
                .all(|(a, b)| a.to_ascii_lowercase() == *b)
        })
    })
}

pub fn format_prompt(profile: &ModelProfile, system: &str, history: &str, user: &str) -> String {
    if profile.name.to_lowercase().contains("mistral") {
        format!("[INST] System: {}\n\nHistory:\n{}\n\nUser: {} [/INST]", system, history, user)
    } else {
        format!("<|im_start|>system\n{}\n<|im_end|>\n{}\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n", system, history, user)
    }
}
