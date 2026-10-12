// permissions.rs — Budhur Permissions / Approval Gate
//
// Black Paper §4, §14, §16:
//   "Deterministic safeguards come before AI guesses."
//   "Destructive operations require explicit confirmation."
//   "Momentum never silently deletes valuable user data."
//
// Three layers:
//   1. ToolRisk   — classify every tool call before execution
//   2. Policy     — allow / confirm / deny based on active project workspace
//   3. ActionLog  — immutable SQLite log of every action considered or executed

use std::path::{Path, PathBuf};

// ── Risk classification ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolRisk {
    /// Read-only, sandboxed. Execute immediately.
    Safe,
    /// Write inside project workspace. Show intent; execute unless cancelled.
    Moderate,
    /// Write outside workspace, destructive, or network upload.
    /// Require explicit user confirmation every time.
    High,
    /// Absolute veto — system paths, protected files, etc.
    Denied,
}

impl ToolRisk {
    pub fn label(&self) -> &'static str {
        match self {
            ToolRisk::Safe => "SAFE",
            ToolRisk::Moderate => "MODERATE",
            ToolRisk::High => "HIGH",
            ToolRisk::Denied => "DENIED",
        }
    }
    pub fn emoji(&self) -> &'static str {
        match self {
            ToolRisk::Safe => "✅",
            ToolRisk::Moderate => "⚠️",
            ToolRisk::High => "🔒",
            ToolRisk::Denied => "🚫",
        }
    }
}

// ── Verdict ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Verdict {
    pub risk: ToolRisk,
    pub reason: String,
    pub needs_confirm: bool,
}

impl Verdict {
    pub fn allow(risk: ToolRisk, reason: impl Into<String>) -> Self {
        let needs_confirm = risk >= ToolRisk::Moderate;
        Self {
            risk,
            reason: reason.into(),
            needs_confirm,
        }
    }
    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            risk: ToolRisk::Denied,
            reason: reason.into(),
            needs_confirm: false,
        }
    }
}

// ── Protected roots (absolute deny-list) ─────────────────────────────────────

static PROTECTED_ROOTS: &[&str] = &[
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/crontab",
    "/usr/bin",
    "/usr/lib",
    "/bin",
    "/lib",
    "/sbin",
];

fn is_protected(path: &Path) -> bool {
    let abs = path.canonicalize().unwrap_or_else(|_| {
        path.parent()
            .and_then(|p| p.canonicalize().ok())
            .map(|p| p.join(path.file_name().unwrap_or_default()))
            .unwrap_or_else(|| path.to_path_buf())
    });
    PROTECTED_ROOTS.iter().any(|root| abs.starts_with(root))
}

fn inside_workspace(path: &Path, workspace: &Path) -> bool {
    let abs_path = path.canonicalize().unwrap_or_else(|_| {
        path.parent()
            .and_then(|p| p.canonicalize().ok())
            .map(|p| p.join(path.file_name().unwrap_or_default()))
            .unwrap_or_else(|| path.to_path_buf())
    });
    let abs_ws = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    abs_path.starts_with(&abs_ws)
}

// ── Policy engine ─────────────────────────────────────────────────────────────

pub struct Policy {
    pub workspace: PathBuf,
    pub project_name: String,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            workspace: crate::config::project_data_dir().join("sandbox"),
            project_name: "sandbox".into(),
        }
    }
}

impl Policy {
    pub fn new(workspace: impl Into<PathBuf>, project_name: impl Into<String>) -> Self {
        Self {
            workspace: workspace.into(),
            project_name: project_name.into(),
        }
    }

    pub fn check_read(&self, path: &str) -> Verdict {
        let p = Path::new(path);
        if is_protected(p) {
            return Verdict::deny(format!("Read blocked: {path} is a protected system path"));
        }
        Verdict::allow(ToolRisk::Safe, format!("Read: {path}"))
    }

    pub fn check_write(&self, path: &str) -> Verdict {
        let p = Path::new(path);
        if is_protected(p) {
            return Verdict::deny(format!("Write blocked: {path} is a protected system path"));
        }
        if inside_workspace(p, &self.workspace) {
            Verdict::allow(
                ToolRisk::Moderate,
                format!("Write inside project '{}': {path}", self.project_name),
            )
        } else {
            Verdict::allow(
                ToolRisk::High,
                format!(
                    "Write OUTSIDE workspace ({}).\nTarget: {path}\nRequires confirmation.",
                    self.workspace.display()
                ),
            )
        }
    }

    pub fn check_list(&self, path: &str) -> Verdict {
        if is_protected(Path::new(path)) {
            return Verdict::deny(format!("List blocked: {path} is a protected path"));
        }
        Verdict::allow(ToolRisk::Safe, format!("List directory: {path}"))
    }

    pub fn check_web(&self, query: &str) -> Verdict {
        Verdict::allow(ToolRisk::Safe, format!("Web search: {query}"))
    }

    /// Unified entry point used by dispatch().
    pub fn check(&self, tool_name: &str, path_arg: Option<&str>) -> Verdict {
        match tool_name {
            "search_web" => self.check_web(path_arg.unwrap_or("")),
            "read_file" | "search_in_file" => self.check_read(path_arg.unwrap_or("")),
            "write_file" | "append_to_file" => self.check_write(path_arg.unwrap_or("")),
            "list_directory" => self.check_list(path_arg.unwrap_or("")),
            "transcribe" | "whisper" => self.check_read(path_arg.unwrap_or("")),
            "speak" | "say" => Verdict::allow(ToolRisk::Safe, String::from("Speak text locally")),
            "screenshot" => Verdict::allow(ToolRisk::Safe, String::from("Screenshot — local only")),
            "ocr" => Verdict::allow(ToolRisk::Safe, String::from("OCR — local only")),
            "desktop_action" => Verdict::allow(
                ToolRisk::Moderate,
                String::from("Desktop control via xdotool"),
            ),
            _ => Verdict::allow(
                ToolRisk::Moderate,
                format!("Unknown tool '{tool_name}' — treating as Moderate risk"),
            ),
        }
    }
}

// ── Pending action (awaiting UI confirmation) ─────────────────────────────────

#[derive(Debug, Clone)]
pub struct PendingAction {
    pub id: u64,
    pub tool: String,
    pub args: Vec<String>,
    pub verdict: Verdict,
    pub raw_tag: String,
}

impl PendingAction {
    pub fn summary(&self) -> String {
        format!(
            "{} [{}] {} {}",
            self.verdict.risk.emoji(),
            self.verdict.risk.label(),
            self.tool,
            self.args.join(" "),
        )
    }
}
