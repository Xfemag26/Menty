// tools.rs - Lightweight toolset (Rust-only, no heavy deps)
// Path resolution is delegated to config.rs; external programs are optional adapters.

use crate::memory::Memory;
use crate::permissions::{Policy, ToolRisk};
use scraper::{Html, Selector};
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct ToolResult {
    pub output: String,
    pub error: Option<ToolError>,
}

impl ToolResult {
    pub fn ok(s: impl Into<String>) -> Self {
        Self {
            output: s.into(),
            error: None,
        }
    }
    pub fn err(s: impl Into<String>) -> Self {
        Self {
            output: s.into(),
            error: Some(ToolError::Unknown("Generic error".into())),
        }
    }
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum ToolError {
    ResourceExceeded(String),
    CommandFailed(String),
    PermissionDenied(String),
    Timeout(String),
    Unknown(String),
}

fn supervised_run(cmd_name: &str, args: &[&str]) -> ToolResult {
    let child = match Command::new(cmd_name)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return ToolResult::err(format!("Failed to spawn {cmd_name}: {e}")),
    };
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return ToolResult::err(format!("IO error running {cmd_name}: {e}")),
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    if output.status.success() {
        ToolResult::ok(stdout)
    } else {
        ToolResult::err(format!(
            "Command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

pub fn search_web(query: &str) -> ToolResult {
    let encoded = urlencoding::encode(query);
    let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);
    let resp = match ureq::get(&url)
        .header(
            "User-Agent",
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        )
        .call()
    {
        Ok(r) => r,
        Err(e) => return ToolResult::err(format!("Search failed: {e}")),
    };
    let body = match resp.into_body().read_to_string() {
        Ok(b) => b,
        Err(e) => return ToolResult::err(format!("Read failed: {e}")),
    };
    let doc = Html::parse_document(&body);
    let result_sel = Selector::parse(".result__body").unwrap();
    let title_sel = Selector::parse(".result__title").unwrap();
    let snippet_sel = Selector::parse(".result__snippet").unwrap();
    let url_sel = Selector::parse(".result__url").unwrap();
    let mut results = Vec::new();
    for el in doc.select(&result_sel).take(3) {
        let title = el
            .select(&title_sel)
            .next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default();
        let snippet = el
            .select(&snippet_sel)
            .next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default();
        let href = el
            .select(&url_sel)
            .next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default();
        if !title.is_empty() {
            results.push(format!(
                "• {}\n  {}\n  {}",
                title.trim(),
                snippet.trim(),
                href.trim()
            ));
        }
    }
    if results.is_empty() {
        ToolResult::err("No results found.")
    } else {
        ToolResult::ok(results.join("\n\n"))
    }
}

pub fn read_file(filepath: &str, offset: usize) -> ToolResult {
    match std::fs::read_to_string(filepath) {
        Err(e) => ToolResult::err(format!("❌ Cannot read {filepath}: {e}")),
        Ok(content) => {
            let chunk: String = content.chars().skip(offset).take(2000).collect();
            if chunk.is_empty() {
                ToolResult::ok("✅ End of file.")
            } else if offset + 2000 < content.chars().count() {
                ToolResult::ok(format!(
                    "{chunk}\n\n--- [CHUNK END] Next offset: {} ---",
                    offset + 2000
                ))
            } else {
                ToolResult::ok(chunk)
            }
        }
    }
}

pub fn create_file(filepath: &str, content: &str) -> ToolResult {
    write_file(filepath, content)
}
pub fn write_file(filepath: &str, content: &str) -> ToolResult {
    if content.trim().is_empty() {
        return ToolResult::err(format!("❌ write_file needs the text as the second argument. Nothing was written. Retry exactly:\n[TOOL: write_file \"{filepath}\" \"the full text to save\"]"));
    }
    if let Some(parent) = std::path::Path::new(filepath).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(filepath, content) {
        Ok(_) => ToolResult::ok(format!("✅ Written {} bytes to {filepath}", content.len())),
        Err(e) => ToolResult::err(format!("❌ Write failed: {e}")),
    }
}

pub fn append_to_file(filepath: &str, content: &str) -> ToolResult {
    use std::io::Write;
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(filepath)
    {
        Err(e) => ToolResult::err(format!("❌ Cannot open {filepath}: {e}")),
        Ok(mut f) => match writeln!(f, "{}", content) {
            Ok(_) => ToolResult::ok(format!("✅ Appended to {filepath}")),
            Err(e) => ToolResult::err(format!("❌ Write error: {e}")),
        },
    }
}

pub fn list_directory(dirpath: &str) -> ToolResult {
    match std::fs::read_dir(dirpath) {
        Err(e) => ToolResult::err(format!("❌ Cannot list {dirpath}: {e}")),
        Ok(entries) => {
            let mut files: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let icon = if e.path().is_dir() { "📁" } else { "📄" };
                    format!("{icon} {name}")
                })
                .collect();
            files.sort();
            ToolResult::ok(files.join("\n"))
        }
    }
}

pub fn search_in_file(filepath: &str, pattern: &str) -> ToolResult {
    match std::fs::read_to_string(filepath) {
        Err(e) => ToolResult::err(format!("❌ Cannot read {filepath}: {e}")),
        Ok(content) => {
            let needle = pattern.to_lowercase();
            let matches: Vec<String> = content
                .lines()
                .enumerate()
                .filter(|(_, line)| line.to_lowercase().contains(&needle))
                .map(|(i, line)| format!("Line {}: {}", i + 1, line.trim()))
                .collect();
            if matches.is_empty() {
                ToolResult::ok(format!("No matches for '{pattern}'"))
            } else {
                ToolResult::ok(matches.join("\n"))
            }
        }
    }
}

pub fn open_url(url: &str) -> ToolResult {
    if !(url.starts_with("http://") || url.starts_with("https://") || url.starts_with("file://")) {
        return ToolResult::err(format!(
            "open_url needs http(s):// or file:// URL, got: {url}"
        ));
    }
    match Command::new("xdg-open").arg(url).spawn() {
        Ok(_) => ToolResult::ok(format!("✅ Opened {url}")),
        Err(e) => ToolResult::err(format!("❌ xdg-open failed: {e}")),
    }
}

// ── General autonomous desktop tools ──────────────────────────────────────────
pub fn shell(cmdline: &str) -> ToolResult {
    let cmdline = cmdline.trim();
    if cmdline.is_empty() {
        return ToolResult::err("shell: empty command");
    }
    let hard_blocked = [
        "rm -rf /",
        "mkfs",
        ":(){:|:&};:",
        "dd if=/dev/zero of=/dev/",
    ];
    for blocked in hard_blocked {
        if cmdline.contains(blocked) {
            return ToolResult::err(format!(
                "shell: refusing destructive command containing '{blocked}'"
            ));
        }
    }
    match Command::new("bash").args(["-c", cmdline]).output() {
        Ok(o) => {
            let mut out = String::from_utf8_lossy(&o.stdout).to_string();
            let err = String::from_utf8_lossy(&o.stderr);
            if !err.is_empty() {
                out.push_str("\n[stderr]\n");
                out.push_str(&err);
            }
            out.push_str(&format!("\n[exit: {}]", o.status));
            let out = out.trim_end().chars().take(4000).collect::<String>();
            if o.status.success() {
                ToolResult::ok(out)
            } else {
                ToolResult::err(out)
            }
        }
        Err(e) => ToolResult::err(format!("bash not found: {e}")),
    }
}

pub fn screen_read(region: &str) -> ToolResult {
    let tmp = "/tmp/momentum_screen_read.png";
    let capture = if region.trim().is_empty() {
        Command::new("scrot").args(["--overwrite", tmp]).output()
    } else {
        Command::new("scrot")
            .args(["--overwrite", "-a", region.trim(), tmp])
            .output()
    };
    if !capture.map(|o| o.status.success()).unwrap_or(false) {
        return ToolResult::err("screen_read: scrot failed");
    }
    let tsv = Command::new("tesseract")
        .args([tmp, "stdout", "--psm", "3", "tsv"])
        .output();
    let plain = Command::new("tesseract")
        .args([tmp, "stdout", "--psm", "3"])
        .output();
    match (plain, tsv) {
        (Ok(p), Ok(t)) => {
            let text = String::from_utf8_lossy(&p.stdout).trim().to_string();
            if text.is_empty() {
                return ToolResult::ok("screen_read: nothing readable on screen");
            }
            let mut positions = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for line in String::from_utf8_lossy(&t.stdout).lines().skip(1) {
                let cols: Vec<&str> = line.split('\t').collect();
                if cols.len() < 12 || cols[10].parse::<f32>().unwrap_or(0.0) < 60.0 {
                    continue;
                }
                let word = cols[11].trim();
                if word.len() < 2 || !seen.insert(word.to_lowercase()) {
                    continue;
                }
                let x =
                    cols[6].parse::<i32>().unwrap_or(0) + cols[8].parse::<i32>().unwrap_or(0) / 2;
                let y =
                    cols[7].parse::<i32>().unwrap_or(0) + cols[9].parse::<i32>().unwrap_or(0) / 2;
                positions.push(format!("{word}→({x},{y})"));
                if positions.len() >= 40 {
                    break;
                }
            }
            ToolResult::ok(format!(
                "SCREEN TEXT:\n{}\n\nWORD POSITIONS (click targets):\n{}",
                text.chars().take(2000).collect::<String>(),
                positions.join(" ")
            ))
        }
        _ => ToolResult::err("screen_read: tesseract failed"),
    }
}

pub fn find_and_click(text: &str, button: &str) -> ToolResult {
    let needle = text.trim().to_lowercase();
    if needle.is_empty() {
        return ToolResult::err("find_and_click: need text to find");
    }
    let tmp = "/tmp/momentum_find_click.png";
    if !Command::new("scrot")
        .args(["--overwrite", tmp])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return ToolResult::err("find_and_click: screenshot failed");
    }
    let tsv = match Command::new("tesseract")
        .args([tmp, "stdout", "--psm", "3", "tsv"])
        .output()
    {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => return ToolResult::err(format!("find_and_click: tesseract failed: {e}")),
    };
    let mut found = None;
    for line in tsv.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 12 || cols[10].parse::<f32>().unwrap_or(0.0) < 50.0 {
            continue;
        }
        let word = cols[11].trim().to_lowercase();
        if word.contains(&needle) || (needle.contains(&word) && word.len() >= 3) {
            let x = cols[6].parse::<i32>().unwrap_or(0) + cols[8].parse::<i32>().unwrap_or(0) / 2;
            let y = cols[7].parse::<i32>().unwrap_or(0) + cols[9].parse::<i32>().unwrap_or(0) / 2;
            found = Some((x, y));
            break;
        }
    }
    let Some((x, y)) = found else {
        return ToolResult::err(format!(
            "find_and_click: '{text}' not found on screen. Use screen_read first."
        ));
    };
    let button = if button.trim().is_empty() {
        "1"
    } else {
        button.trim()
    };
    match Command::new("xdotool")
        .args(["mousemove", &x.to_string(), &y.to_string(), "click", button])
        .output()
    {
        Ok(o) if o.status.success() => {
            ToolResult::ok(format!("✅ Clicked '{text}' at ({x},{y}) button={button}"))
        }
        Ok(o) => ToolResult::err(format!(
            "xdotool failed: {}",
            String::from_utf8_lossy(&o.stderr)
        )),
        Err(e) => ToolResult::err(format!("xdotool not found: {e}")),
    }
}

pub fn window_list() -> ToolResult {
    match Command::new("wmctrl").args(["-l"]).output() {
        Ok(o) if o.status.success() => ToolResult::ok(
            String::from_utf8_lossy(&o.stdout)
                .chars()
                .take(2000)
                .collect::<String>(),
        ),
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(_) => ToolResult::err("wmctrl not found — install wmctrl"),
    }
}

pub fn window_focus(title: &str) -> ToolResult {
    match Command::new("wmctrl").args(["-a", title.trim()]).output() {
        Ok(o) if o.status.success() => {
            ToolResult::ok(format!("✅ Focused window matching '{title}'"))
        }
        _ => ToolResult::err(format!("No window matching '{title}'")),
    }
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

pub fn clipboard(action: &str, text: &str) -> ToolResult {
    match action.trim() {
        "read" => match Command::new("bash").args(["-c", "xsel --clipboard --output 2>/dev/null || xclip -o -selection clipboard 2>/dev/null || echo ''"]).output() {
            Ok(o) => {
                let content = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if content.is_empty() { ToolResult::ok("clipboard is empty") } else { ToolResult::ok(content.chars().take(2000).collect::<String>()) }
            }
            Err(e) => ToolResult::err(format!("clipboard read failed: {e}")),
        },
        "write" if !text.is_empty() => {
            let quoted = shell_escape(text);
            let command = format!("printf '%s' {quoted} | xsel --clipboard --input 2>/dev/null || printf '%s' {quoted} | xclip -selection clipboard 2>/dev/null");
            match Command::new("bash").args(["-c", &command]).output() {
                Ok(o) if o.status.success() => ToolResult::ok(format!("✅ Copied {} chars", text.chars().count())),
                Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
                Err(e) => ToolResult::err(format!("clipboard write failed: {e}")),
            }
        }
        "write" => ToolResult::err("clipboard write: need text"),
        _ => ToolResult::err("clipboard: action must be 'read' | 'write'"),
    }
}

pub fn notify(title: &str, body: &str, urgency: &str) -> ToolResult {
    let urgency = match urgency.trim() {
        "low" | "critical" => urgency.trim(),
        _ => "normal",
    };
    let mut command = Command::new("notify-send");
    command.args(["-u", urgency, title.trim()]);
    if !body.trim().is_empty() {
        command.arg(body.trim());
    }
    match command.output() {
        Ok(o) if o.status.success() => ToolResult::ok(format!("✅ Notified: {}", title.trim())),
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => ToolResult::err(format!("notify-send failed: {e}")),
    }
}

static ALLOWED_BINS: &[&str] = &[
    "blender", "python3", "python", "ls", "cat", "echo", "head", "wc", "ffmpeg", "git", "arecord",
];

pub fn transcribe(audio_path: &str) -> ToolResult {
    let path = audio_path.trim();
    if path.is_empty() {
        return ToolResult::err("transcribe: need a local audio path (flac/mp3/ogg/wav)");
    }
    if path.contains("..") || path.starts_with('-') {
        return ToolResult::err("transcribe: refuse relative-escape or flag-like path");
    }
    if ![".wav", ".mp3", ".flac", ".ogg"]
        .iter()
        .any(|ext| path.ends_with(ext))
    {
        return ToolResult::err(format!("transcribe: unsupported audio extension: {path}"));
    }
    if !std::path::Path::new(path).is_file() {
        return ToolResult::err(format!("transcribe: file not found: {path}"));
    }
    let cli = crate::config::whisper_cli_path();
    let model = crate::config::whisper_model_path();
    if !model.is_file() {
        return ToolResult::err(format!("transcribe: model missing at {}", model.display()));
    }
    let model_arg = model.to_string_lossy().into_owned();
    let threads = crate::config::N_THREADS.to_string();
    match Command::new(cli)
        .args(["-m", &model_arg, "-nt", "-t", &threads, path])
        .output()
    {
        Ok(o) => {
            let mut out = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if out.is_empty() {
                out = String::from_utf8_lossy(&o.stderr)
                    .chars()
                    .take(800)
                    .collect();
            }
            if o.status.success() {
                ToolResult::ok(out.chars().take(4000).collect::<String>())
            } else {
                ToolResult::err(out.chars().take(4000).collect::<String>())
            }
        }
        Err(e) => ToolResult::err(format!("whisper-cli spawn failed: {e}")),
    }
}

const SPEAK_MAX: usize = 500;
pub fn speak(text: &str) -> ToolResult {
    let text = text.trim();
    if text.is_empty() {
        return ToolResult::err("speak: empty text");
    }
    if text.chars().count() > SPEAK_MAX {
        return ToolResult::err(format!("speak: text longer than {SPEAK_MAX} chars"));
    }
    match Command::new("spd-say").args(["--wait", text]).output() {
        Ok(o) if o.status.success() => {
            ToolResult::ok(format!("spoke {} chars", text.chars().count()))
        }
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => ToolResult::err(format!("spd-say spawn failed: {e}")),
    }
}

pub fn screenshot() -> ToolResult {
    let out = "/tmp/momentum_screen.png";
    match Command::new("scrot").args(["--overwrite", out]).output() {
        Ok(o) if o.status.success() => ToolResult::ok(format!("screenshot saved to {out}")),
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => ToolResult::err(format!("scrot not found: {e}")),
    }
}

pub fn ocr(image_path: &str) -> ToolResult {
    let path = if image_path.trim().is_empty() {
        "/tmp/momentum_screen.png"
    } else {
        image_path.trim()
    };
    if path.contains("..") || path.starts_with('-') {
        return ToolResult::err("ocr: invalid path");
    }
    if !std::path::Path::new(path).is_file() {
        return ToolResult::err(format!("ocr: image not found: {path}"));
    }
    match Command::new("tesseract").args([path, "stdout"]).output() {
        Ok(o) if o.status.success() => ToolResult::ok(
            String::from_utf8_lossy(&o.stdout)
                .chars()
                .take(4000)
                .collect::<String>(),
        ),
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => ToolResult::err(format!("tesseract not found: {e}")),
    }
}

pub fn desktop_action(action: &str) -> ToolResult {
    let action = action.trim();
    if action.is_empty() {
        return ToolResult::err("desktop_action: empty action");
    }
    let blocked = [
        "ctrl+alt+del",
        "ctrl+alt+f2",
        "ctrl+alt+f3",
        "ctrl+alt+backspace",
        "key ctrl+alt+delete",
    ];
    for pattern in blocked {
        if action.to_ascii_lowercase().contains(pattern) {
            return ToolResult::err(format!("desktop_action: blocked action '{pattern}'"));
        }
    }
    let args = shell_split(action);
    match Command::new("xdotool").args(args).output() {
        Ok(o) if o.status.success() => ToolResult::ok(format!("ok: xdotool {action}")),
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => ToolResult::err(format!("xdotool not found: {e}")),
    }
}

pub fn run_command(cmdline: &str) -> ToolResult {
    shell(cmdline)
}

fn blender_bin() -> std::path::PathBuf {
    if Command::new("blender").arg("--version").output().is_ok() {
        std::path::PathBuf::from("blender")
    } else {
        crate::config::blender_path()
    }
}

pub fn blender_script(code: &str) -> ToolResult {
    let script_path = format!("/tmp/momentum_blender_{}.py", std::process::id());
    let output = crate::config::creative_output_dir()
        .to_string_lossy()
        .into_owned();
    let source = format!("BLENDER_OUTPUT = r'''{}'''\n{}", output, code);
    if let Err(e) = std::fs::write(&script_path, source) {
        return ToolResult::err(format!("temporary script write failed: {e}"));
    }
    let result = Command::new(blender_bin())
        .args(["--background", "--python", &script_path])
        .output();
    let _ = std::fs::remove_file(&script_path);
    match result {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr);
            if o.status.success() {
                ToolResult::ok(stdout.chars().take(4000).collect::<String>())
            } else {
                ToolResult::err(format!(
                    "{}\n{}",
                    stdout.chars().take(1000).collect::<String>(),
                    stderr.chars().take(2000).collect::<String>()
                ))
            }
        }
        Err(e) => ToolResult::err(format!("Blender not found: {e}")),
    }
}

pub fn prproj_read(path: &str) -> ToolResult {
    if !std::path::Path::new(path).is_file() {
        return ToolResult::err(format!("prproj_read: file not found: {path}"));
    }
    let path_arg = path.to_string();
    let script = r#"import gzip, xml.etree.ElementTree as ET, json, sys
p=sys.argv[1]
with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f: root=ET.fromstring(f.read())
def local(tag): return tag.rsplit('}',1)[-1]
def val(node,name,default=''):
 for x in node.iter():
  if local(x.tag)==name and x.text: return x.text.strip()
 return default
clips=[]
for c in root.iter():
 if local(c.tag)=='ClipItem': clips.append({'name':val(c,'name','(unnamed)'),'start':val(c,'start','0'),'end':val(c,'end','?'),'in':val(c,'in','0'),'out':val(c,'out','?')})
print(json.dumps({'sequence_count':sum(1 for x in root.iter() if local(x.tag)=='Sequence'),'clips':clips[:40]},indent=2))"#;
    match Command::new(crate::config::python_path())
        .args(["-c", script, &path_arg])
        .output()
    {
        Ok(o) if o.status.success() => {
            ToolResult::ok(String::from_utf8_lossy(&o.stdout).to_string())
        }
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => ToolResult::err(format!("Python unavailable: {e}")),
    }
}

pub fn draw_frame(script: &str, output_path: &str, input_path: &str) -> ToolResult {
    let output = if output_path.trim().is_empty() {
        format!("/tmp/momentum_frame_{}.png", std::process::id())
    } else {
        output_path.trim().to_string()
    };
    if let Some(parent) = std::path::Path::new(&output).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let py = format!("from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance\nOUTPUT_PATH = r'{}'\nINPUT_PATH = r'{}'\n{}\n", output.replace('\'', "\\'"), input_path.replace('\'', "\\'"), script);
    let script_path = format!("/tmp/momentum_draw_{}.py", std::process::id());
    if let Err(e) = std::fs::write(&script_path, py) {
        return ToolResult::err(format!("temporary script write failed: {e}"));
    }
    let result = Command::new(crate::config::python_path())
        .arg(&script_path)
        .output();
    let _ = std::fs::remove_file(&script_path);
    match result {
        Ok(o) if o.status.success() => ToolResult::ok(format!("✅ Frame saved → {output}")),
        Ok(o) => ToolResult::err(
            String::from_utf8_lossy(&o.stderr)
                .chars()
                .take(1200)
                .collect::<String>(),
        ),
        Err(e) => ToolResult::err(format!("Python unavailable: {e}")),
    }
}

pub fn find_files(root: &str, pattern: &str) -> ToolResult {
    fn matches(pattern: &str, name: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        if let Some(ext) = pattern.strip_prefix("*.") {
            return name.ends_with(&format!(".{ext}"));
        }
        if let Some(prefix) = pattern.strip_suffix('*') {
            return name.starts_with(prefix);
        }
        pattern == name
    }
    fn walk(path: &std::path::Path, pattern: &str, out: &mut Vec<String>) {
        if out.len() >= 200 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            if out.len() >= 200 {
                break;
            }
            let p = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') && p.is_dir() {
                continue;
            }
            if p.is_dir() {
                walk(&p, pattern, out);
            } else if matches(pattern, &name) {
                out.push(p.to_string_lossy().into_owned());
            }
        }
    }
    let mut results = Vec::new();
    walk(std::path::Path::new(root), pattern, &mut results);
    if results.is_empty() {
        ToolResult::ok(format!("No files matching '{pattern}' under '{root}'"))
    } else {
        ToolResult::ok(results.join("\n"))
    }
}

pub fn open_blender(blend_file: &str) -> ToolResult {
    let mut command = Command::new(blender_bin());
    if !blend_file.trim().is_empty() {
        command.arg(blend_file.trim());
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    match command.spawn() {
        Ok(child) => ToolResult::ok(format!("✅ Blender launched (pid {})", child.id())),
        Err(e) => ToolResult::err(format!("Blender launch failed: {e}")),
    }
}

pub fn kill_process(pid_str: &str) -> ToolResult {
    let pid: i32 = match pid_str.trim().parse() {
        Ok(v) if v > 1 => v,
        _ => return ToolResult::err(format!("invalid or protected pid '{pid_str}'")),
    };
    let result = unsafe { libc::kill(pid, libc::SIGTERM) };
    if result == 0 {
        ToolResult::ok(format!("✅ SIGTERM pid {pid}"))
    } else {
        ToolResult::err(format!("kill failed: {}", std::io::Error::last_os_error()))
    }
}

fn shell_split(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut started = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                started = true;
                while let Some(next) = chars.next() {
                    match next {
                        '"' => break,
                        '\\' => {
                            if let Some(escaped) = chars.next() {
                                current.push(escaped);
                            }
                        }
                        other => current.push(other),
                    }
                }
            }
            '\'' => {
                started = true;
                for next in chars.by_ref() {
                    if next == '\'' {
                        break;
                    } else {
                        current.push(next);
                    }
                }
            }
            ' ' | '\t' if !started => {}
            ' ' | '\t' => {
                args.push(std::mem::take(&mut current));
                started = false;
            }
            other => {
                current.push(other);
                started = true;
            }
        }
    }
    if started {
        args.push(current);
    }
    args
}

pub fn blender_render(blend_file: &str, output_dir: &str, start: &str, end: &str) -> ToolResult {
    if blend_file.trim().is_empty() || !std::path::Path::new(blend_file).is_file() {
        return ToolResult::err(format!(
            "blender_render: invalid .blend file '{}'; file must exist",
            blend_file
        ));
    }
    let out = if output_dir.trim().is_empty() {
        crate::config::creative_output_dir()
            .join("frames")
            .to_string_lossy()
            .into_owned()
    } else {
        output_dir.to_string()
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        return ToolResult::err(format!("Could not create output directory: {e}"));
    }
    let start = if start.trim().is_empty() {
        "1"
    } else {
        start.trim()
    };
    let end = if end.trim().is_empty() {
        "250"
    } else {
        end.trim()
    };
    let pattern = format!("{}/frame_", out.trim_end_matches('/'));
    let script = format!("import bpy\nbpy.context.scene.render.filepath = r'{}'\nbpy.context.scene.render.image_settings.file_format='PNG'\nbpy.context.scene.frame_start={}\nbpy.context.scene.frame_end={}\nbpy.ops.render.render(animation=True)\n", pattern, start, end);
    blender_script(&script)
}

pub fn ffmpeg_compose(operation: &str, a: &str, b: &str, c: &str, d: &str) -> ToolResult {
    let args: Vec<String> = match operation.trim() {
        "frames_to_video" => vec![
            "-framerate".into(),
            if b.is_empty() { "24".into() } else { b.into() },
            "-i".into(),
            a.into(),
            "-c:v".into(),
            "libx264".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-y".into(),
            c.into(),
        ],
        "add_audio" => vec![
            "-i".into(),
            a.into(),
            "-i".into(),
            b.into(),
            "-c:v".into(),
            "copy".into(),
            "-c:a".into(),
            "aac".into(),
            "-shortest".into(),
            "-y".into(),
            c.into(),
        ],
        "trim" => vec![
            "-ss".into(),
            b.into(),
            "-t".into(),
            c.into(),
            "-i".into(),
            a.into(),
            "-c".into(),
            "copy".into(),
            "-y".into(),
            d.into(),
        ],
        "concat" => vec![
            "-f".into(),
            "concat".into(),
            "-safe".into(),
            "0".into(),
            "-i".into(),
            a.into(),
            "-c".into(),
            "copy".into(),
            "-y".into(),
            b.into(),
        ],
        "extract_audio" => vec![
            "-i".into(),
            a.into(),
            "-vn".into(),
            "-acodec".into(),
            "pcm_s16le".into(),
            "-ar".into(),
            "44100".into(),
            "-y".into(),
            b.into(),
        ],
        "convert" => vec!["-i".into(), a.into(), "-y".into(), b.into()],
        _ => return ToolResult::err(format!("ffmpeg_compose: unknown operation '{operation}'")),
    };
    match Command::new("ffmpeg").args(args).output() {
        Ok(o) if o.status.success() => ToolResult::ok(
            String::from_utf8_lossy(&o.stderr)
                .lines()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        Ok(o) => ToolResult::err(
            String::from_utf8_lossy(&o.stderr)
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        Err(e) => ToolResult::err(format!("ffmpeg not found: {e}")),
    }
}

pub fn download_sfx(url: &str, output_path: &str) -> ToolResult {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return ToolResult::err("download_sfx: need an http(s) URL");
    }
    let output = if output_path.trim().is_empty() {
        format!("/tmp/momentum_sfx_{}.wav", std::process::id())
    } else {
        output_path.trim().to_owned()
    };
    if let Some(parent) = std::path::Path::new(&output).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match Command::new("yt-dlp")
        .args([
            "--no-playlist",
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "-o",
            &output,
            url,
        ])
        .output()
    {
        Ok(o) if o.status.success() => ToolResult::ok(format!("✅ Downloaded SFX → {output}")),
        Ok(o) => ToolResult::err(
            String::from_utf8_lossy(&o.stderr)
                .chars()
                .take(1000)
                .collect::<String>(),
        ),
        Err(e) => ToolResult::err(format!("yt-dlp not found: {e}")),
    }
}

pub fn piper_tts(text: &str, output_wav: &str) -> ToolResult {
    let text = text.trim();
    if text.is_empty() || text.chars().count() > 2000 {
        return ToolResult::err("piper_tts: text must be 1–2000 characters");
    }
    let output = if output_wav.trim().is_empty() {
        "/tmp/momentum_tts.wav".to_owned()
    } else {
        output_wav.trim().to_owned()
    };
    let binary = crate::config::piper_bin_path();
    let model = crate::config::piper_model_path();
    if !binary.is_file() {
        return speak(&text.chars().take(500).collect::<String>());
    }
    if !model.is_file() {
        return ToolResult::err(format!("piper model missing: {}", model.display()));
    }
    let model_arg = model.to_string_lossy().into_owned();
    let mut child = match Command::new(binary)
        .args(["--model", &model_arg, "--output_file", &output])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => return ToolResult::err(format!("piper spawn failed: {e}")),
    };
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = std::io::Write::write_all(stdin, text.as_bytes());
    }
    match child.wait_with_output() {
        Ok(o) if o.status.success() => ToolResult::ok(format!("✅ TTS → {output}")),
        Ok(o) => ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => ToolResult::err(format!("piper wait failed: {e}")),
    }
}

pub fn ebsynth_run(
    keyframe: &str,
    frames_dir: &str,
    output_dir: &str,
    key_number: &str,
) -> ToolResult {
    if !std::path::Path::new(keyframe).is_file() || !std::path::Path::new(frames_dir).is_dir() {
        return ToolResult::err("ebsynth_run: keyframe/frames directory missing");
    }
    let root = crate::config::ezsynth_root();
    let root_s = root.to_string_lossy().into_owned();
    if !root.is_dir() {
        return ToolResult::err(format!("Ezsynth not found at {}", root.display()));
    }
    let out = if output_dir.trim().is_empty() {
        format!("{frames_dir}_styled")
    } else {
        output_dir.to_owned()
    };
    let _ = std::fs::create_dir_all(&out);
    let key_n = if key_number.trim().is_empty() {
        "1"
    } else {
        key_number.trim()
    };
    let py = format!(
        r#"
import sys, numpy as np, glob, os
from PIL import Image
root = r'''{root_s}'''; sys.path.insert(0, root)
from ezsynth.utils._ebsynth import ebsynth
keyframe = r'''{keyframe}'''; frames_dir = r'''{frames_dir}'''; out_dir = r'''{out}'''; key_n = int('{key_n}')
style = np.array(Image.open(keyframe).convert('RGB'))
frames = sorted(glob.glob(os.path.join(frames_dir,'*.png')) + glob.glob(os.path.join(frames_dir,'*.jpg')))
source_path = next((p for p in frames if f'{{key_n:04d}}' in os.path.basename(p)), None)
if source_path is None: raise RuntimeError('keyframe frame not found')
source = np.array(Image.open(source_path).convert('RGB')); engine = ebsynth()
for path in frames:
    target = np.array(Image.open(path).convert('RGB')); result, err = engine.run(style, [(source,target,1.0)])
    Image.fromarray(result).save(os.path.join(out_dir,os.path.basename(path)))
"#
    );
    let python = crate::config::python_path();
    let python = python.to_string_lossy().into_owned();
    supervised_run(&python, &["-c", &py])
}

pub async fn web_learn(query: &str, mem: &Arc<Mutex<Memory>>) -> ToolResult {
    let query = query.trim();
    if query.is_empty() {
        return ToolResult::err("web_learn: empty query");
    }
    let result = search_web(query);
    if !result.is_success() {
        return ToolResult::err(format!("web_learn: search failed — {}", result.output));
    }
    let content: String = result.output.chars().take(600).collect();
    let label = format!("web:{}", query.chars().take(50).collect::<String>());
    let mem = mem.lock().await;
    match mem.add_node(&label, "concept", &content) {
        Ok(id) => ToolResult::ok(format!(
            "✅ Learned and stored: \"{label}\" (id={id})\n\n{content}"
        )),
        Err(e) => ToolResult::err(format!("web_learn: memory store failed — {e}")),
    }
}

pub fn install_piper() -> ToolResult {
    let binary = crate::config::piper_bin_path();
    let model = crate::config::piper_model_path();
    let bin_dir = binary.parent().unwrap_or(std::path::Path::new("."));
    let voice_dir = model.parent().unwrap_or(std::path::Path::new("."));
    if let Err(e) = std::fs::create_dir_all(bin_dir) {
        return ToolResult::err(format!("cannot create Piper directory: {e}"));
    }
    if let Err(e) = std::fs::create_dir_all(voice_dir) {
        return ToolResult::err(format!("cannot create Piper voice directory: {e}"));
    }
    if binary.is_file() && model.is_file() {
        return ToolResult::ok(format!(
            "✅ Piper already installed at {}",
            bin_dir.display()
        ));
    }
    let binary_s = binary.to_string_lossy().into_owned();
    let model_s = model.to_string_lossy().into_owned();
    let bin_dir_s = bin_dir.to_string_lossy().into_owned();
    let tar = bin_dir.join("piper.tar.gz").to_string_lossy().into_owned();
    let mut messages = String::new();
    if !binary.is_file() {
        let url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz";
        match Command::new("curl").args(["-L", "-o", &tar, url]).output() {
            Ok(o) if o.status.success() => {
                let extract = Command::new("tar")
                    .args(["-xzf", &tar, "-C", &bin_dir_s])
                    .output();
                if !extract
                    .as_ref()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return ToolResult::err("Piper archive extraction failed");
                }
                let _ = std::fs::remove_file(&tar);
                let _ = Command::new("chmod").args(["+x", &binary_s]).output();
                messages.push_str("Piper binary downloaded.\n");
            }
            Ok(o) => return ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
            Err(e) => return ToolResult::err(format!("curl unavailable: {e}")),
        }
    }
    if !model.is_file() {
        let urls = [
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        ];
        for (url, dest) in urls
            .iter()
            .zip([model_s.clone(), format!("{model_s}.json")])
        {
            match Command::new("curl").args(["-L", "-o", &dest, url]).output() {
                Ok(o) if o.status.success() => messages.push_str(&format!("Downloaded {dest}\n")),
                Ok(o) => return ToolResult::err(String::from_utf8_lossy(&o.stderr).to_string()),
                Err(e) => return ToolResult::err(format!("curl unavailable: {e}")),
            }
        }
    }
    ToolResult::ok(format!("Piper installed.\n{messages}"))
}

pub fn install_ebsynth() -> ToolResult {
    let library = crate::config::ezsynth_root().join("ebsynth/lib/libebsynth.so");
    if library.is_file() {
        ToolResult::ok(format!("✅ Ezsynth present at {}", library.display()))
    } else {
        ToolResult::err(format!("Ebsynth library not found: {}", library.display()))
    }
}

pub fn dispatch(tag: &str) -> Option<ToolResult> {
    let inner = tag
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_start_matches("TOOL:")
        .trim();
    let parts = shell_split(inner);
    if parts.is_empty() {
        return None;
    }
    let name = parts[0].as_str();
    let args: Vec<&str> = parts[1..].iter().map(String::as_str).collect();
    let verdict = Policy::default().check(name, args.first().copied());
    if verdict.risk == ToolRisk::Denied {
        return Some(ToolResult::err(format!("denied: {}", verdict.reason)));
    }
    let result = match name {
        "search_web" => search_web(args.first().copied().unwrap_or("")),
        "read_file" => read_file(args.first().copied().unwrap_or(""), args.get(1).and_then(|x|x.parse().ok()).unwrap_or(0)),
        "write_file" | "create_file" => write_file(args.first().copied().unwrap_or(""), args.get(1).copied().unwrap_or("")),
        "append_to_file" => append_to_file(args.first().copied().unwrap_or(""), args.get(1).copied().unwrap_or("")),
        "list_directory" => list_directory(args.first().copied().unwrap_or("")),
        "search_in_file" => search_in_file(args.first().copied().unwrap_or(""), args.get(1).copied().unwrap_or("")),
        "open_url" | "browse" | "open_browser" => open_url(args.first().copied().unwrap_or("")),
        "run_command" => run_command(args.first().copied().unwrap_or("")),
        "blender_script" => blender_script(args.first().copied().unwrap_or("")),
        "kill_process" => kill_process(args.first().copied().unwrap_or("")),
        "find_files" => find_files(args.first().copied().unwrap_or("/"), args.get(1).copied().unwrap_or("*")),
        "open_blender" => open_blender(args.first().copied().unwrap_or("")),
        "transcribe" | "whisper" => transcribe(args.first().copied().unwrap_or("")),
        "speak" | "say" => speak(args.first().copied().unwrap_or("")),
        "screenshot" => screenshot(),
        "ocr" => ocr(args.first().copied().unwrap_or("")),
        "desktop_action" => desktop_action(args.first().copied().unwrap_or("")),
        "blender_render" => blender_render(args.get(0).copied().unwrap_or(""), args.get(1).copied().unwrap_or(""), args.get(2).copied().unwrap_or(""), args.get(3).copied().unwrap_or("")),
        "ffmpeg_compose" => ffmpeg_compose(args.get(0).copied().unwrap_or(""), args.get(1).copied().unwrap_or(""), args.get(2).copied().unwrap_or(""), args.get(3).copied().unwrap_or(""), args.get(4).copied().unwrap_or("")),
        "download_sfx" => download_sfx(args.get(0).copied().unwrap_or(""), args.get(1).copied().unwrap_or("")),
        "piper_tts" | "tts" => piper_tts(args.get(0).copied().unwrap_or(""), args.get(1).copied().unwrap_or("")),
        "ebsynth_run" | "ebsynth" => ebsynth_run(args.get(0).copied().unwrap_or(""), args.get(1).copied().unwrap_or(""), args.get(2).copied().unwrap_or(""), args.get(3).copied().unwrap_or("")),
        "install_piper" => install_piper(),
        "install_ebsynth" => install_ebsynth(),
        "prproj_read" | "premiere_read" => prproj_read(args.first().copied().unwrap_or("")),
        "draw_frame" => draw_frame(args.first().copied().unwrap_or(""), args.get(1).copied().unwrap_or(""), args.get(2).copied().unwrap_or("")),
        "shell" | "bash" | "cmd" => shell(args.first().copied().unwrap_or("")),
        "screen_read" => screen_read(args.first().copied().unwrap_or("")),
        "find_and_click" => find_and_click(args.first().copied().unwrap_or(""), args.get(1).copied().unwrap_or("")),
        "window_list" | "windows" => window_list(),
        "window_focus" => window_focus(args.first().copied().unwrap_or("")),
        "clipboard" => clipboard(args.first().copied().unwrap_or("read"), args.get(1).copied().unwrap_or("")),
        "notify" => notify(args.first().copied().unwrap_or(""), args.get(1).copied().unwrap_or(""), args.get(2).copied().unwrap_or("")),
        _ => ToolResult::err(format!("Unknown tool: {name}. Use one of the documented file, desktop, web, memory, Blender, audio or media tools.")),
    };
    Some(result)
}

pub async fn dispatch_with_memory(tag: &str, mem: &Arc<Mutex<Memory>>) -> Option<ToolResult> {
    let inner = tag
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_start_matches("TOOL:")
        .trim();
    let parts = shell_split(inner);
    if parts.is_empty() {
        return None;
    }
    let name = parts[0].as_str();
    match name {
        "memory_search" => {
            let query = parts.get(1).map(String::as_str).unwrap_or("");
            if query.is_empty() {
                return Some(ToolResult::err("memory_search requires a query"));
            }
            let mem = mem.lock().await;
            match mem.search_nodes(query, 8) {
                Ok(hits) if hits.is_empty() => Some(ToolResult::ok("No matching memories.")),
                Ok(hits) => Some(ToolResult::ok(
                    hits.iter()
                        .map(|(n, s)| {
                            format!(
                                "id={} label={} type={} score={:.2}\n  {}",
                                n.id, n.label, n.node_type, s, n.content
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                )),
                Err(e) => Some(ToolResult::err(format!("memory_search failed: {e}"))),
            }
        }
        "memory_add" => {
            let label = parts.get(1).map(String::as_str).unwrap_or("");
            let kind = parts.get(2).map(String::as_str).unwrap_or("note");
            let content = parts.get(3).map(String::as_str).unwrap_or("");
            if label.is_empty() {
                return Some(ToolResult::err("memory_add requires a label"));
            }
            match mem.lock().await.add_node(label, kind, content) {
                Ok(id) => Some(ToolResult::ok(format!("Memory added. id={id}"))),
                Err(e) => Some(ToolResult::err(format!("memory_add failed: {e}"))),
            }
        }
        "memory_update" => {
            let id = parts.get(1).map(String::as_str).unwrap_or("");
            let label = parts.get(2).map(String::as_str).unwrap_or("");
            let content = parts.get(3).map(String::as_str).unwrap_or("");
            match mem.lock().await.update_node(id, label, content) {
                Ok(()) => Some(ToolResult::ok(format!("Memory {id} updated."))),
                Err(e) => Some(ToolResult::err(format!("memory_update failed: {e}"))),
            }
        }
        "memory_hide" => {
            let id = parts.get(1).map(String::as_str).unwrap_or("");
            match mem.lock().await.hide_node(id) {
                Ok(()) => Some(ToolResult::ok(format!("Memory {id} hidden."))),
                Err(e) => Some(ToolResult::err(format!("memory_hide failed: {e}"))),
            }
        }
        "memory_delete" => {
            let id = parts.get(1).map(String::as_str).unwrap_or("");
            match mem.lock().await.delete_node(id) {
                Ok(()) => Some(ToolResult::ok(format!("Memory {id} permanently deleted."))),
                Err(e) => Some(ToolResult::err(format!("memory_delete failed: {e}"))),
            }
        }
        "memory_get" => {
            let query = parts.get(1).map(String::as_str).unwrap_or("");
            Some(ToolResult::ok(mem.lock().await.get_context(query)))
        }
        "web_learn" | "learn" => {
            let query = parts.get(1).map(String::as_str).unwrap_or("");
            Some(web_learn(query, mem).await)
        }
        _ => dispatch(tag),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_shell_split() {
        assert_eq!(
            shell_split("search_web \"Rust lang\" \"second arg\""),
            vec!["search_web", "Rust lang", "second arg"]
        );
    }
    #[test]
    fn test_dispatch_file_ops() {
        let path = "/tmp/momentum_test_file.txt";
        assert!(
            dispatch(&format!("[TOOL: write_file \"{path}\" \"hello momentum\"]"))
                .unwrap()
                .is_success()
        );
        assert_eq!(
            dispatch(&format!("[TOOL: read_file \"{path}\" 0]"))
                .unwrap()
                .output,
            "hello momentum"
        );
        assert!(
            dispatch(&format!("[TOOL: search_in_file \"{path}\" \"momentum\"]"))
                .unwrap()
                .output
                .contains("Line 1")
        );
        let _ = std::fs::remove_file(path);
    }
}
