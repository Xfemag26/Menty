// memory.rs - Unified SQLite memory layer
// Replaces: skill_diary.json + memory_graph.json + ChromaDB (memory.py) + memory_graph.py
// Why SQLite: single file, zero server process, handles all three memory types,
// built-in FTS5 for keyword search alongside our vector cosine search.
//
// Hindsight-inspired additions:
//   • Ingestion sinkhole  — regex-based secret/PII redaction before any INSERT
//   • FTS5 full-text index — BM25 ranked keyword search, replaces slow LIKE scans
//   • 1-hop graph context  — seed nodes → typed edge expansion for richer recall
//   • Dual-speed decay     — episodic sessions decay 5× faster than semantic nodes
//   • Graph compaction     — hard cap of MAX_LIVE_NODES, weakest leaves evicted first
//   • evidence_count       — reinforcement counter slows decay on repeatedly recalled nodes

use crate::config;
use crate::llm::{cosine_sim, embed_text};
use chrono::Utc;
use regex::Regex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use uuid::Uuid;

// ── Ingestion sinkhole ────────────────────────────────────────────────────────
// Pre-compiled once at first use (OnceLock → zero cost on subsequent calls).
// Patterns cover the most common secret/PII forms that should never enter the
// memory DB or context window.  Redacted to a static tag so token count stays
// predictable.
static SINKHOLE: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();

fn sinkhole_patterns() -> &'static Vec<(&'static str, Regex)> {
    SINKHOLE.get_or_init(|| {
        // (replacement_tag, pattern)
        // All patterns are anchored / bounded to avoid false positives on prose.
        let raw: &[(&str, &str)] = &[
            // GitHub / GitLab / npm / PyPI tokens
            ("[REDACTED_TOKEN]", r"gh[pousr]_[A-Za-z0-9]{36,}"),
            ("[REDACTED_TOKEN]", r"glpat-[A-Za-z0-9_-]{20,}"),
            ("[REDACTED_TOKEN]", r"npm_[A-Za-z0-9]{36,}"),
            // OpenAI / Anthropic / generic Bearer tokens
            ("[REDACTED_TOKEN]", r"sk-[A-Za-z0-9]{32,}"),
            // ASCII whitespace avoids Unicode regex features; '=' covers padded JWTs.
            ("[REDACTED_TOKEN]", r"Bearer[ \t]+[A-Za-z0-9._~+/=-]{20,}"),
            // AWS access key IDs
            ("[REDACTED_KEY]", r"AKIA[0-9A-Z]{16}"),
            // PEM private keys (any type)
            ("[REDACTED_KEY]", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
            // Generic password= / secret= assignments. Restrict names to avoid
            // treating prose such as "token=ghp_..." as a generic password field.
            (
                "[REDACTED_SECRET]",
                r"(?i:password|passwd|secret|api_key)[ \t]*[:=][ \t]*[^ \t\r\n]{8,}",
            ),
            // IPv4 addresses — keep rough structure but not exact value
            ("[REDACTED_IP]", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            // Email addresses
            (
                "[REDACTED_EMAIL]",
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            ),
        ];
        raw.iter()
            .filter_map(|(tag, pat)| Regex::new(pat).ok().map(|r| (*tag, r)))
            .collect()
    })
}

/// Redact secrets and PII from any string before it enters the DB.
/// Returns a sanitized copy; the original is never mutated.
/// Cost: one pass per pattern over the string — typically < 0.1 ms.
pub fn sanitize(input: &str) -> String {
    let mut out = input.to_owned();
    for (tag, re) in sinkhole_patterns() {
        out = re.replace_all(&out, *tag).into_owned();
    }
    out
}

// ---- Schema types ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub label: String,
    pub node_type: String, // "concept" | "fact" | "session" | "diary"
    pub content: String,
    pub strength: f32,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Puzzle {
    pub id: String,
    pub question: String,
    pub status: String, // "open" | "solved" | "abandoned"
    pub attempts: i32,
    pub created_at: String,
}

pub struct Memory {
    conn: Connection,
}

fn encode_embedding(vector: Vec<f32>) -> Vec<u8> {
    vector.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

impl Memory {
    pub fn open() -> anyhow::Result<Self> {
        let db_path = config::database_path();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        let m = Self { conn };
        m.init_schema()?;
        m.migrate_legacy()?;
        Ok(m)
    }

    fn init_schema(&self) -> anyhow::Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS nodes (
                id               TEXT PRIMARY KEY,
                label            TEXT NOT NULL,
                node_type        TEXT NOT NULL,
                content          TEXT NOT NULL DEFAULT '',
                strength         REAL NOT NULL DEFAULT 1.0,
                embedding        BLOB,
                embedding_model  TEXT NOT NULL DEFAULT '',
                evidence_count   INTEGER NOT NULL DEFAULT 0,
                last_recalled_at TEXT    NOT NULL DEFAULT '',
                created_at       TEXT    NOT NULL,
                hidden           INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS edges (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source       TEXT NOT NULL,
                target       TEXT NOT NULL,
                relationship TEXT NOT NULL,
                reason       TEXT NOT NULL DEFAULT '',
                strength     REAL NOT NULL DEFAULT 0.5,
                created_at   TEXT NOT NULL,
                UNIQUE(source, target)
            );
            CREATE TABLE IF NOT EXISTS puzzles (
                id          TEXT PRIMARY KEY,
                question    TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'open',
                attempts    INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS diary (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entry       TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                topic       TEXT NOT NULL,
                summary     TEXT NOT NULL,
                emotion     TEXT NOT NULL DEFAULT '',
                embedding   BLOB,
                created_at  TEXT NOT NULL
            );
            -- FTS5 virtual table for BM25-ranked keyword search.
            -- Mirrors label + content from nodes; kept in sync via add_node.
            -- Zero extra RAM overhead — all data lives on SQLite pages.
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                node_id UNINDEXED,
                label,
                content,
                tokenize = 'porter ascii'
            );
            -- Raw HW telemetry ring-buffer (pruned to HW_SAMPLE_KEEP_HOURS).
            -- outcome: 0=smooth, 1=crash/hot/oom, NULL=still open.
            CREATE TABLE IF NOT EXISTS hw_samples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                ram_pct     REAL NOT NULL,
                cpu_pct     REAL NOT NULL,
                temp_c      REAL NOT NULL,
                generating  INTEGER NOT NULL DEFAULT 0,
                outcome     INTEGER
            );
            -- 5-float gradient vector persisted as key/value rows.
            -- Keys: w_ram, w_cpu, w_temp, w_gen, bias
            -- Initialised bias=-2 (assume safe until we see a crash).
            CREATE TABLE IF NOT EXISTS hw_grad (
                k  TEXT PRIMARY KEY,
                v  REAL NOT NULL
            );
            -- Sentinel row: written on clean exit, cleared on boot.
            -- If missing at boot -> previous session crashed.
            CREATE TABLE IF NOT EXISTS boot_sentinel (
                id    INTEGER PRIMARY KEY CHECK (id = 1),
                clean INTEGER NOT NULL DEFAULT 0
            );
            -- Raw session ring-buffer.
            -- Stores the verbatim exchange (user+assistant turns) for one
            -- reflection audit pass. Deleted by reflection after audit, or
            -- evicted by prune_raw_sessions() when the cap is hit.
            -- audited: 0=pending, 1=done (safe to delete).
            CREATE TABLE IF NOT EXISTS raw_sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                raw_text   TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                audited    INTEGER NOT NULL DEFAULT 0
            );
        ",
        )?;

        // Schema migrations for existing DBs.
        // SQLite returns "duplicate column name" (error 1) for already-added
        // columns — silently ignored to keep this idempotent.
        let migrations = [
            "ALTER TABLE nodes ADD COLUMN evidence_count   INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE nodes ADD COLUMN last_recalled_at TEXT    NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN embedding_model  TEXT    NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN hidden           INTEGER NOT NULL DEFAULT 0",
        ];
        for sql in &migrations {
            let _ = self.conn.execute(sql, []);
        }

        // Seed gradient weights on first-ever run.
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM hw_grad", [], |r| r.get(0))?;
        if count == 0 {
            for (k, v) in &[
                ("w_ram", 0.0f64),
                ("w_cpu", 0.0),
                ("w_temp", 0.0),
                ("w_gen", 0.0),
                ("bias", -2.0),
            ] {
                self.conn.execute(
                    "INSERT OR IGNORE INTO hw_grad (k, v) VALUES (?1, ?2)",
                    params![k, v],
                )?;
            }
        }

        Ok(())
    }

    fn migrate_legacy(&self) -> anyhow::Result<()> {
        let project_dir = config::project_data_dir();
        let diary_path = project_dir.join("skill_diary.json");
        if diary_path.exists() {
            let count: i64 = self
                .conn
                .query_row("SELECT COUNT(*) FROM diary", [], |r| r.get(0))?;
            if count == 0 {
                if let Ok(text) = std::fs::read_to_string(&diary_path) {
                    for line in text.lines() {
                        let entry = line.trim_start_matches("- ");
                        if !entry.is_empty() {
                            self.add_diary_entry(entry)?;
                        }
                    }
                    println!("   ✅ Migrated skill_diary.json → SQLite");
                }
            }
        }
        let graph_path = project_dir.join("memory_graph.json");
        if graph_path.exists() {
            let puzzle_count: i64 =
                self.conn
                    .query_row("SELECT COUNT(*) FROM puzzles", [], |r| r.get(0))?;
            if puzzle_count == 0 {
                if let Ok(text) = std::fs::read_to_string(graph_path) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(puzzles) = v["puzzles"].as_array() {
                            for p in puzzles {
                                let q = p["question"].as_str().unwrap_or("").to_string();
                                let s = p["status"].as_str().unwrap_or("open").to_string();
                                let a = p["attempts"].as_i64().unwrap_or(0) as i32;
                                if !q.is_empty() {
                                    self.conn.execute(
                                        "INSERT OR IGNORE INTO puzzles (id,question,status,attempts,created_at) VALUES (?1,?2,?3,?4,?5)",
                                        params![Uuid::new_v4().to_string(), q, s, a, Utc::now().to_rfc3339()],
                                    )?;
                                }
                            }
                            println!("   ✅ Migrated memory_graph.json → SQLite");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn add_diary_entry(&self, entry: &str) -> anyhow::Result<()> {
        let clean = sanitize(entry);
        self.conn.execute(
            "INSERT INTO diary (entry, created_at) VALUES (?1, ?2)",
            params![clean, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn get_diary(&self, limit: usize) -> anyhow::Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT entry FROM diary ORDER BY id DESC LIMIT ?1")?;
        let rows = stmt.query_map(params![limit as i64], |r| r.get(0))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Update the label and/or content of an existing node by id.
    /// Pass empty string to leave a field unchanged.
    /// Also refreshes the FTS5 mirror so searches stay consistent.
    pub fn update_node(&self, id: &str, new_label: &str, new_content: &str) -> anyhow::Result<()> {
        // Fetch current values so we only overwrite what was provided
        let (cur_label, cur_content): (String, String) = self
            .conn
            .query_row(
                "SELECT label, content FROM nodes WHERE id = ?1 AND hidden = 0",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(|_| anyhow::anyhow!("node not found: {}", id))?;

        let label = if new_label.is_empty() {
            cur_label.clone()
        } else {
            sanitize(new_label)
        };
        let content = if new_content.is_empty() {
            cur_content.clone()
        } else {
            sanitize(new_content)
        };

        let embedding_text = format!("{label}: {content}");
        let replacement_embedding = embed_text(&embedding_text);
        let (embedding_model, embedding_blob) = match replacement_embedding {
            Some((model, vector)) => (model, Some(encode_embedding(vector))),
            None => (String::new(), None),
        };
        self.conn.execute(
            "UPDATE nodes SET label = ?1, content = ?2, embedding = ?3, embedding_model = ?4 WHERE id = ?5",
            params![label, content, embedding_blob, embedding_model, id],
        )?;

        // Refresh FTS mirror
        self.conn
            .execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![id])?;
        self.conn.execute(
            "INSERT INTO nodes_fts (node_id, label, content) VALUES (?1, ?2, ?3)",
            params![id, label, content],
        )?;
        self.conn.execute(
            "DELETE FROM edges WHERE relationship = 'RELATES_TO' AND (source = ?1 OR target = ?1)",
            params![id],
        )?;
        if let Some(vector) = embedding_blob.as_deref() {
            self.link_similar_nodes(&id, &embedding_model, vector)?;
        }
        Ok(())
    }

    /// Soft-delete a node: sets hidden=1 so it disappears from search and
    /// context injection but the graph stays structurally intact.
    /// Prefer this over delete_node for most LLM-directed forgetting.
    pub fn hide_node(&self, id: &str) -> anyhow::Result<()> {
        let rows = self
            .conn
            .execute("UPDATE nodes SET hidden = 1 WHERE id = ?1", params![id])?;
        if rows == 0 {
            anyhow::bail!("node not found: {}", id);
        }
        // Remove from FTS so it won't surface in keyword searches either
        self.conn
            .execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![id])?;
        Ok(())
    }

    /// Hard-delete a node and all its edges. Irreversible.
    /// Use hide_node instead unless the data must be permanently removed.
    pub fn delete_node(&self, id: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "DELETE FROM edges WHERE source = ?1 OR target = ?1",
            params![id],
        )?;
        self.conn
            .execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![id])?;
        let rows = self
            .conn
            .execute("DELETE FROM nodes WHERE id = ?1", params![id])?;
        if rows == 0 {
            anyhow::bail!("node not found: {}", id);
        }
        Ok(())
    }

    /// Insert a new node. Sanitizes text, stores an active-model retrieval vector,
    /// mirrors the text into FTS5, and links sufficiently similar existing nodes.
    pub fn add_node(&self, label: &str, node_type: &str, content: &str) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();
        let clean_label = sanitize(label);
        let clean_content = sanitize(content);
        let embedding_text = format!("{}: {}", clean_label, clean_content);
        let (embedding_model, embedding_blob) = match embed_text(&embedding_text) {
            Some((model, vector)) => (model, Some(encode_embedding(vector))),
            None => (String::new(), None),
        };
        self.conn.execute(
            "INSERT OR IGNORE INTO nodes
             (id, label, node_type, content, strength, embedding, embedding_model, evidence_count, last_recalled_at, created_at)
             VALUES (?1, ?2, ?3, ?4, 1.0, ?5, ?6, 0, '', ?7)",
            params![id, clean_label, node_type, clean_content, embedding_blob, embedding_model, Utc::now().to_rfc3339()],
        )?;
        // Mirror into FTS5 only when a new row was actually inserted
        if self.conn.changes() > 0 {
            self.conn.execute(
                "INSERT INTO nodes_fts (node_id, label, content) VALUES (?1, ?2, ?3)",
                params![id, clean_label, clean_content],
            )?;
            if let Some(vector) = embedding_blob.as_deref() {
                self.link_similar_nodes(&id, &embedding_model, vector)?;
            }
        }
        Ok(id)
    }

    /// Link the new item to existing nodes using cosine similarity of native
    /// token embeddings. This is bounded by MAX_LIVE_NODES and runs only at ingest.
    fn link_similar_nodes(&self, new_id: &str, model: &str, new_blob: &[u8]) -> anyhow::Result<()> {
        let Some(new_vector) = decode_embedding(new_blob) else {
            return Ok(());
        };
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding FROM nodes WHERE id != ?1 AND embedding_model = ?2 AND hidden = 0 AND embedding IS NOT NULL LIMIT ?3"
        )?;
        let rows: Vec<(String, Vec<u8>)> = stmt
            .query_map(
                params![new_id, model, crate::config::MAX_LIVE_NODES as i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?
            .filter_map(|row| row.ok())
            .collect();
        let mut similar = rows
            .into_iter()
            .filter_map(|(other_id, blob)| {
                let other = decode_embedding(&blob)?;
                if new_vector.len() != other.len() {
                    return None;
                }
                let score = cosine_sim(&new_vector, &other);
                (score >= 0.55).then_some((other_id, score))
            })
            .collect::<Vec<_>>();
        similar.sort_by(|left, right| right.1.total_cmp(&left.1));
        for (other_id, score) in similar
            .into_iter()
            .take(crate::config::GRAPH_HOP_NEIGHBOURS)
        {
            self.conn.execute(
                "INSERT INTO edges (source,target,relationship,reason,strength,created_at)
                 VALUES (?1,?2,'RELATES_TO',?3,?4,?5)
                 ON CONFLICT(source,target) DO UPDATE SET strength=MIN(edges.strength+0.05,1.0)",
                params![
                    new_id,
                    other_id,
                    format!("native embedding similarity {:.2}", score),
                    score,
                    Utc::now().to_rfc3339()
                ],
            )?;
        }
        Ok(())
    }

    /// Rebuild a bounded batch of retrieval vectors for records created before
    /// an inference model was loaded or before the embedding column was used.
    pub fn backfill_missing_embeddings(&self, limit: usize) -> anyhow::Result<usize> {
        let Some(active_model) = crate::llm::current_embedding_model() else {
            return Ok(0);
        };
        let mut stmt = self.conn.prepare(
            "SELECT id,label,content FROM nodes
             WHERE hidden = 0 AND (embedding IS NULL OR embedding_model != ?1)
             ORDER BY created_at DESC LIMIT ?2",
        )?;
        let rows: Vec<(String, String, String)> = stmt
            .query_map(params![active_model, limit as i64], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(|row| row.ok())
            .collect();
        let mut updated = 0;
        for (id, label, content) in rows {
            let text = sanitize(&format!("{label}: {content}"));
            let Some((model, vector)) = embed_text(&text) else {
                break;
            };
            let blob = encode_embedding(vector);
            self.conn.execute(
                "UPDATE nodes SET embedding = ?1, embedding_model = ?2 WHERE id = ?3",
                params![blob, model, id],
            )?;
            if let Ok(vector) = self.conn.query_row(
                "SELECT embedding FROM nodes WHERE id = ?1",
                params![id],
                |row| row.get::<_, Vec<u8>>(0),
            ) {
                let model: String = self.conn.query_row(
                    "SELECT embedding_model FROM nodes WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )?;
                self.link_similar_nodes(&id, &model, &vector)?;
            }
            updated += 1;
        }
        Ok(updated)
    }

    /// Add or reinforce a typed directed edge between two nodes.
    /// Edge types: CAUSED_BY | RESOLVED_BY | PREFERS | RELATES_TO
    #[allow(dead_code)]
    pub fn add_edge(
        &self,
        source: &str,
        target: &str,
        relationship: &str,
        reason: &str,
    ) -> anyhow::Result<()> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM edges WHERE source=?1 AND target=?2",
                params![source, target],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if exists {
            self.conn.execute(
                "UPDATE edges SET strength=MIN(strength+0.1,1.0) WHERE source=?1 AND target=?2",
                params![source, target],
            )?;
        } else {
            self.conn.execute(
                "INSERT INTO edges (source,target,relationship,reason,strength,created_at) VALUES (?1,?2,?3,?4,0.5,?5)",
                params![source, target, relationship, sanitize(reason), Utc::now().to_rfc3339()],
            )?;
        }
        Ok(())
    }

    /// Search nodes using FTS5 BM25 ranking with a LIKE fallback for short
    /// queries. Bumps evidence_count + last_recalled_at on every hit so the
    /// dual-decay formula keeps frequently recalled nodes alive longer.
    pub fn search_nodes(
        &self,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<(MemoryNode, f32)>> {
        let query_embedding = embed_text(&sanitize(query));
        let mut scored: Vec<(MemoryNode, f32)> = Vec::new();

        // Compare like-model vectors only. FTS5 remains the safe fallback if no
        // model is loaded or the active profile differs from historical vectors.
        if let Some((embedding_model, query_vec)) = query_embedding {
            let mut stmt = self.conn.prepare(
                "SELECT id,label,node_type,content,strength,embedding,created_at
                 FROM nodes WHERE embedding_model = ?1 AND embedding IS NOT NULL AND hidden = 0",
            )?;
            let vec_scored: Vec<(MemoryNode, f32)> = stmt
                .query_map([&embedding_model], |r| {
                    Ok((
                        MemoryNode {
                            id: r.get(0)?,
                            label: r.get(1)?,
                            node_type: r.get(2)?,
                            content: r.get(3)?,
                            strength: r.get(4)?,
                            created_at: r.get(6)?,
                        },
                        r.get::<_, Option<Vec<u8>>>(5)?,
                    ))
                })?
                .filter_map(|r| r.ok())
                .filter_map(|(node, blob)| {
                    let vec = decode_embedding(blob.as_deref()?)?;
                    (vec.len() == query_vec.len()).then(|| (node, cosine_sim(&query_vec, &vec)))
                })
                .filter(|(_, score)| *score > 0.18)
                .collect();
            scored.extend(vec_scored);
        }

        // FTS5 BM25 ranked search — replaces the old LIKE scan.
        // Sanitise: wrap each word as a quoted phrase so FTS5 operators / special
        // chars (AND, OR, NOT, NEAR, *, (, )) can't break the query syntax.
        if scored.is_empty() {
            let fts_query = query
                .split_whitespace()
                .map(|w| format!("\"{}\"", w.replace('"', "")))
                .collect::<Vec<_>>()
                .join(" OR ");
            let sql = "
                SELECT n.id, n.label, n.node_type, n.content, n.strength, n.created_at,
                       -bm25(nodes_fts) AS score
                FROM nodes_fts
                JOIN nodes n ON n.id = nodes_fts.node_id
                WHERE nodes_fts MATCH ?1
                  AND n.hidden = 0
                ORDER BY score DESC
                LIMIT ?2";
            if let Ok(mut stmt) = self.conn.prepare(sql) {
                if let Ok(rows) = stmt.query_map(params![fts_query, top_k as i64], |r| {
                    Ok((
                        MemoryNode {
                            id: r.get(0)?,
                            label: r.get(1)?,
                            node_type: r.get(2)?,
                            content: r.get(3)?,
                            strength: r.get(4)?,
                            created_at: r.get(5)?,
                        },
                        r.get::<_, f64>(6)? as f32,
                    ))
                }) {
                    let fts_hits: Vec<(MemoryNode, f32)> = rows.filter_map(|r| r.ok()).collect();
                    scored.extend(fts_hits);
                }
            }
        }

        // LIKE fallback for queries too short for FTS5 tokenization
        if scored.is_empty() {
            let keywords: Vec<&str> = query.split_whitespace().filter(|w| w.len() >= 3).collect();
            let search_terms = if keywords.is_empty() {
                vec![query]
            } else {
                keywords
            };
            for kw in search_terms {
                if let Ok(mut stmt) = self.conn.prepare(
                    "SELECT id,label,node_type,content,strength,created_at FROM nodes
                     WHERE (label LIKE '%'||?1||'%' OR content LIKE '%'||?1||'%')
                       AND hidden = 0
                     ORDER BY strength DESC LIMIT ?2",
                ) {
                    let matches: Vec<(MemoryNode, f32)> = stmt
                        .query_map(params![kw, top_k as i64], |r| {
                            Ok((
                                MemoryNode {
                                    id: r.get(0)?,
                                    label: r.get(1)?,
                                    node_type: r.get(2)?,
                                    content: r.get(3)?,
                                    strength: r.get(4)?,
                                    created_at: r.get(5)?,
                                },
                                0.8_f32,
                            ))
                        })?
                        .filter_map(|r| r.ok())
                        .collect();
                    for (node, score) in matches {
                        if !scored.iter().any(|(n, _)| n.id == node.id) {
                            scored.push((node, score));
                        }
                    }
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Reinforce every recalled node: bump evidence_count, stamp last_recalled_at.
        let now = Utc::now().to_rfc3339();
        for (node, _) in &scored {
            let _ = self.conn.execute(
                "UPDATE nodes SET evidence_count = evidence_count + 1, last_recalled_at = ?1 WHERE id = ?2",
                params![now, node.id],
            );
        }

        Ok(scored)
    }

    /// Build context using FTS5 seed search + 1-hop typed edge expansion.
    fn archive_path() -> Option<String> {
        config::memory_archive_paths().into_iter().find_map(|path| {
            path.parent()
                .filter(|parent| parent.exists())
                .map(|_| path.to_string_lossy().into_owned())
        })
    }
    /// Keep RAM lean: if nodes+sessions exceed 500+200 rows, move oldest sessions to SSD archive.
    /// Called after each store_session and in the 24h reflection cycle.
    pub fn archive_old_sessions(&self) -> anyhow::Result<usize> {
        use crate::config::MEMORY_ARCHIVE_MAX_MB;
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        if (count as usize) <= 300 {
            return Ok(0);
        } // no pressure yet
        let arch = match Self::archive_path() {
            Some(p) => p,
            None => return Ok(0),
        };
        // Lazy create archive DB with same sessions schema
        if !std::path::Path::new(&arch).exists() {
            if let Ok(ac) = Connection::open(&arch) {
                let _= ac.execute_batch("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, topic TEXT, summary TEXT, emotion TEXT, embedding BLOB, created_at TEXT); CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, label TEXT, node_type TEXT, content TEXT, strength REAL, created_at TEXT);");
            }
        }
        // Size guard — skip if archive already large
        if let Ok(meta) = std::fs::metadata(&arch) {
            if meta.len() > MEMORY_ARCHIVE_MAX_MB * 1024 * 1024 {
                return Ok(0);
            }
        }
        let to_move: i64 = count - 200; // keep 200 most recent in hot DB
        let mut stmt = self.conn.prepare("SELECT id,topic,summary,emotion,created_at FROM sessions ORDER BY created_at ASC LIMIT ?1")?;
        let rows: Vec<(String, String, String, String, String)> = stmt
            .query_map(params![to_move], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        if rows.is_empty() {
            return Ok(0);
        }
        let arch_conn = Connection::open(&arch)?;
        for (id, topic, summary, emotion, created_at) in &rows {
            let _ = arch_conn.execute(
                "INSERT OR IGNORE INTO sessions VALUES (?1,?2,?3,?4,NULL,?5)",
                params![id, topic, summary, emotion, created_at],
            );
        }
        for (id, _, _, _, _) in &rows {
            let _ = self
                .conn
                .execute("DELETE FROM sessions WHERE id = ?1", params![id]);
        }
        Ok(rows.len())
    }

    /// Format: seed node bullet + indented neighbour lines → ~120 tokens max.
    /// Hard-capped to CONTEXT_MAX_CHARS so the LLM prompt never bloats.
    pub fn get_context(&self, query: &str) -> String {
        use crate::config::{CONTEXT_MAX_CHARS, GRAPH_HOP_NEIGHBOURS, GRAPH_MIN_EDGE_STRENGTH};

        let mut results = match self.search_nodes(query, 5) {
            Ok(r) => r,
            Err(_) => vec![],
        };

        // Recency fallback: keyword search whiffs when query shares no tokens with
        // stored sessions (e.g. generic greetings, context-free follow-ups).
        // Always pad to at least 3 entries with the most-recent session nodes so
        // cross-session recall works even without keyword overlap.
        if results.len() < 3 {
            let seen_ids: std::collections::HashSet<String> =
                results.iter().map(|(n, _)| n.id.clone()).collect();
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT id, label, node_type, content, strength, created_at
                 FROM nodes
                 WHERE node_type = 'session' AND hidden = 0
                 ORDER BY created_at DESC LIMIT 5",
            ) {
                let recent: Vec<(MemoryNode, f32)> = stmt
                    .query_map([], |r| {
                        Ok(MemoryNode {
                            id: r.get(0)?,
                            label: r.get(1)?,
                            node_type: r.get(2)?,
                            content: r.get(3)?,
                            strength: r.get(4)?,
                            created_at: r.get(5)?,
                        })
                    })
                    .into_iter()
                    .flatten()
                    .filter_map(|r| r.ok())
                    .filter(|n| !seen_ids.contains(&n.id))
                    .map(|n| (n, 0.0f32))
                    .collect();
                results.extend(recent);
            }
        }

        if results.is_empty() {
            let diary = self.get_diary(3).unwrap_or_default();
            if diary.is_empty() {
                return "No memories yet.".to_string();
            }
            return format!("Recent notes:\n{}", diary.join("\n"));
        }

        let mut lines = vec![];
        for (node, score) in &results {
            // For recency-padded entries score==0.0, omit the relevance tag
            if *score > 0.0 {
                lines.push(format!("• {} [relevance: {:.2}]", node.label, score));
            } else {
                lines.push(format!("• {} [recent]", node.label));
            }
            if !node.content.is_empty() {
                // Include a one-line snippet of the session summary
                let snippet: String = node.content.chars().take(120).collect();
                lines.push(format!("  {}", snippet));
            }
            // 1-hop expansion: follow edges above the minimum strength threshold
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT e.relationship, e.reason, e.source, e.target
                 FROM edges e
                 WHERE (e.source = ?1 OR e.target = ?1) AND e.strength >= ?2
                 ORDER BY e.strength DESC LIMIT ?3",
            ) {
                let edges: Vec<(String, String, String, String)> = stmt
                    .query_map(
                        params![
                            node.id,
                            GRAPH_MIN_EDGE_STRENGTH,
                            GRAPH_HOP_NEIGHBOURS as i64
                        ],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                    )
                    .into_iter()
                    .flatten()
                    .filter_map(|e| e.ok())
                    .collect();
                for (rel, reason, src, tgt) in edges {
                    let other_id = if src == node.id { &tgt } else { &src };
                    if let Ok(label) = self.conn.query_row(
                        "SELECT label FROM nodes WHERE id = ?1",
                        params![other_id],
                        |r| r.get::<_, String>(0),
                    ) {
                        lines.push(format!("  -> {} {}", rel, label));
                        if !reason.is_empty() {
                            lines.push(format!("     ({})", reason));
                        }
                    }
                }
            }
        }
        let out = lines.join("\n");
        if out.len() > CONTEXT_MAX_CHARS {
            out.chars().take(CONTEXT_MAX_CHARS).collect::<String>() + "\n…[truncated]"
        } else {
            out
        }
    }

    /// Compact vector/FTS recall with a small 1-hop graph expansion for prompts.
    pub fn route_context(&self, query: &str) -> String {
        const ROUTE_MAX: usize = 600;
        use crate::config::{GRAPH_HOP_NEIGHBOURS, GRAPH_MIN_EDGE_STRENGTH};

        let hits = self.search_nodes(query, 2).unwrap_or_default();
        let mut out = String::new();
        let mut seen = std::collections::HashSet::new();
        for (node, score) in hits {
            if !seen.insert(node.id.clone()) {
                continue;
            }
            let snippet: String = node.content.chars().take(180).collect();
            let line = format!("• {} [match {:.2}]: {}\n", node.label, score, snippet);
            if out.len() + line.len() > ROUTE_MAX {
                break;
            }
            out.push_str(&line);

            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT n.label, e.relationship, e.reason
                 FROM edges e JOIN nodes n ON n.id = CASE WHEN e.source = ?1 THEN e.target ELSE e.source END
                 WHERE (e.source = ?1 OR e.target = ?1) AND e.strength >= ?2 AND n.hidden = 0
                 ORDER BY e.strength DESC LIMIT ?3")
            {
                let neighbors: Vec<(String, String, String)> = stmt.query_map(
                    params![node.id, GRAPH_MIN_EDGE_STRENGTH, GRAPH_HOP_NEIGHBOURS as i64],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                ).map(|rows| rows.filter_map(|row| row.ok()).collect()).unwrap_or_default();
                for (label, relationship, reason) in neighbors {
                    let edge_line = format!("  -> {relationship} {label}{}\n", if reason.is_empty() { String::new() } else { format!(" ({reason})") });
                    if out.len() + edge_line.len() > ROUTE_MAX { break; }
                    out.push_str(&edge_line);
                }
            }
        }
        if !out.is_empty() {
            return out.trim_end().to_string();
        }

        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT label FROM nodes WHERE node_type = 'session' AND hidden = 0
             ORDER BY created_at DESC LIMIT 1",
        ) {
            if let Some(label) = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .into_iter()
                .flatten()
                .filter_map(|row| row.ok())
                .next()
            {
                return format!("• (recent) {label}");
            }
        }
        String::new()
    }

    // ── Raw session ring-buffer ───────────────────────────────────────────────

    /// Write the verbatim exchange alongside the compressed summary.
    /// `raw_text` is the full user+assistant history string for this turn.
    /// After reflection audits it, it is deleted. If the ring-buffer cap is
    /// hit before reflection runs, the oldest unaudited rows are evicted.
    /// Net storage: ≤ RAW_SESSION_KEEP × ~6 KB ≈ 300 KB max.
    fn store_raw_session(&self, session_id: &str, raw_text: &str) -> anyhow::Result<()> {
        use crate::config::RAW_SESSION_KEEP;
        let clean = sanitize(raw_text);
        self.conn.execute(
            "INSERT INTO raw_sessions (session_id, raw_text, created_at) VALUES (?1, ?2, ?3)",
            params![session_id, clean, Utc::now().to_rfc3339()],
        )?;
        // Enforce ring-buffer cap: delete oldest unaudited rows beyond the limit.
        // One DELETE, zero extra reads — O(1) cost.
        self.conn.execute(
            "DELETE FROM raw_sessions WHERE id IN (
                SELECT id FROM raw_sessions
                WHERE audited = 0
                ORDER BY id ASC
                LIMIT MAX(0, (SELECT COUNT(*) FROM raw_sessions WHERE audited = 0) - ?1)
            )",
            params![RAW_SESSION_KEEP as i64],
        )?;
        Ok(())
    }

    /// Called by reflection: returns up to `limit` unaudited raw sessions
    /// in insertion order (oldest first). Marks them audited atomically.
    pub fn pop_unaudited_raw_sessions(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, String)>> {
        // Fetch
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, raw_text FROM raw_sessions
             WHERE audited = 0
             ORDER BY id ASC LIMIT ?1",
        )?;
        let rows: Vec<(i64, String, String)> = stmt
            .query_map(params![limit as i64], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        if rows.is_empty() {
            return Ok(vec![]);
        }

        // Mark audited in one statement
        let ids: Vec<String> = rows.iter().map(|(id, _, _)| id.to_string()).collect();
        let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "UPDATE raw_sessions SET audited = 1 WHERE id IN ({})",
            placeholders
        );
        let mut upd = self.conn.prepare(&sql)?;
        // rusqlite params_from_iter for dynamic list
        upd.execute(rusqlite::params_from_iter(ids.iter()))?;

        // Delete audited rows — they've been processed, no reason to keep them
        self.conn
            .execute("DELETE FROM raw_sessions WHERE audited = 1", [])?;

        Ok(rows.into_iter().map(|(_, sid, txt)| (sid, txt)).collect())
    }

    /// Safety valve: delete all audited rows + any excess unaudited beyond cap.
    /// Called from the 24h reflection prune pass — zero cost when ring is healthy.
    pub fn prune_raw_sessions(&self) -> anyhow::Result<usize> {
        use crate::config::RAW_SESSION_KEEP;
        // Audited rows should already be gone, but clean up any stragglers
        let n1 = self
            .conn
            .execute("DELETE FROM raw_sessions WHERE audited = 1", [])?;
        // Enforce cap on unaudited rows
        let n2 = self.conn.execute(
            "DELETE FROM raw_sessions WHERE audited = 0 AND id NOT IN (
                SELECT id FROM raw_sessions WHERE audited = 0
                ORDER BY id DESC LIMIT ?1
            )",
            params![RAW_SESSION_KEEP as i64],
        )?;
        Ok(n1 + n2)
    }

    /// Store a session with both a compressed summary and the raw verbatim exchange.
    /// `raw_text` is the full user+assistant history for this turn.
    pub fn store_session(&self, topic: &str, summary: &str, emotion: &str) -> anyhow::Result<()> {
        let id = Uuid::new_v4().to_string();
        let clean_topic = sanitize(topic);
        let clean_summary = sanitize(summary);
        let embedding_blob: Option<Vec<u8>> = None;
        self.conn.execute(
            "INSERT INTO sessions (id,topic,summary,emotion,embedding,created_at) VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, clean_topic, clean_summary, emotion, embedding_blob, Utc::now().to_rfc3339()],
        )?;
        // Mirror as a "session" node so FTS + graph can find it
        self.add_node(&clean_topic, "session", &clean_summary)?;
        let _ = self.archive_old_sessions();
        Ok(())
    }

    /// Store session with raw verbatim exchange for reflection auditing.
    /// Prefer this over store_session() everywhere in main.rs.
    pub fn store_session_with_raw(
        &self,
        topic: &str,
        summary: &str,
        emotion: &str,
        raw_text: &str,
    ) -> anyhow::Result<()> {
        let id = Uuid::new_v4().to_string();
        let clean_topic = sanitize(topic);
        let clean_summary = sanitize(summary);
        let embedding_blob: Option<Vec<u8>> = None;
        self.conn.execute(
            "INSERT INTO sessions (id,topic,summary,emotion,embedding,created_at) VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, clean_topic, clean_summary, emotion, embedding_blob, Utc::now().to_rfc3339()],
        )?;
        self.add_node(&clean_topic, "session", &clean_summary)?;
        // Write raw text into ring-buffer — storage-bounded, deleted after reflection
        if !raw_text.trim().is_empty() {
            let _ = self.store_raw_session(&id, raw_text);
        }
        let _ = self.archive_old_sessions();
        Ok(())
    }

    #[allow(dead_code)]
    pub fn add_puzzle(&self, question: &str) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();
        self.conn.execute(
            "INSERT INTO puzzles (id,question,status,attempts,created_at) VALUES (?1,?2,'open',0,?3)",
            params![id, question, Utc::now().to_rfc3339()],
        )?;
        Ok(id)
    }

    pub fn get_open_puzzles(&self) -> anyhow::Result<Vec<Puzzle>> {
        let mut stmt = self.conn.prepare(
            "SELECT id,question,status,attempts,created_at FROM puzzles WHERE status='open' ORDER BY created_at ASC")?;
        let rows = stmt.query_map([], |r| {
            Ok(Puzzle {
                id: r.get(0)?,
                question: r.get(1)?,
                status: r.get(2)?,
                attempts: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn update_puzzle(&self, id: &str, status: &str, attempts: i32) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE puzzles SET status=?1,attempts=?2 WHERE id=?3",
            params![status, attempts, id],
        )?;
        Ok(())
    }

    /// Remove weak edges and orphaned low-strength nodes.
    /// Also cleans up the FTS5 mirror for deleted nodes.
    /// Returns (edges_pruned, nodes_remaining).
    pub fn prune(&self, min_strength: f32) -> anyhow::Result<(usize, usize)> {
        let edges_before: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        self.conn.execute(
            "DELETE FROM edges WHERE strength < ?1",
            params![min_strength],
        )?;
        let edges_after: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        // Clean FTS5 mirror for nodes about to be deleted
        self.conn.execute(
            "DELETE FROM nodes_fts WHERE node_id IN (
                SELECT id FROM nodes WHERE strength < ?1
                  AND node_type NOT IN ('diary','session')
                  AND id NOT IN (SELECT source FROM edges UNION SELECT target FROM edges))",
            params![min_strength],
        )?;
        self.conn.execute(
            "DELETE FROM nodes WHERE strength < ?1
             AND node_type NOT IN ('diary','session')
             AND id NOT IN (SELECT source FROM edges UNION SELECT target FROM edges)",
            params![min_strength],
        )?;
        let nodes_remaining: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
            .unwrap_or(0);
        Ok((
            (edges_before - edges_after) as usize,
            nodes_remaining as usize,
        ))
    }

    /// Enforce the MAX_LIVE_NODES cap by evicting the weakest disconnected
    /// leaf nodes. Leaf = no edges in or out. Called after decay_strengths.
    /// Returns the number of nodes evicted.
    pub fn compact_nodes(&self) -> anyhow::Result<usize> {
        use crate::config::MAX_LIVE_NODES;
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        let excess = (total as usize).saturating_sub(MAX_LIVE_NODES);
        if excess == 0 {
            return Ok(0);
        }

        let mut stmt = self.conn.prepare(
            "SELECT id FROM nodes
             WHERE id NOT IN (SELECT source FROM edges UNION SELECT target FROM edges)
               AND node_type NOT IN ('diary','session')
             ORDER BY strength ASC LIMIT ?1",
        )?;
        let ids: Vec<String> = stmt
            .query_map(params![excess as i64], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        let evicted = ids.len();
        for id in &ids {
            self.conn
                .execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![id])?;
            self.conn
                .execute("DELETE FROM nodes WHERE id = ?1", params![id])?;
        }
        Ok(evicted)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        let m = Self { conn };
        m.init_schema()?;
        Ok(m)
    }

    pub fn boost_recent_nodes(&self, keywords: &[&str], delta: f32) -> anyhow::Result<()> {
        for kw in keywords {
            self.conn.execute(
                "UPDATE nodes SET strength=MIN(strength+?1,2.0) WHERE label LIKE '%'||?2||'%'",
                params![delta, kw],
            )?;
        }
        Ok(())
    }

    // ── HW telemetry ──────────────────────────────────────────────────────────

    /// Insert one hardware snapshot. `generating` = true while llama-server
    /// is producing tokens. outcome is None until the session is labelled.
    pub fn insert_hw_sample(
        &self,
        ram_pct: f32,
        cpu_pct: f32,
        temp_c: f32,
        generating: bool,
    ) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO hw_samples (ts, ram_pct, cpu_pct, temp_c, generating)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                Utc::now().to_rfc3339(),
                ram_pct,
                cpu_pct,
                temp_c,
                generating as i32,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Label all still-open samples up to cutoff_id with outcome
    /// (0 = smooth, 1 = bad) and run one SGD step per sample.
    pub fn label_hw_samples(&self, cutoff_id: i64, outcome: i32) -> anyhow::Result<()> {
        use crate::config::{HW_GRAD_CLIP, HW_LR};
        let mut stmt = self.conn.prepare(
            "SELECT id, ram_pct, cpu_pct, temp_c, generating
             FROM hw_samples
             WHERE outcome IS NULL AND id <= ?1",
        )?;
        let rows: Vec<(i64, f32, f32, f32, i32)> = stmt
            .query_map(params![cutoff_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        for (id, ram, cpu, temp, gen) in rows {
            self.conn.execute(
                "UPDATE hw_samples SET outcome = ?1 WHERE id = ?2",
                params![outcome, id],
            )?;
            let x = [ram / 100.0, cpu / 100.0, temp / 100.0, gen as f32];
            let keys = ["w_ram", "w_cpu", "w_temp", "w_gen"];
            let mut weights: [f32; 4] = [0.0; 4];
            for (i, k) in keys.iter().enumerate() {
                weights[i] = self
                    .conn
                    .query_row("SELECT v FROM hw_grad WHERE k = ?1", params![k], |r| {
                        r.get(0)
                    })
                    .unwrap_or(0.0);
            }
            let bias: f32 = self
                .conn
                .query_row("SELECT v FROM hw_grad WHERE k = 'bias'", [], |r| r.get(0))
                .unwrap_or(-2.0);
            let logit: f32 = weights
                .iter()
                .zip(x.iter())
                .map(|(w, xi)| w * xi)
                .sum::<f32>()
                + bias;
            let p_bad = 1.0 / (1.0 + (-logit).exp());
            let grad = outcome as f32 - p_bad;
            for (i, k) in keys.iter().enumerate() {
                let new_w = (weights[i] + HW_LR * grad * x[i]).clamp(-HW_GRAD_CLIP, HW_GRAD_CLIP);
                self.conn
                    .execute("UPDATE hw_grad SET v = ?1 WHERE k = ?2", params![new_w, k])?;
            }
            let new_b = (bias + HW_LR * grad).clamp(-HW_GRAD_CLIP, HW_GRAD_CLIP);
            self.conn
                .execute("UPDATE hw_grad SET v = ?1 WHERE k = 'bias'", params![new_b])?;
        }
        Ok(())
    }

    /// Sigmoid risk score [0,1] for the current HW snapshot.
    /// Reads 5 floats from hw_grad — pure arithmetic, no heap allocations.
    pub fn hw_risk(&self, ram_pct: f32, cpu_pct: f32, temp_c: f32, generating: bool) -> f32 {
        let x = [
            ram_pct / 100.0,
            cpu_pct / 100.0,
            temp_c / 100.0,
            generating as u8 as f32,
        ];
        let keys = ["w_ram", "w_cpu", "w_temp", "w_gen"];
        let logit: f32 = keys
            .iter()
            .zip(x.iter())
            .map(|(k, xi)| {
                let w: f32 = self
                    .conn
                    .query_row("SELECT v FROM hw_grad WHERE k = ?1", params![k], |r| {
                        r.get(0)
                    })
                    .unwrap_or(0.0);
                w * xi
            })
            .sum::<f32>();
        let bias: f32 = self
            .conn
            .query_row("SELECT v FROM hw_grad WHERE k = 'bias'", [], |r| r.get(0))
            .unwrap_or(-2.0);
        1.0 / (1.0 + (-(logit + bias)).exp())
    }

    /// Delete hw_samples older than HW_SAMPLE_KEEP_HOURS.
    /// Called from the 24h reflection cycle — no extra thread.
    pub fn prune_hw_samples(&self) -> anyhow::Result<usize> {
        use crate::config::HW_SAMPLE_KEEP_HOURS;
        let cutoff = (Utc::now() - chrono::Duration::hours(HW_SAMPLE_KEEP_HOURS)).to_rfc3339();
        let n = self
            .conn
            .execute("DELETE FROM hw_samples WHERE ts < ?1", params![cutoff])?;
        Ok(n)
    }

    // ── Boot sentinel (crash detection) ──────────────────────────────────────

    /// Call once at startup. Missing or clean=0 sentinel → previous crash.
    /// Returns true if a crash was detected. Resets the sentinel for this run.
    pub fn mark_crash_if_unclean(&self) -> anyhow::Result<bool> {
        let clean: i32 = self
            .conn
            .query_row("SELECT clean FROM boot_sentinel WHERE id = 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(0);
        self.conn.execute(
            "INSERT INTO boot_sentinel (id, clean) VALUES (1, 0)
             ON CONFLICT(id) DO UPDATE SET clean = 0",
            [],
        )?;
        Ok(clean == 0)
    }

    /// Write clean=1 sentinel. Call just before process exit.
    pub fn mark_clean_exit(&self) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO boot_sentinel (id, clean) VALUES (1, 1)
             ON CONFLICT(id) DO UPDATE SET clean = 1",
            [],
        )?;
        Ok(())
    }

    // ── Node strength decay ───────────────────────────────────────────────────

    /// Dual-speed decay:
    ///   • Episodic session nodes decay at STRENGTH_DECAY^SESSION_DECAY_MULTIPLIER
    ///   • Semantic nodes decay at STRENGTH_DECAY^(1/(1+evidence_count))
    ///     — the more often a node is recalled, the slower it fades.
    pub fn decay_strengths(&self) -> anyhow::Result<()> {
        use crate::config::{SESSION_DECAY_MULTIPLIER, STRENGTH_DECAY};

        // Session nodes: fast decay
        let session_factor = STRENGTH_DECAY.powf(SESSION_DECAY_MULTIPLIER);
        self.conn.execute(
            "UPDATE nodes SET strength = MAX(strength * ?1, 0.01) WHERE node_type = 'session'",
            params![session_factor],
        )?;

        // Semantic nodes: evidence-modulated decay.
        // Pre-compute lookup table for buckets [0..20] since SQLite lacks pow()
        for ec in 0..=20 {
            let factor = STRENGTH_DECAY.powf(1.0 / (1.0 + ec as f32));
            self.conn.execute(
                "UPDATE nodes SET strength = MAX(strength * ?1, 0.01)
                 WHERE node_type != 'session' AND evidence_count = ?2",
                params![factor, ec as i64],
            )?;
        }
        let max_factor = STRENGTH_DECAY.powf(1.0 / 21.0_f32);
        self.conn.execute(
            "UPDATE nodes SET strength = MAX(strength * ?1, 0.01)
             WHERE node_type != 'session' AND evidence_count > 20",
            params![max_factor],
        )?;
        Ok(())
    }

    // ── HW diary summary ─────────────────────────────────────────────────────

    /// One-line summary of current gradient weights for the diary.
    /// Returns None when weights are still near zero (nothing learned yet).
    pub fn summarize_hw_lessons(&self) -> anyhow::Result<Option<String>> {
        let keys = ["w_ram", "w_cpu", "w_temp", "w_gen"];
        let mut vals = [0.0f32; 4];
        for (i, k) in keys.iter().enumerate() {
            vals[i] = self
                .conn
                .query_row("SELECT v FROM hw_grad WHERE k = ?1", params![k], |r| {
                    r.get(0)
                })
                .unwrap_or(0.0);
        }
        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs < 0.15 {
            return Ok(None);
        }
        let names = ["RAM%", "CPU%", "temp°C", "generating"];
        let dominant = names
            .iter()
            .zip(vals.iter())
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(name, w)| format!("{} (w={:.2})", name, w))
            .unwrap_or_default();
        let p_ex = self.hw_risk(80.0, 70.0, 85.0, true);
        Ok(Some(format!(
            "[hw-grad] top predictor: {}; p_bad@RAM80/CPU70/temp85/gen={:.2}",
            dominant, p_ex
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_schema_and_puzzles() {
        let mem = Memory::open_in_memory().unwrap();
        let pid = mem.add_puzzle("How does Rust prevent data races?").unwrap();
        let puzzles = mem.get_open_puzzles().unwrap();
        assert_eq!(puzzles.len(), 1);
        assert_eq!(puzzles[0].question, "How does Rust prevent data races?");

        mem.update_puzzle(&pid, "solved", 1).unwrap();
        let open_puzzles = mem.get_open_puzzles().unwrap();
        assert_eq!(open_puzzles.len(), 0);
    }

    #[test]
    fn test_synaptic_pruning() {
        let mem = Memory::open_in_memory().unwrap();
        mem.add_node("Obsolete Topic", "concept", "something old")
            .unwrap();
        // Lower its strength
        mem.conn
            .execute(
                "UPDATE nodes SET strength = 0.05 WHERE label = 'Obsolete Topic'",
                [],
            )
            .unwrap();
        let (pruned_edges, remaining_nodes) = mem.prune(0.2).unwrap();
        assert_eq!(pruned_edges, 0);
        assert_eq!(remaining_nodes, 0);
    }

    // ── Pillar 1: ingestion sinkhole ─────────────────────────────────────────

    #[test]
    fn test_sanitize_redacts_secrets() {
        let s = sanitize("Here is my key: sk-abcdefghijklmnopqrstuvwxyzABCDEFGH1234");
        assert!(
            s.contains("[REDACTED_TOKEN]"),
            "sk- token not redacted: {}",
            s
        );
        assert!(!s.contains("sk-abc"), "raw token still present");

        let s2 = sanitize("token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ab");
        assert!(
            s2.contains("[REDACTED_TOKEN]"),
            "gh_ token not redacted: {}",
            s2
        );

        let s3 = sanitize("key=AKIAIOSFODNN7EXAMPLE");
        assert!(
            s3.contains("[REDACTED_KEY]"),
            "AWS key not redacted: {}",
            s3
        );

        let s4 = sanitize("Contact me at user@example.com");
        assert!(
            s4.contains("[REDACTED_EMAIL]"),
            "email not redacted: {}",
            s4
        );

        let clean = "Rust uses ownership to prevent data races.";
        assert_eq!(sanitize(clean), clean);
    }

    #[test]
    fn test_sanitize_hooks_into_add_node() {
        let mem = Memory::open_in_memory().unwrap();
        mem.add_node(
            "API experiment",
            "concept",
            "Used sk-secretkeyABCDEFGHIJKLMNOPQRSTUVWXYZ12345678 to call OpenAI",
        )
        .unwrap();
        let results = mem.search_nodes("API experiment", 5).unwrap();
        assert!(!results.is_empty());
        let content = &results[0].0.content;
        assert!(
            content.contains("[REDACTED_TOKEN]"),
            "secret leaked into DB: {}",
            content
        );
        assert!(!content.contains("sk-secret"), "raw token found in DB");
    }

    // ── Pillar 2: FTS5 + 1-hop graph context ─────────────────────────────────

    #[test]
    fn test_embedding_codec_and_automatic_graph_link() {
        let mem = Memory::open_in_memory().unwrap();
        let root = mem
            .add_node(
                "GPU crash notes",
                "incident",
                "driver reset at high memory pressure",
            )
            .unwrap();
        let related = mem
            .add_node(
                "CUDA memory pressure",
                "concept",
                "GPU driver ran out of memory",
            )
            .unwrap();
        let first = vec![1.0f32, 0.0, 0.0];
        let second = vec![0.9f32, 0.4358899, 0.0];
        assert_eq!(
            decode_embedding(&encode_embedding(first.clone())).unwrap(),
            first
        );
        mem.conn
            .execute(
                "UPDATE nodes SET embedding=?1, embedding_model='test-model' WHERE id=?2",
                params![encode_embedding(second), related],
            )
            .unwrap();
        mem.conn
            .execute(
                "UPDATE nodes SET embedding_model='test-model' WHERE id=?1",
                params![root],
            )
            .unwrap();
        mem.link_similar_nodes(&root, "test-model", &encode_embedding(first))
            .unwrap();
        let edge_count: i64 = mem.conn.query_row("SELECT COUNT(*) FROM edges WHERE source=?1 AND target=?2 AND relationship='RELATES_TO'", params![root, related], |row| row.get(0)).unwrap();
        assert_eq!(edge_count, 1);
        let context = mem.get_context("GPU crash notes");
        assert!(
            context.contains("RELATES_TO"),
            "auto-created relation missing from context: {context}"
        );
    }

    #[test]
    fn test_fts5_search_and_evidence_bump() {
        let mem = Memory::open_in_memory().unwrap();
        let id_a = mem
            .add_node(
                "Rust ownership model",
                "concept",
                "Ownership prevents use-after-free and double-free bugs.",
            )
            .unwrap();
        let id_b = mem
            .add_node(
                "Memory safety without GC",
                "concept",
                "Rust achieves memory safety through the borrow checker.",
            )
            .unwrap();
        mem.add_node(
            "Unrelated topic about cooking",
            "concept",
            "Pasta is best al dente.",
        )
        .unwrap();

        let hits = mem.search_nodes("rust memory safety", 5).unwrap();
        assert!(!hits.is_empty(), "FTS5 returned no results");
        let ids: Vec<&str> = hits.iter().map(|(n, _)| n.id.as_str()).collect();
        assert!(
            ids.contains(&id_a.as_str()) || ids.contains(&id_b.as_str()),
            "Expected rust nodes in results, got: {:?}",
            ids
        );

        for (node, _) in &hits {
            let ec: i64 = mem
                .conn
                .query_row(
                    "SELECT evidence_count FROM nodes WHERE id = ?1",
                    params![node.id],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(ec > 0, "evidence_count not bumped for node {}", node.label);
        }
    }

    #[test]
    fn test_1hop_context_expansion() {
        let mem = Memory::open_in_memory().unwrap();
        let root = mem
            .add_node("Server crash", "incident", "llama-server OOM at 92% RAM")
            .unwrap();
        let cause = mem
            .add_node(
                "High batch size",
                "concept",
                "batch=512 caused excessive VRAM pressure",
            )
            .unwrap();
        let fix = mem
            .add_node(
                "Reduce batch size",
                "tool_pattern",
                "Set batch=64 to stay under 75% RAM",
            )
            .unwrap();

        mem.add_edge(&root, &cause, "CAUSED_BY", "batch too large")
            .unwrap();
        mem.add_edge(&root, &fix, "RESOLVED_BY", "lower batch fixed OOM")
            .unwrap();

        let ctx = mem.get_context("server crash OOM");
        assert!(
            ctx.contains("CAUSED_BY") || ctx.contains("RESOLVED_BY"),
            "1-hop edges missing from context: {}",
            ctx
        );
    }

    // ── Pillar 3: dual-speed decay ────────────────────────────────────────────

    #[test]
    fn test_dual_speed_decay() {
        use crate::config::{SESSION_DECAY_MULTIPLIER, STRENGTH_DECAY};
        let mem = Memory::open_in_memory().unwrap();
        mem.add_node("Semantic concept", "concept", "some knowledge")
            .unwrap();
        mem.add_node("Session topic", "session", "what we discussed")
            .unwrap();

        mem.decay_strengths().unwrap();

        let sem: f32 = mem
            .conn
            .query_row(
                "SELECT strength FROM nodes WHERE node_type = 'concept'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let ses: f32 = mem
            .conn
            .query_row(
                "SELECT strength FROM nodes WHERE node_type = 'session'",
                [],
                |r| r.get(0),
            )
            .unwrap();

        assert!(
            ses < sem,
            "session ({:.4}) should decay faster than concept ({:.4})",
            ses,
            sem
        );
        let expected_ses = STRENGTH_DECAY.powf(SESSION_DECAY_MULTIPLIER);
        assert!(
            (ses - expected_ses).abs() < 0.001,
            "expected {:.4}, got {:.4}",
            expected_ses,
            ses
        );
    }

    #[test]
    fn test_compact_nodes_enforces_cap() {
        use crate::config::MAX_LIVE_NODES;
        let mem = Memory::open_in_memory().unwrap();
        for i in 0..(MAX_LIVE_NODES + 10) {
            let id = mem
                .add_node(&format!("Leaf {}", i), "concept", "ephemeral")
                .unwrap();
            mem.conn
                .execute(
                    "UPDATE nodes SET strength = 0.02 WHERE id = ?1",
                    params![id],
                )
                .unwrap();
        }
        let evicted = mem.compact_nodes().unwrap();
        assert!(evicted >= 10, "expected >= 10 evictions, got {}", evicted);
        let remaining: i64 = mem
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
            .unwrap();
        assert!(
            remaining as usize <= MAX_LIVE_NODES,
            "count {} exceeds cap {}",
            remaining,
            MAX_LIVE_NODES
        );
    }

    #[test]
    fn test_sentinel_clean_and_crash_lifecycle() {
        let mem = Memory::open_in_memory().unwrap();
        // Initially no sentinel row exists -> detected as unclean/new
        let was_unclean = mem.mark_crash_if_unclean().unwrap();
        assert!(
            was_unclean,
            "first boot should indicate sentinel was absent"
        );

        // Mark clean exit
        mem.mark_clean_exit().unwrap();

        // Next boot check should see clean exit
        let was_unclean_now = mem.mark_crash_if_unclean().unwrap();
        assert!(
            !was_unclean_now,
            "after clean exit, should not be marked as crash"
        );
    }

    #[test]
    fn test_sanitize_in_diary_and_sessions() {
        let mem = Memory::open_in_memory().unwrap();
        // Diary
        let bearer_entry =
            "Reflected on auth: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID";
        let sanitized_bearer = sanitize(bearer_entry);
        assert!(
            sanitized_bearer.contains("[REDACTED_TOKEN]"),
            "direct Bearer sanitization failed: {}",
            sanitized_bearer
        );
        mem.add_diary_entry(bearer_entry).unwrap();
        let entries = mem.get_diary(1).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0].contains("[REDACTED_TOKEN]"),
            "diary entry leaked Bearer token: {}",
            entries[0]
        );

        // Sessions
        mem.store_session(
            "Config topic",
            "User configured password=SuperSecretPassword123",
            "neutral",
        )
        .unwrap();
        let summary: String = mem
            .conn
            .query_row(
                "SELECT summary FROM sessions ORDER BY created_at DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            summary.contains("[REDACTED_SECRET]"),
            "session summary leaked password: {}",
            summary
        );
    }
}

#[cfg(test)]
mod route_tests {
    use super::*;

    #[test]
    fn test_route_context_returns_relevant_node() {
        let mem = Memory::open_in_memory().unwrap();
        mem.add_node(
            "Rust ownership rules",
            "fact",
            "Ownership: each value has one owner; when owner goes out of scope it is dropped.",
        )
        .unwrap();
        mem.add_node(
            "Python basics",
            "fact",
            "Python uses reference counting with a GC for cycles.",
        )
        .unwrap();

        let ctx = mem.route_context("rust ownership borrow");
        assert!(
            ctx.contains("Rust") || ctx.contains("ownership") || ctx.contains("owner"),
            "route_context missed relevant node: {}",
            ctx
        );
        assert!(
            ctx.len() <= 400,
            "route_context too long: {} chars",
            ctx.len()
        );
    }

    #[test]
    fn test_route_context_fallback_on_no_match() {
        let mem = Memory::open_in_memory().unwrap();
        mem.store_session("idle chat", "user said hello", "neutral")
            .unwrap();
        let ctx = mem.route_context("xyzzyqqqwwweee");
        assert!(
            ctx.is_empty() || ctx.contains("idle chat"),
            "unexpected fallback: {}",
            ctx
        );
    }

    #[test]
    fn test_needs_tools_detects_file_ops() {
        use crate::config::needs_tools;
        assert!(needs_tools("read the file at /home/user/notes.txt"));
        assert!(needs_tools("search the web for cats"));
        assert!(needs_tools("list directory contents"));
        assert!(!needs_tools("what is the capital of France"));
        assert!(!needs_tools("how are you today"));
    }
}
