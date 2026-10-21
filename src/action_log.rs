use anyhow::Result;
use chrono::Utc;
use rusqlite::{params, Connection};

pub struct ActionLog {
    conn: Connection,
}

impl ActionLog {
    pub fn open() -> Result<Self> {
        let dir = crate::config::project_data_dir();
        std::fs::create_dir_all(&dir)?;

        let path = dir.join("pending_actions.sqlite");
        let conn = Connection::open(path)?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_actions (
                id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                args TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            ",
        )?;

        Ok(Self { conn })
    }

    pub fn insert_pending(&self, tool_name: &str, args: &str) -> Result<String> {
        self.insert_with_status(tool_name, args, "pending")
    }

    pub fn insert_with_status(&self, tool_name: &str, args: &str, status: &str) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT INTO pending_actions (id, tool_name, args, created_at, status) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, tool_name, args, now, status],
        )?;

        Ok(id)
    }

    pub fn remove_done(&self) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM pending_actions WHERE status IN ('done', 'rejected')",
            [],
        )?)
    }
}

