use anyhow::Result;
use chrono::Utc;
use rusqlite::{params, Connection};

#[derive(Debug, Clone)]
pub struct PendingAction {
    pub id: String,
    pub tool_name: String,
    pub args: String,
    pub created_at: String,
    pub status: String,
}

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

    pub fn list_pending(&self) -> Result<Vec<PendingAction>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, tool_name, args, created_at, status FROM pending_actions WHERE status = 'pending' ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(PendingAction {
                id: row.get(0)?,
                tool_name: row.get(1)?,
                args: row.get(2)?,
                created_at: row.get(3)?,
                status: row.get(4)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn mark_done(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pending_actions SET status = 'done' WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn mark_rejected(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pending_actions SET status = 'rejected' WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn remove_done(&self) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM pending_actions WHERE status IN ('done', 'rejected')",
            [],
        )?)
    }
}
