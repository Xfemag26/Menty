use crate::pipeline::{Pipeline, PipelineContext};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

#[derive(Clone, Copy, Debug)]
pub enum Priority {
    Interactive,
    Background,
}

pub struct Request {
    pub id: String,
    pub ctx: PipelineContext,
    pub priority: Priority,
    pub response: oneshot::Sender<Result<PipelineContext>>,
}

pub struct Scheduler {
    tx: mpsc::Sender<Request>,
}

impl Scheduler {
    pub fn start(pipeline: Arc<Pipeline>, capacity: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<Request>(capacity);

        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let pipeline = pipeline.clone();

                tokio::spawn(async move {
                    let timeout = match req.priority {
                        Priority::Interactive => std::time::Duration::from_secs(30),
                        Priority::Background => std::time::Duration::from_secs(120),
                    };

                    let result = tokio::time::timeout(timeout, pipeline.execute(req.ctx)).await;

                    let _ = req.response.send(match result {
                        Ok(Ok(ctx)) => Ok(ctx),
                        Ok(Err(err)) => Err(err),
                        Err(_) => Err(anyhow::anyhow!("pipeline timeout")),
                    });
                });
            }
        });

        Self { tx }
    }

    pub async fn submit(
        &self,
        ctx: PipelineContext,
        priority: Priority,
    ) -> Result<PipelineContext> {
        let (tx, rx) = oneshot::channel();

        let req = Request {
            id: Uuid::new_v4().to_string(),
            ctx,
            priority,
            response: tx,
        };

        self.tx
            .send(req)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        rx.await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    }
}
