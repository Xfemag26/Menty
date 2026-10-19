use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

#[async_trait]
pub trait Stage: Send + Sync {
    fn name(&self) -> &'static str;
    async fn run(&self, ctx: &mut PipelineContext) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct PipelineContext {
    pub id: String,
    pub user_input: String,
    pub profile_name: Option<String>,
    pub metadata: HashMap<String, String>,
    pub retrieval_ctx: Option<String>,
    pub prompt: Option<String>,
    pub final_output: Option<String>,
    pub tool_outputs: Vec<String>,
}

impl PipelineContext {
    pub fn new(id: impl Into<String>, user_input: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            user_input: user_input.into(),
            profile_name: None,
            metadata: HashMap::new(),
            retrieval_ctx: None,
            prompt: None,
            final_output: None,
            tool_outputs: Vec::new(),
        }
    }
}

pub struct Pipeline {
    stages: Vec<Arc<dyn Stage>>,
}

impl Pipeline {
    pub fn new(stages: Vec<Arc<dyn Stage>>) -> Self {
        Self { stages }
    }

    pub async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        for stage in &self.stages {
            stage
                .run(&mut ctx)
                .await
                .map_err(|e| anyhow::anyhow!("Stage {} failed: {e}", stage.name()))?;
        }
        Ok(ctx)
    }
}
