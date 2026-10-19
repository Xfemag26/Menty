use crate::pipeline::{PipelineContext, Stage};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

pub struct ResourceSelectionStage {
    pub model_profiles: &'static [crate::config::ModelProfile],
}

#[async_trait]
impl Stage for ResourceSelectionStage {
    fn name(&self) -> &'static str {
        "resource_selection"
    }

    async fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let profile = self.model_profiles.first().unwrap();
        ctx.profile_name = Some(profile.name.to_string());
        ctx.metadata.insert("profile".into(), profile.name.to_string());
        Ok(())
    }
}

pub struct RetrievalStage {
    pub memory: Arc<tokio::sync::Mutex<crate::memory::Memory>>,
    pub max_nodes: usize,
    pub hops: usize,
}

#[async_trait]
impl Stage for RetrievalStage {
    fn name(&self) -> &'static str {
        "retrieval"
    }

    async fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let query = ctx.user_input.clone();
        let mem = self.memory.clone();

        let output = tokio::task::spawn_blocking(move || {
            let mem_guard = futures::executor::block_on(mem.lock());

            let routed = mem_guard.route_context(&query);
            if routed.is_empty() {
                mem_guard.get_context(&query)
            } else {
                routed
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("retrieval task failed: {e}"))?;

        ctx.retrieval_ctx = Some(output);
        Ok(())
    }
}

pub struct PromptBuildStage {
    pub persona_core: String,
    pub persona_tools: String,
    pub max_prompt_chars: usize,
}

#[async_trait]
impl Stage for PromptBuildStage {
    fn name(&self) -> &'static str {
        "prompt_build"
    }

    async fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let mut prompt = String::new();
        prompt.push_str(&self.persona_core);

        if let Some(context) = &ctx.retrieval_ctx {
            prompt.push_str("\n\nVerified Context:\n");
            prompt.push_str(context);
        }

        prompt.push_str("\n<|im_start|>user\n");
        prompt.push_str(&ctx.user_input);
        prompt.push_str("\n<|im_end|>\n<|im_start|>assistant\n");

        if prompt.len() > self.max_prompt_chars {
            prompt.truncate(self.max_prompt_chars);
        }

        ctx.prompt = Some(prompt);
        Ok(())
    }
}

pub struct InferenceStage {
    pub max_tokens: u32,
    pub chat_tx: crossbeam_channel::Sender<crate::ChatMessage>,
}

#[async_trait]
impl Stage for InferenceStage {
    fn name(&self) -> &'static str {
        "inference"
    }

    async fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let prompt = ctx.prompt.clone().unwrap_or_default();
        let tx = self.chat_tx.clone();

        let output = tokio::task::spawn_blocking(move || {
            let prompt_for_capture = prompt.clone();

            // We seed a blank assistant bubble so the UI can stream.
            let _ = tx.send(crate::ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                time: chrono::Local::now().format("%H:%M").to_string(),
            });

            let result = crate::llm::complete_stream(&prompt_for_capture, 128, |piece| {
                let _ = tx.send(crate::ChatMessage {
                    role: "delta".into(),
                    content: piece.to_string(),
                    time: String::new(),
                });
            });

            result.unwrap_or_else(|e| format!("Error: {e}"))
        })
        .await
        .map_err(|e| anyhow::anyhow!("inference task failed: {e}"))?;

        ctx.final_output = Some(output);
        Ok(())
    }
}

pub struct ToolLoopStage {
    pub memory: Arc<tokio::sync::Mutex<crate::memory::Memory>>,
}

#[async_trait]
impl Stage for ToolLoopStage {
    fn name(&self) -> &'static str {
        "tool_loop"
    }

    async fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let output = ctx.final_output.clone().unwrap_or_default();

        let tool_lines: Vec<String> = output
            .lines()
            .filter(|line| line.contains("[TOOL:"))
            .map(|line| line.trim().to_string())
            .collect();

        if tool_lines.is_empty() {
            return Ok(());
        }

        let mut round_ctx = String::new();

        for tool_str in &tool_lines {
            if let Some(result) = crate::tools::dispatch_with_memory(tool_str, &self.memory).await {
                round_ctx.push_str(&format!("\n{tool_str} -> {}", result.output));
                ctx.tool_outputs.push(result.output);
            }
        }

        ctx.metadata.insert("tool_round".into(), round_ctx);

        Ok(())
    }
}

pub struct PostprocessStage {
    pub memory: Arc<tokio::sync::Mutex<crate::memory::Memory>>,
}

#[async_trait]
impl Stage for PostprocessStage {
    fn name(&self) -> &'static str {
        "postprocess"
    }

    async fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let final_output = ctx.final_output.clone().unwrap_or_default();
        let mem = self.memory.clone();

        tokio::task::spawn_blocking(move || {
            let _guard = mem.blocking_lock();
            // keep this extremely small at first:
            // let _ = mem.store_session_with_raw(...);
        })
        .await
        .map_err(|e| anyhow::anyhow!("postprocess failed: {e}"))?;

        Ok(())
    }
}
