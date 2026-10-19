use anyhow::Result;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

pub fn init() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .or_else(|_| EnvFilter::try_new("info"))
                .unwrap(),
        )
        .with_target(false)
        .try_init();

    info!("telemetry initialized");
    Ok(())
}

pub fn log_stage_start(request_id: &str, stage: &str) {
    debug!(request_id, stage, "stage_start");
}

pub fn log_stage_done(request_id: &str, stage: &str, elapsed_ms: u128) {
    debug!(request_id, stage, elapsed_ms, "stage_done");
}

pub fn log_stage_error(request_id: &str, stage: &str, err: &str) {
    error!(request_id, stage, error = err, "stage_failed");
}

pub fn log_tool_dispatch(tool: &str, status: &str) {
    info!(tool, status, "tool_dispatch");
}

pub fn log_tool_failure(tool: &str, err: &str) {
    warn!(tool, error = err, "tool_failed");
}

pub fn log_request_latency(request_id: &str, ms: u128) {
    debug!(request_id, ms, "request_latency_ms");
}
