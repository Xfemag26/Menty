#!/bin/bash
# Momentum - Ghostwater Native Launcher
# Tuned for i7-8550U (15W TDP) — kills both llama-server instances before starting

PROJECT_DIR="/home/mehtculos-xfemag/AI_project"
cd "$PROJECT_DIR"

# Kill any stale llama-server instances (both chat + embed ports)
pkill -f "$PROJECT_DIR/llama-bin/llama-server" 2>/dev/null || true
sleep 1

# Verify the embed server is actually gone — it holds 500MB we need back
if pgrep -f "llama-server.*8766" > /dev/null 2>&1; then
    kill -9 $(pgrep -f "llama-server.*8766") 2>/dev/null || true
fi

export LD_LIBRARY_PATH="$PROJECT_DIR/llama-bin:${LD_LIBRARY_PATH}"

echo "🌊 Starting Ghostwater Momentum (low-power mode)..."
exec "$PROJECT_DIR/momentum/target/release/momentum"

