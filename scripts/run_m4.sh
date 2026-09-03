#!/usr/bin/env bash
# ==============================================================================
# PAINT-CODE-RL: ONE-CLICK APPLE SILICON M4 LAUNCHER
# ==============================================================================
set -e

echo "================================================================================"
echo "   🎨 PAINT-CODE-RL: APPLE SILICON M4 (MPS) HIGH-PERFORMANCE TRAINING"
echo "================================================================================"

# 1. macOS & MPS Environment Setup
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export ENV=mps

# 2. Check Node & Python
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required for Headless WebGL rendering. Install with: brew install node"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Install with: brew install python@3.11"
    exit 1
fi

# 3. Ensure renderer dependencies
if [ ! -d "renderer/node_modules" ]; then
    echo "📦 Installing renderer dependencies..."
    (cd renderer && npm install --silent)
fi

# 4. Start Renderer Daemon in background
echo "🚀 Starting Metal-accelerated WebGL Renderer daemon on port 3000..."
export RENDERER_SHUTDOWN_TOKEN=${RENDERER_SHUTDOWN_TOKEN:-$(head -c 16 /dev/urandom 2>/dev/null | xxd -p 2>/dev/null || echo "m4_session_token_12345")}
node renderer/server.js &
RENDERER_PID=$!
trap "kill $RENDERER_PID 2>/dev/null || true" EXIT

# Wait for renderer health
RENDERER_READY=0
sleep 2
for i in {1..15}; do
    if curl -s http://127.0.0.1:3000/health | grep -q '"status":"ok"'; then
        echo "✅ WebGL Renderer ready (Metal ANGLE Accelerated)"
        RENDERER_READY=1
        break
    fi
    sleep 1
done

if [ $RENDERER_READY -ne 1 ]; then
    echo "❌ Error: WebGL Renderer failed to start on port 3000."
    exit 1
fi

# 5. Run GRPO RL Training Loop
echo ""
echo "🔥 Starting GRPO Policy Training on Apple Silicon M4 GPU..."
python3 scripts/train_grpo.py --mode train --max-steps 200

echo ""
echo "🎉 GRPO Training Complete! Checkpoints saved to artifacts/checkpoints/"
