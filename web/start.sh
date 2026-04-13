#!/bin/bash
# ── Lionet Web Platform Startup Script ──
# Usage: bash start.sh
# This script starts both the FastAPI backend and the Vue frontend dev server.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "╔══════════════════════════════════════════╗"
echo "║   Lionet 因子挖掘平台 - 启动中...        ║"
echo "╚══════════════════════════════════════════╝"

# ── 1. Start FastAPI backend ──
echo ""
echo "🔧 启动 FastAPI 后端 (port 8000)..."
cd "$PROJECT_ROOT"
conda activate future 2>/dev/null || true
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "   后端 PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# ── 2. Start Vue frontend dev server ──
echo ""
echo "🎨 启动 Vue 前端 (port 5173)..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "   前端 PID: $FRONTEND_PID"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅ 启动完成!                                    ║"
echo "║  前端地址: http://localhost:5173                  ║"
echo "║  后端地址: http://localhost:8000                  ║"
echo "║  API文档:  http://localhost:8000/docs             ║"
echo "║                                                   ║"
echo "║  按 Ctrl+C 停止所有服务                            ║"
echo "╚══════════════════════════════════════════════════╝"

# Trap SIGINT to kill both processes
trap "echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for both
wait

