#!/bin/zsh
# Gomoku v24 MCTS Lab — 一条命令启动
cd "$(dirname "$0")"
PY=${PY:-/Users/kid/AlphaZero_Gomoku_15x15_v24/.venv/bin/python}
export LAB_PORT=${LAB_PORT:-5055}      # 不用 $PORT: 本机 shell 里 PORT 被别的服务占用
echo "→ http://127.0.0.1:$LAB_PORT"
exec "$PY" server.py
