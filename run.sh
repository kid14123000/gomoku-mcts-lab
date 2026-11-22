#!/bin/zsh
# Gomoku v24 MCTS Lab — 一条命令启动
# 依赖: 一个装了 torch / flask / gym / boardgame2 的 python (能跑 v24 训练的那个 venv 就行)
cd "$(dirname "$0")"

# python 解释器: 环境变量 > 仓库内 .venv > 本机训练 venv > 系统 python3
if [ -z "$PY" ]; then
  for c in "./.venv/bin/python" "/Users/kid/AlphaZero_Gomoku_15x15_v24/.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
fi
PY=${PY:-python3}

export LAB_PORT=${LAB_PORT:-5055}      # 不用 $PORT: 本机 shell 里 PORT 被别的服务占用
echo "python: $PY"
echo "→ http://127.0.0.1:$LAB_PORT"
exec "$PY" server.py
