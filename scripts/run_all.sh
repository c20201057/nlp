#!/bin/bash
# 在仓库根目录执行：bash scripts/run_all.sh
# 依次运行 Task 1、Task 2、Task 3，结果写入 results/，完整输出同时写入 logs/run_all.log。
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONNOUSERSITE=1          # 不读取 ~/.local 中的包，保证环境独立
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-$ROOT/env/bin/python}

mkdir -p logs
exec > >(tee logs/run_all.log) 2>&1

echo "start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "===== Task 1 $(date '+%T')"
"$PY" -u src/task1_bow.py
echo "===== Task 2 $(date '+%T')"
"$PY" -u src/task2_embeddings.py
echo "===== Task 3 $(date '+%T')"
"$PY" -u src/task3_bert.py --model models/bert-base-uncased
nvidia-smi --query-gpu=memory.used --format=csv

echo "end=$(date '+%F %T')"
