#!/bin/bash
# 在仓库根目录执行一次：bash scripts/setup_env.sh
# 在 env/ 下创建独立的 Python 3.10 环境，并把 bert-base-uncased 权重下载到 models/bert-base-uncased。
# 需要事先装好 micromamba 或 conda。无法直连 huggingface.co 时，可先执行 export HF_ENDPOINT=https://hf-mirror.com。
set -euo pipefail
export PYTHONNOUSERSITE=1  # 不读取 ~/.local 中的包，保证环境独立
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV=$ROOT/env
cd "$ROOT"

if [ ! -x "$ENV/bin/python" ]; then
  if command -v micromamba > /dev/null; then
    micromamba create -y -p "$ENV" -c conda-forge python=3.10
  else
    conda create -y -p "$ENV" -c conda-forge python=3.10
  fi
fi
PIP="$ENV/bin/python -m pip"
$PIP install -q -U pip
$PIP install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
$PIP install -q -r requirements.txt

"$ENV/bin/python" - "$ROOT/models/bert-base-uncased" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download("google-bert/bert-base-uncased", local_dir=sys.argv[1],
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "vocab.txt"])
print("model at", p)
PY
"$ENV/bin/python" -c "import torch, transformers, sklearn, gensim, nltk; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'gensim', gensim.__version__)"
echo SETUP_DONE
