#!/bin/bash
# 在仓库根目录执行：bash scripts/download_glove.sh
# 下载约 822MB 的 glove.6B.zip，只解压出 100 维向量 embeddings/glove.6B.100d.txt。
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/embeddings"
cd "$ROOT/embeddings"
[ -f glove.6B.100d.txt ] && { echo "glove.6B.100d.txt already exists"; exit 0; }
curl -L -C - -o glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip -o glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip
