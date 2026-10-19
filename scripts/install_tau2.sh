#!/usr/bin/env bash
# Install tau2-bench (Python 3.12+) from source at the pinned commit, with its data directory,
# plus this project's tau2 extras. tau2 reads its tasks and documents from the source checkout.
set -euo pipefail
COMMIT=96a3ef92329baf908ffe9dfe188cea34e8701d2a
DEST=${1:-tau2-bench}
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none https://github.com/sierra-research/tau2-bench "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
python -m pip install -e "$DEST" "rank-bm25>=0.2.2" "websockets>=13"
echo "tau2-bench installed from $DEST at ${COMMIT:0:8}"
