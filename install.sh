#!/usr/bin/env bash
# One-shot setup for creativity-steer. Run once: ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
warn() { printf "\033[1;33m!! %s\033[0m\n" "$1"; }

# --- node via nvm (the frontend needs npm) ---
ensure_node() {
  if command -v node >/dev/null 2>&1; then return; fi
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # shellcheck disable=SC1091
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm use default >/dev/null 2>&1 || nvm use node >/dev/null 2>&1 || true
}

say "1/5  Python environment (uv)"
command -v uv >/dev/null || { warn "uv not found — install from https://docs.astral.sh/uv/"; exit 1; }
uv sync --extra api --extra deberta --extra web

say "2/5  Pre-downloading the DeBERTa novelty model"
uv run python - <<'PY'
from creativity_steer.backends import MockBackend
from creativity_steer.entailment import make_entailment
make_entailment("deberta", MockBackend())   # triggers download + load
print("DeBERTa ready")
PY

say "3/5  Frontend dependencies (npm)"
ensure_node
command -v npm >/dev/null || { warn "node/npm not found (nvm?) — install Node 20+"; exit 1; }
( cd web && npm install --include=dev )

say "4/5  Embedding model (Ollama)"
if command -v ollama >/dev/null 2>&1; then
  ollama pull "${CS_EMBED_MODEL:-embeddinggemma}" || warn "could not pull embeddinggemma"
else
  warn "ollama not found — install from https://ollama.com (needed for embeddings)"
fi

say "5/5  Checks"
command -v unsloth >/dev/null 2>&1 || warn "unsloth not found — install Unsloth Studio: https://unsloth.ai/docs/new/studio/install"
[ -f .env ] || { cp .env.example .env; echo "Created .env"; }

printf "\n\033[1;32mDone. Start everything with:  ./start.sh\033[0m\n"
