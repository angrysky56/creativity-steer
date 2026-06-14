#!/usr/bin/env bash
# Start everything: Unsloth model server(s) + backend + frontend.
# Stop everything with Ctrl+C (or ./stop.sh).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p .run logs
: > .run/pids

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
ok()  { printf "\033[1;32m   %s\033[0m\n" "$1"; }
die() { printf "\033[1;31m!! %s\033[0m\n" "$1"; exit 1; }

cleanup() {
  printf "\n\033[1;33mShutting down…\033[0m\n"
  if [ -f .run/pids ]; then
    while read -r pid; do
      [ -n "$pid" ] || continue
      pkill -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    done < .run/pids
  fi
  rm -f .run/pids
  printf "\033[1;32mAll stopped.\033[0m\n"
}
trap cleanup EXIT INT TERM

track() { echo "$1" >> .run/pids; }
port_of() { echo "$1" | sed -E 's#.*://[^:/]+:([0-9]+).*#\1#'; }
CTX="${CS_CONTEXT:-8192}"   # small context => fits a 12GB GPU and faster

[ -f .env ] || cp .env.example .env
set -a; . ./.env; set +a
command -v unsloth >/dev/null 2>&1 || die "unsloth not found — install Unsloth Studio."

if ! command -v node >/dev/null 2>&1; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # shellcheck disable=SC1091
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm use default >/dev/null 2>&1 || nvm use node >/dev/null 2>&1 || true
fi

say "Embedding model (Ollama)"
ollama pull "${CS_EMBED_MODEL:-embeddinggemma}" >/dev/null 2>&1 || \
  printf "   (could not pull %s — is ollama running?)\n" "${CS_EMBED_MODEL:-embeddinggemma}"

LAST_KEY=""
start_unsloth() {   # <port> <model-id> <logfile>
  local port=$1 model=$2 log=$3
  say "Unsloth: $model on :$port  (first run downloads the model)"
  unsloth run --model "unsloth/$model" --reasoning off --disable-tools \
    -c "$CTX" -p "$port" -y > "$log" 2>&1 &
  track $!
  local key=""
  for _ in $(seq 1 240); do
    key=$(grep -oE 'sk-unsloth-[A-Za-z0-9_-]+' "$log" | head -n1 || true)
    [ -n "$key" ] && break
    sleep 2
  done
  [ -n "$key" ] || die "no API key from :$port; check $log"
  printf "   waiting for the model to actually serve (loads after download)…\n"
  for _ in $(seq 1 240); do
    if curl -sf -m 90 -X POST "http://localhost:$port/v1/chat/completions" \
         -H "Authorization: Bearer $key" -H "Content-Type: application/json" \
         -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
         2>/dev/null | grep -q '"choices"'; then
      ok "ready on :$port"; LAST_KEY="$key"; return
    fi
    sleep 3
  done
  die "model on :$port never served a completion; check $log"
}

GEN_PORT=$(port_of "${CS_GEN_API_BASE_URL:-http://localhost:8001/v1}")
JUDGE_PORT=$(port_of "${CS_JUDGE_API_BASE_URL:-http://localhost:8001/v1}")

start_unsloth "$GEN_PORT" "${CS_GEN_MODEL:?set CS_GEN_MODEL}" "logs/unsloth-$GEN_PORT.log"
GEN_KEY="$LAST_KEY"; JUDGE_KEY="$LAST_KEY"
if [ "$JUDGE_PORT" != "$GEN_PORT" ]; then   # second model only if a different port
  start_unsloth "$JUDGE_PORT" "${CS_JUDGE_MODEL:?set CS_JUDGE_MODEL}" "logs/unsloth-$JUDGE_PORT.log"
  JUDGE_KEY="$LAST_KEY"
fi
export CS_GEN_API_KEY="$GEN_KEY" CS_JUDGE_API_KEY="$JUDGE_KEY"

say "Backend (FastAPI)"
uv run creativity-steer-serve > logs/backend.log 2>&1 &
track $!
for _ in $(seq 1 60); do
  curl -sf "http://${CS_HOST:-127.0.0.1}:${CS_PORT:-8000}/api/health" >/dev/null 2>&1 && { ok "backend ready"; break; }
  sleep 1
done

say "Frontend (Vite)"
command -v npm >/dev/null 2>&1 || die "node/npm not found (nvm?)."
( cd web && exec npm run dev ) > logs/frontend.log 2>&1 &
track $!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:5173/" >/dev/null 2>&1 && { ok "frontend ready"; break; }
  sleep 1
done

printf "\n\033[1;32m================================================\033[0m\n"
printf "\033[1;32m  Open  http://localhost:5173\033[0m\n"
printf "\033[1;32m  Stop  Ctrl+C  (or ./stop.sh)\033[0m\n"
printf "\033[1;32m================================================\033[0m\n"
command -v xdg-open >/dev/null 2>&1 && xdg-open "http://localhost:5173" >/dev/null 2>&1 || true

wait
