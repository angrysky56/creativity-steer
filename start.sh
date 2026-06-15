#!/usr/bin/env bash
# Start everything: model server(s) + backend + frontend.
# Stop with Ctrl+C (or ./stop.sh).
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

[ -f .env ] || cp .env.example .env
set -a; . ./.env; set +a
CTX="${CS_CONTEXT:-8192}"
LLAMA_BIN="${CS_LLAMA_SERVER:-$HOME/.unsloth/llama.cpp/llama-server}"
LLAMA_LIBS="${CS_LLAMA_LIBS:-$HOME/.unsloth/llama.cpp/build/bin}"

kind_of() { local v="CS_${1}_BACKEND"; echo "${!v:-${CS_BACKEND:-ollama}}"; }
val()     { local v="CS_${1}_${2}"; echo "${!v:-}"; }

# node for the frontend
if ! command -v node >/dev/null 2>&1; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # shellcheck disable=SC1091
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm use default >/dev/null 2>&1 || nvm use node >/dev/null 2>&1 || true
fi

say "Local models (Ollama)"
pull_ollama() { ollama pull "$1" >/dev/null 2>&1 && ok "$1" || echo "   (could not pull $1 — is ollama running?)"; }
[ "$(kind_of GEN)" = ollama ]   && pull_ollama "$CS_GEN_MODEL"
[ "$(kind_of JUDGE)" = ollama ] && pull_ollama "$CS_JUDGE_MODEL"
pull_ollama "${CS_EMBED_MODEL:-embeddinggemma}"

# Launch a llama-server for each unique API port (thinking OFF).
declare -A SERVED
start_llama() {   # <port> <hf-repo:quant> <alias> <logfile>
  local port=$1 hf=$2 alias=$3 log=$4
  [ -x "$LLAMA_BIN" ] || die "llama-server not found at $LLAMA_BIN — install Unsloth or set CS_LLAMA_SERVER."
  say "Model server: $alias on :$port  (first run downloads the model)"
  LD_LIBRARY_PATH="$LLAMA_LIBS:${LD_LIBRARY_PATH:-}" "$LLAMA_BIN" \
    -hf "$hf" --host 127.0.0.1 --port "$port" -c "$CTX" -ngl 99 --jinja \
    --reasoning off --temp 1.0 --top-p 0.95 --top-k 64 --alias "$alias" \
    > "$log" 2>&1 &
  track $!
  for _ in $(seq 1 300); do
    curl -s -m 5 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && { ok "ready on :$port"; return; }
    sleep 3
  done
  die "model on :$port never became ready; check $log"
}
maybe_serve() {   # <role>
  [ "$(kind_of "$1")" = api ] || return 0
  local port hf; port=$(port_of "$(val "$1" API_BASE_URL)"); hf=$(val "$1" HF)
  [ -n "$hf" ] || die "CS_${1}_HF is empty (stale .env?). Run: rm .env && ./start.sh"
  [ -n "${SERVED[$port]:-}" ] && return 0
  start_llama "$port" "$hf" "$(val "$1" MODEL)" "logs/llama-$port.log"
  SERVED[$port]=1
}
maybe_serve GEN
maybe_serve JUDGE

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
