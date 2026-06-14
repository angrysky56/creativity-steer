#!/usr/bin/env bash
# Stop everything started by ./start.sh.
set -uo pipefail
cd "$(dirname "$0")"

if [ ! -f .run/pids ]; then
  echo "Nothing to stop (no .run/pids)."
  exit 0
fi

printf "\033[1;33mStopping…\033[0m\n"
while read -r pid; do
  [ -n "$pid" ] || continue
  pkill -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
done < .run/pids
rm -f .run/pids
printf "\033[1;32mAll stopped.\033[0m\n"
