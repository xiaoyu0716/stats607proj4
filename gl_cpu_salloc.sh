#!/bin/bash
# GreatLakes: open an interactive CPU shell via srun.
# Usage: ./gl_cpu_shell.sh [-A ACCOUNT] [-p PARTITION] [--cpus N] [--mem MEM] [--time HH:MM:SS]
# Examples:
#   ./gl_cpu_shell.sh -A eecs-foo -p standard --cpus 8 --mem 32G --time 04:00:00
#   ./gl_cpu_shell.sh                           # auto-detect account/partition (best effort)

set -euo pipefail

ACCOUNT=''
PARTITION="standard"
CPUS=8
MEM=16G
TIME=05:00:00

# Simple arg parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    -A|--account)
      ACCOUNT="$2"; shift 2;;
    -p|--partition)
      PARTITION="$2"; shift 2;;
    --cpus)
      CPUS="$2"; shift 2;;
    --mem)
      MEM="$2"; shift 2;;
    --time)
      TIME="$2"; shift 2;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

# Best-effort auto-detect PARTITION if not provided (account optional)
if [[ -z "$PARTITION" ]]; then
  if command -v sacctmgr >/dev/null 2>&1; then
    # List user associations and pick a CPU-like partition if available
    mapfile -t ASSOC < <(sacctmgr show associations user="$USER" -Pn 2>/dev/null | awk -F '|' '{print $2"|"$4}' | sort -u)
    BEST_PARTITION=""
    for ap in "${ASSOC[@]}"; do
      p="${ap##*|}"
      if [[ -z "$p" ]]; then continue; fi
      if [[ "$p" == *standard* || "$p" == *cpu* ]]; then BEST_PARTITION="$p"; break; fi
    done
    # Fallback: take first partition if none matches
    if [[ -z "$BEST_PARTITION" && ${#ASSOC[@]} -gt 0 ]]; then
      BEST_PARTITION="${ASSOC[0]##*|}"
    fi
    PARTITION=${PARTITION:-$BEST_PARTITION}
  fi
fi

if [[ -z "$PARTITION" ]]; then
  echo "Failed to auto-detect partition. Please pass -p <PARTITION>." >&2
  exit 2
fi

echo "Opening interactive CPU shell with:"
echo "  partition : $PARTITION"
echo "  cpus      : $CPUS"
echo "  mem       : $MEM"
echo "  time      : $TIME"

if [[ -n "$ACCOUNT" ]]; then EXTRA_A=( -A "$ACCOUNT" ); else EXTRA_A=(); fi

exec srun "${EXTRA_A[@]}" -p "$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --pty bash -l