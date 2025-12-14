#!/usr/bin/env bash
# Great Lakes (UMich) GPU interactive allocation helper
# Usage:
#   ./gl_gpu_salloc.sh
#   ./gl_gpu_salloc.sh -A youraccount -p gpu -G 1 -c 8 -m 32G -t 04:00:00
#   ./gl_gpu_salloc.sh -A youraccount -p gpu -G 1 -c 16 -m 64G -t 08:00:00 -C a100
#
# Notes:
# - 用 -C 指定 GPU 约束（例如 a100 / l40s 等），不确定就不要加。
# - 进入节点后会自动打开交互 shell（bash -l）。
# - 按需在下面的 POST_LOGIN_CMDS 里添加 module / conda 初始化。

set -euo pipefail

# -------- defaults (按需修改) --------
# ACCOUNT="${ACCOUNT:-youraccount}"     # 必填：你的 LSA/CoE 等账户名，如 eecs-abc
PARTITION="${PARTITION:-gpu}"         # 分区：gpu / gpu-a100 / 其它（按需）
GPUS="${GPUS:-1}"                     # 每节点 GPU 数
CPUS="${CPUS:-8}"                     # 每任务 CPU 线程数
MEM="${MEM:-32G}"                     # 每节点内存
TIME="${TIME:-02:00:00}"              # 运行时间上限
CONSTRAINT="${CONSTRAINT:-}"          # 例如 a100, l40s；不需要就留空
NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"

# 进入节点后自动执行的命令（可按需修改/添加）
read -r -d '' POST_LOGIN_CMDS <<'EOF' || true
# ==== auto setup on allocated node ====
module purge
# 示例：按需加载你需要的环境/模块（请根据 Great Lakes 实际可用模块调整）
# module load gcc/13.2.0 cuda/12.1 python/3.11.5
# 如果你用 conda:
# source ~/.bashrc 2>/dev/null || true
# conda activate myenv 2>/dev/null || true
nvidia-smi || true
echo "[INFO] You're now on $(hostname). Happy computing!"
EOF
# -------------------------------------

# -------- parse args --------
usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -A|--account) ACCOUNT="$2"; shift 2;;
    -p|--partition) PARTITION="$2"; shift 2;;
    -G|--gpus|--gpus-per-node) GPUS="$2"; shift 2;;
    -c|--cpus|--cpus-per-task) CPUS="$2"; shift 2;;
    -m|--mem) MEM="$2"; shift 2;;
    -t|--time) TIME="$2"; shift 2;;
    -C|--constraint) CONSTRAINT="$2"; shift 2;;
    -N|--nodes) NODES="$2"; shift 2;;
    --ntasks-per-node) NTASKS_PER_NODE="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

# if [[ -z "$ACCOUNT" || "$ACCOUNT" == "youraccount" ]]; then
#   echo "[ERROR] 请通过 -A/--account 或环境变量 ACCOUNT 指定你的 Slurm 账户名 (如 eecs-xxx)。"
#   exit 2
# fi

# 组装 salloc 参数
args=(
#   --account="$ACCOUNT"
  --partition="$PARTITION"
  --nodes="$NODES"
  --ntasks-per-node="$NTASKS_PER_NODE"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --gpus-per-node="$GPUS"
  --time="$TIME"
)

if [[ -n "$CONSTRAINT" ]]; then
  args+=(--constraint="$CONSTRAINT")
fi

echo "[INFO] Running: salloc ${args[*]}"
# 使用 --preserve-env 继承当前环境变量（如 CUDA_VISIBLE_DEVICES 不需要，依然安全）
salloc "${args[@]}" bash -lc '
  set -euo pipefail
  '"$POST_LOGIN_CMDS"'
  # 打开交互式登录 shell（-l 读取登录配置）
  exec bash -l
'