#!/usr/bin/env bash
# 在master的SLURM分配里跑本仓的**任意一条命令**（M6的profile要用它）。
#
#   bash tools/master/run_on_master.sh '<命令>'
#   例：bash tools/master/run_on_master.sh 'python tools/profile_hotspots.py suite --marker batch'
#
# ## 为什么要有它
#
# `run_accept_on_master.sh`只会跑`accept.py`。M6的第一条纪律是"先profile再改"，
# 而profile必须在安静机器上跑——本机Mac负载常年5—20，同一功能面实测量到
# 93.7/121.5/225三个数（plans/07已登记）。本脚本把那份脚本的三条形制选择
# （git bundle送码、远端独立venv、srun分配内跑）原样复用，只把末尾那条命令
# 换成可变的——**不是新发明一套同步，是同一套的第二个入口**。
#
# 命令里的`python`由远端venv提供（脚本会把venv的bin放进PATH），
# `PYTHONPATH`指向检出目录的`src`。
set -euo pipefail

CMD="${1:?用法: run_on_master.sh '<要在仓库根跑的命令>'}"
HOST="${PE_MASTER_HOST:-master}"
REMOTE_DIR="${PE_MASTER_DIR:-program/physics-engine-accept}"
CORES="${PE_MASTER_CORES:-8}"
PARTITION="${PE_MASTER_PARTITION:-amd96c}"
TIMELIMIT="${PE_MASTER_TIME:-02:00:00}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -n "$(git status --porcelain)" ]; then
    echo "工作树不干净——发出去的回执必须能对回一个commit。先提交或stash。" >&2
    exit 2
fi
HEAD_SHA="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git bundle create "$STAGE/pe.bundle" HEAD 2>/dev/null
rtime-sync push "$STAGE/pe.bundle" "$HOST:/tmp/pe-run-$SHORT.bundle" >/dev/null
echo "[master] 已送达 $SHORT"

# 命令**写进远端脚本文件再逐字送过去**，不用嵌套heredoc——
# 上一份脚本第三次实跑就是被两层转义拆坏的（那条注释仍在`run_accept_on_master.sh`里）。
cat > "$STAGE/remote.sh" <<REMOTE_HEADER
set -euo pipefail
SHORT="$SHORT"
HEAD_SHA="$HEAD_SHA"
REMOTE_DIR="$REMOTE_DIR"
PARTITION="$PARTITION"
CORES="$CORES"
TIMELIMIT="$TIMELIMIT"
REMOTE_HEADER
printf 'USER_CMD=%q\n' "$CMD" >> "$STAGE/remote.sh"
cat >> "$STAGE/remote.sh" <<'REMOTE_BODY'
DIR="$HOME/$REMOTE_DIR/$SHORT"
rm -rf "$DIR"
mkdir -p "$(dirname "$DIR")"
git clone -q "/tmp/pe-run-$SHORT.bundle" "$DIR"
rm -f "/tmp/pe-run-$SHORT.bundle"
cd "$DIR"
GOT="$(git rev-parse HEAD)"
if [ "$GOT" != "$HEAD_SHA" ]; then
    echo "远端HEAD $GOT 与本仓 $HEAD_SHA 不同 —— 回执就对不回一个commit了" >&2
    exit 3
fi
VENV="$HOME/$REMOTE_DIR/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install -q --upgrade pip
fi
"$VENV/bin/python" -m pip install -q 'pytest>=8' 'ruff>=0.15,<1' 'numpy>=1.24'
ln -sfn "$VENV" "$DIR/.venv"
echo "[master] uptime: $(uptime)"
echo "[master] 分区=$PARTITION 核数=$CORES 命令: $USER_CMD"
srun -p "$PARTITION" -c "$CORES" --time="$TIMELIMIT" bash -lc "cd '$DIR' && export PYTHONPATH='$DIR/src' && export PATH='$VENV/bin:\$PATH' && $USER_CMD"
REMOTE_BODY

rtime-sync push "$STAGE/remote.sh" "$HOST:/tmp/pe-run-$SHORT.sh" >/dev/null
rtime-ssh "$HOST" "bash /tmp/pe-run-$SHORT.sh; rm -f /tmp/pe-run-$SHORT.sh"
