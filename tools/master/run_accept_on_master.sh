#!/usr/bin/env bash
# 把本仓同步到master并在SLURM分配里跑`accept.py full`。
#
#   bash tools/master/run_accept_on_master.sh [full|quick]
#
# ## 为什么要有它（决策/计划出处：plans/16 第M8节）
#
# 本仓的`accept full`墙钟**从没在一台安静机器上被量过一次**。2026-08-18同一功能面
# 在本机量到三个数：load 2.67时93.7秒、load 4.9—10.7时121.5秒、load 7.19时225秒，
# 而资源轴的load限值15.0**看不见这一档**（已四次实证，登记在plans/07）。
# 这台Mac整天没安静过（存储索引、RustDesk、别的项目），**在它上面量墙钟没有意义**。
#
# master空闲188/192核、324GB内存，`srun --test-only`测算可0分钟起跑——
# 那正是缺的那台安静机器。
#
# ## 三条形制上的选择，逐条写明为什么
#
# 1. **走`git bundle`而不是rsync整棵树，也不是`git archive`**。
#    `accept.py`有一条`repo_stable`轴，它要问`git rev-parse HEAD`——
#    **`git archive`交出的树没有`.git`，第一次实跑当场撞上`exit status 128`**。
#    rsync整棵树又会把`.venv`/`work`/`dist`带过去（或要维护一张会漂的排除表）。
#    `git bundle`两头都占：克隆出来的**HEAD SHA与本仓逐位相同**（实测`adf5a17`对`adf5a17`）、
#    工作树干净、体积只有3.2MB（`.git`是24MB）。
#    master上本来就有前人留下的`physics-engine-release-*.bundle`，**形制一致不是新发明**。
#    代价仍是**未提交的改动不会被测到**——这正是想要的：发出去的回执必须能对回一个commit。
# 2. **远端用独立venv不用系统python**。master的系统pytest是7.4.4，而本仓
#    `pyproject.toml`要求`pytest>=8`、`ruff>=0.15,<1`。用系统的会静默跑在一个
#    本仓从未声明支持的版本上。
# 3. **必须在`srun`分配内跑**。master是共享集群，直接在登录节点跑重活是违规的
#    （见rtime-fabric的master-compute-norms）。分配拿到的核数进回执，
#    **这样"那次是在几核上量的"永远有据**。
set -euo pipefail

PROFILE="${1:-full}"
HOST="${PE_MASTER_HOST:-master}"
# 远端落点：不写死任何用户主目录（本仓路径策略禁止硬编码`/home/...`），
# 由远端自己的`$HOME`展开。
REMOTE_DIR="${PE_MASTER_DIR:-program/physics-engine-accept}"
CORES="${PE_MASTER_CORES:-8}"
PARTITION="${PE_MASTER_PARTITION:-amd96c}"

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
echo "[master] 打包 $SHORT（$(wc -c < "$STAGE/pe.bundle") 字节）"

rtime-sync push "$STAGE/pe.bundle" "$HOST:/tmp/pe-accept-$SHORT.bundle" >/dev/null
echo "[master] 已送达 $HOST"

# 远端脚本：解包 → 备venv → 在srun分配里跑accept → 打印回执路径
# shellcheck disable=SC2087
rtime-ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
DIR="\$HOME/$REMOTE_DIR/$SHORT"
rm -rf "\$DIR"
mkdir -p "\$(dirname "\$DIR")"
git clone -q /tmp/pe-accept-$SHORT.bundle "\$DIR"
rm -f /tmp/pe-accept-$SHORT.bundle
cd "\$DIR"
# 克隆出来是分离头指针，SHA与本仓相同；核一遍，不相同就停。
GOT="\$(git rev-parse HEAD)"
if [ "\$GOT" != "$HEAD_SHA" ]; then
    echo "远端HEAD \$GOT 与本仓 $HEAD_SHA 不同 —— 回执就对不回一个commit了" >&2
    exit 3
fi

VENV="\$HOME/$REMOTE_DIR/.venv"
if [ ! -x "\$VENV/bin/python" ]; then
    echo "[master] 建venv（一次性）"
    python3 -m venv "\$VENV"
    "\$VENV/bin/python" -m pip install -q --upgrade pip
fi
# 依赖只装本仓声明的dev档；**版本按pyproject的约束**，不用系统那份7.4.4。
"\$VENV/bin/python" -m pip install -q 'pytest>=8' 'ruff>=0.15,<1' 'numpy>=1.24'

echo "[master] 分区=$PARTITION 核数=$CORES 起跑 accept.py $PROFILE"
srun -p "$PARTITION" -c "$CORES" --time=01:00:00 \\
     "\$VENV/bin/python" tools/accept.py "$PROFILE" 2>&1 | tail -40
echo "[master] 回执：\$DIR/work/acceptance/$PROFILE-latest.json"
REMOTE
