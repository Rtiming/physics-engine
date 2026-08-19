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
#
# ## 远端落点带一个**运行号**，不只带SHA（2026-08-18实测撞上）
#
# 第一版按`$HOME/<dir>/<short-sha>`落点并在开头`rm -rf`。同一个SHA上**并发**发两个
# 作业时，后发的那个把先发的那个的检出目录连同正在跑的进程一起删了——
# 实测回执是`cd: .../7bab386: No such file or directory`，**先发那个作业的
# 全部结果直接丢失，而且它自己的日志里没有任何"我被删了"的痕迹**。
# 这与本仓反复记的"静默出错"同族：不报错、只算错（这里是只丢结果）。
# 现在落点是`<short-sha>-<运行号>`，运行号取秒级时戳＋PID，**并发不再互相踩**。
set -euo pipefail

CMD="${1:?用法: run_on_master.sh '<要在仓库根跑的命令>'}"
HOST="${PE_MASTER_HOST:-master}"
REMOTE_DIR="${PE_MASTER_DIR:-program/physics-engine-accept}"
CORES="${PE_MASTER_CORES:-8}"
PARTITION="${PE_MASTER_PARTITION:-amd96c}"
TIMELIMIT="${PE_MASTER_TIME:-02:00:00}"
#: 运行号：同一SHA上的并发作业各自一份检出目录。
RUN_TAG="${PE_MASTER_RUN_TAG:-$(date +%Y%m%d%H%M%S)-$$}"

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
# **带上`main`这个引用，不只是`HEAD`。** 与`run_accept_on_master.sh`第54—58行同一条：
# 只打包`HEAD`时克隆出来没有`main`这个ref，而`tools/check_worktree_env.py`第④条断言
# 要解析基线分支，解析不出来就红。那次实测的回执是**读起来像失败、实际全绿**——
# 比没有回执更危险。那条在`run_accept_on_master.sh`里修过一次（提交5288bb5），
# **这个通用入口是同一个形状的第二处，2026-08-18补**：它一样会被用来跑`accept.py`
# 或任何要解析基线分支的命令，届时会重现同一份假失败的回执。
git bundle create "$STAGE/pe.bundle" main HEAD 2>/dev/null
rtime-sync push "$STAGE/pe.bundle" "$HOST:/tmp/pe-run-$SHORT-$RUN_TAG.bundle" >/dev/null
echo "[master] 已送达 $SHORT"

# 命令**写进远端脚本文件再逐字送过去**，不用嵌套heredoc——
# 上一份脚本第三次实跑就是被两层转义拆坏的（那条注释仍在`run_accept_on_master.sh`里）。
cat > "$STAGE/remote.sh" <<REMOTE_HEADER
set -euo pipefail
SHORT="$SHORT"
RUN_TAG="$RUN_TAG"
HEAD_SHA="$HEAD_SHA"
REMOTE_DIR="$REMOTE_DIR"
PARTITION="$PARTITION"
CORES="$CORES"
TIMELIMIT="$TIMELIMIT"
REMOTE_HEADER
printf 'USER_CMD=%q\n' "$CMD" >> "$STAGE/remote.sh"
cat >> "$STAGE/remote.sh" <<'REMOTE_BODY'
DIR="$HOME/$REMOTE_DIR/$SHORT-$RUN_TAG"
rm -rf "$DIR"
mkdir -p "$(dirname "$DIR")"
git clone -q "/tmp/pe-run-$SHORT-$RUN_TAG.bundle" "$DIR"
rm -f "/tmp/pe-run-$SHORT-$RUN_TAG.bundle"
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

# 作业体**再生成一个文件**，不往`srun ... bash -lc "..."`里塞第二层引号。
#
# 第一版塞了，代价是一次**静默挂死**（2026-08-18实测，作业跑了19分钟没有任何输出）：
# 那一层里写的是`export PATH='$VENV/bin:\$PATH'`——**单引号**，于是远端拿到的
# `PATH`字面量里带着一个没被展开的`$PATH`，`/usr/bin`当场从PATH上消失。
# 结果是命令里凡是用到外部程序的那一半（`tail`、`grep`……）全都找不到，
# 而`python`因为在venv里还在，于是管道一头在写、另一头根本不存在——
# **进程停在`pipe_read`上不退不报错**。
# 前几次跑之所以没撞上，纯粹是因为那几条命令碰巧只用了`python`一个外部程序。
#
# 这与本脚本开头那条"嵌套heredoc把srun那行拆坏"是同一个病：
# **多一层引号就多一次静默出错的机会**。作业体现在是一个逐字写下的文件。
JOB="$DIR/.pe_job.sh"
{
    echo 'set -euo pipefail'
    printf 'cd %q\n' "$DIR"
    printf 'export PYTHONPATH=%q\n' "$DIR/src"
    printf 'export PATH=%q:"$PATH"\n' "$VENV/bin"
    printf '%s\n' "$USER_CMD"
} > "$JOB"
srun -p "$PARTITION" -c "$CORES" --time="$TIMELIMIT" bash "$JOB"
REMOTE_BODY

rtime-sync push "$STAGE/remote.sh" "$HOST:/tmp/pe-run-$SHORT-$RUN_TAG.sh" >/dev/null
rtime-ssh "$HOST" "bash /tmp/pe-run-$SHORT-$RUN_TAG.sh; rm -f /tmp/pe-run-$SHORT-$RUN_TAG.sh"
