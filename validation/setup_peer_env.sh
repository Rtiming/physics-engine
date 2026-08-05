#!/usr/bin/env bash
# 建同行库对比实验的独立环境。
#
# 硬约束：同行库**只**装进validation/.venv，绝不碰仓根的.venv。
# 本脚本因此从不接受"装进哪个环境"的参数——那正是要防的事。
#
# 路径全部从脚本自身位置算，无写死的用户主目录/盘符（rtime-project范式）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
VENV="${HERE}/.venv"

# 缺省用主环境的解释器**作基**建同行环境（不是装进主环境——只是借它的小版本）。
# 理由：run_comparison.py要在同行环境里import纯Python的physics_engine，
# 两边Python小版本一致最省事；而系统的python3常常不是同一版（本机是3.14，主环境是3.13）。
if [ -n "${PEER_PYTHON:-}" ]; then
  PYTHON="${PEER_PYTHON}"
elif [ -x "${ROOT}/.venv/bin/python" ]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

if [ -x "${VENV}/bin/python" ]; then
  echo "已存在：${VENV}（要重建就先删掉它）"
else
  echo "建环境：${VENV}（解释器 ${PYTHON}；覆盖用 PEER_PYTHON=... ）"
  "${PYTHON}" -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip
fi

"${VENV}/bin/pip" install -r "${HERE}/requirements-peer.txt"

echo
echo "--- 自检 ---"
"${VENV}/bin/python" - <<'PROBE'
import importlib.metadata as md
import platform
import fcl
import numpy
print("python-fcl", md.version("python-fcl"))
print("numpy     ", numpy.__version__)
print("platform  ", platform.platform(), platform.machine())
print("fcl module", fcl.__file__)
PROBE

echo
echo "跑一次对拍："
echo "  ${VENV}/bin/python ${HERE}/peer_fcl/run_comparison.py --out-dir work/peer_fcl --name run-001"
echo "跑门（同行库缺席会skip）："
echo "  .venv/bin/python -m pytest tests/cases -q -m batch"
