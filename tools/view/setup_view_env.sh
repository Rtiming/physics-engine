#!/usr/bin/env bash
# 建查看器的独立环境。形制抄validation/setup_peer_env.sh（决策0025）。
#
# 硬约束：rerun**只**装进tools/view/.venv，绝不碰仓根的.venv。
# 本脚本因此从不接受"装进哪个环境"的参数——那正是要防的事。
#
# 路径全部从脚本自身位置算，无写死的用户主目录/盘符（rtime-project范式）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
VENV="${HERE}/.venv"

# 缺省用主环境的解释器**作基**（不是装进主环境——只是借它的小版本）。
# 与peer那条同源：系统的python3常常不是同一版。
# 另有一条rerun特有的理由：0.34.1的wheel是`cp310-abi3`，
# 主环境的3.13在它的支持窗口内，而系统python若跑在窗口外会退回**源码构建**——
# 那需要Rust工具链，且实测不在本机的假设内。
if [ -n "${VIEW_PYTHON:-}" ]; then
  PYTHON="${VIEW_PYTHON}"
elif [ -x "${ROOT}/.venv/bin/python" ]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

if [ -x "${VENV}/bin/python" ]; then
  echo "已存在：${VENV}（要重建就先删掉它）"
else
  echo "建环境：${VENV}（解释器 ${PYTHON}；覆盖用 VIEW_PYTHON=... ）"
  "${PYTHON}" -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip
fi

"${VENV}/bin/pip" install -r "${HERE}/requirements-view.txt"

# rerun的CLI**默认开启匿名使用统计**（首次运行会打印一段说明）。
# 本仓的默认取"最不外传"的那一档，所以这里显式关掉。
# 它写的是使用者自己的rerun配置目录，不动本仓任何字节。
"${VENV}/bin/rerun" analytics disable >/dev/null 2>&1 || \
  echo "（提示：rerun analytics disable 没跑成，可手动跑一次）"

echo
echo "--- 自检 ---"
"${VENV}/bin/python" - <<'PROBE'
import importlib.metadata as md
import platform
import rerun as rr
print("rerun-sdk", md.version("rerun-sdk"))
print("license  ", md.metadata("rerun-sdk").get("License-Expression"))
print("platform ", platform.platform(), platform.machine())
print("module   ", rr.__file__)
# 0074第5.3节选它的三样，逐个当场点名——**选型理由要能被自检打出来**。
print("Scalars  ", hasattr(rr, "Scalars"))   # 标量时间序列曲线（张力波形）
print("set_time ", hasattr(rr, "set_time"))  # 时间轴拖动回放
print("save     ", hasattr(rr, "save"))      # 一次run存成文件离线分享
PROBE

echo
echo "跑一次端到端（**两条命令用的是两个环境**，中间只有一份JSON）："
echo "  PYTHONPATH=\"\${PWD}/src\" .venv/bin/python tools/view/trace_from_closed_loop.py \\"
echo "      --band nominal_band --out work/view/nominal_band.trace.json"
echo "  ${VENV}/bin/python tools/view/replay.py \\"
echo "      work/view/nominal_band.trace.json --out work/view/nominal_band.rrd"
echo
echo "打开（另一台机器上同样这一条）："
echo "  ${VENV}/bin/rerun work/view/nominal_band.rrd"
echo "不开窗地验它读得回来："
echo "  ${VENV}/bin/rerun rrd verify work/view/nominal_band.rrd"
