#!/usr/bin/env bash
# 建SDF烘焙档的独立环境。形制抄tools/view/setup_view_env.sh（决策0073/0076）。
#
# 硬约束：point-cloud-utils**只**装进tools/model/sdf_bake/.venv，绝不碰仓根的.venv。
# 本脚本因此从不接受"装进哪个环境"的参数——那正是要防的事。
#
# 路径全部从脚本自身位置算，无写死的用户主目录/盘符（rtime-project范式）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
VENV="${HERE}/.venv"

# 缺省用主环境的解释器**作基**（不是装进主环境——只是借它的小版本）。
# 与peer/view那两条同源：系统的python3常常不是同一版。
# 另有一条pcu特有的理由：0.34.0在PyPI上给的是cp38—cp313的平台wheel，
# 主环境的3.13在窗口内；系统python若跑在窗口外会退回**源码构建**，
# 那需要C++工具链与Eigen，实测不在本机的假设内。
if [ -n "${SDF_BAKE_PYTHON:-}" ]; then
  PYTHON="${SDF_BAKE_PYTHON}"
elif [ -x "${ROOT}/.venv/bin/python" ]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

if [ -x "${VENV}/bin/python" ]; then
  echo "已存在：${VENV}（要重建就先删掉它）"
else
  echo "建环境：${VENV}（解释器 ${PYTHON}；覆盖用 SDF_BAKE_PYTHON=... ）"
  "${PYTHON}" -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip
fi

"${VENV}/bin/pip" install -r "${HERE}/requirements-sdf-bake.txt"

echo
echo "--- 自检 ---"
"${VENV}/bin/python" - <<'PROBE'
import importlib.metadata as md
import platform

import point_cloud_utils as pcu

print("point-cloud-utils", md.version("point-cloud-utils"))
print("license          ", md.metadata("point-cloud-utils").get("License")
      or md.metadata("point-cloud-utils").get("License-Expression"))
print("platform         ", platform.platform(), platform.machine())
print("module           ", pcu.__file__)
# 0074第5.1节选它的那一样，**当场点名**——选型理由要能被自检打出来。
# 名字里带"soup"的那条才是脏网格路：它不要求封闭流形。
for name in ("triangle_soup_fast_winding_number", "signed_distance_to_mesh",
             "closest_points_on_mesh", "load_triangle_mesh"):
    print(f"{name:34} {hasattr(pcu, name)}")
PROBE

echo
echo "--- 传递闭包（实测，写进requirements的备查段）---"
"${VENV}/bin/pip" list --format=freeze

echo
echo "烘一个解析球并与内核的解析采样逐项对比（**两条命令用的是两个环境**）："
echo "  PYTHONPATH=\"\${PWD}/src\" ${VENV}/bin/python tools/model/sdf_bake/bake_sphere_probe.py \\"
echo "      --out work/sdf_bake/sphere_probe.json"
echo "  PYTHONPATH=\"\${PWD}/src\" ${ROOT}/.venv/bin/python -m pytest tests/test_sdf_bake_tool.py -q"
