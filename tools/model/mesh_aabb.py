#!/usr/bin/env python3
"""网格资产 → 真实AABB与SHA-256，产出可直接粘进`physics_scene`的声明字段。

## 这个工具补的洞

`shapes.MeshAsset`携带的AABB是**声明**的，不是引擎算出来的——引擎不解析网格字节
（spec/11第二之二，决策0073第五节）。声明式是对的：**声明可以是错的，但错了会红**，
比"引擎自己读所以一定对"更合本仓其余部分的形制。

**代价是那个声明得有人填。** 0017抓到的真实缺陷正是这个：
`examples/collision_preview_cell.scene.json`里声明的AABB是编的、
与真实包盒在z轴上**完全不相交**。`cases/mesh_asset_integrity`为此建了逐轴零容差的
包含门——**而那道门在已核查的同行里没有一家做**（决策0073第三节）。

**但那道门今天只覆盖一个语料**：仓内284字节的合成四面体。真实资产在消费方仓，
案例页第四节明写"无字节可查、不判"。本工具把声明从**人工填写**变成
**工具生成＋案例门复验**。

## 它不做什么

* **不进内核。** `src/physics_engine`永不import本模块（决策0073第四节的硬边界2，
  有门守着）。它是开发期工具，产物是一段可粘贴的JSON。
* **不算凸包。** 凸性是**声明**（spec/11规则2点名的"MuJoCo教训"）。
  本工具只报包盒；`convexity`要调用方自己判并自己负责。
  ——顺带一条核实过的事实：MuJoCo连关掉自动凸包的`convexhull`开关都在3.2.5删了。
* **不算质量属性。** `geometry.mass_properties`对`MeshAsset`失败关闭，
  **不拿AABB冒充质量分布**（spec/11第二之二）。本工具同样不越这条线。
* **不猜格式。** 后缀不认识就抛；ASCII STL当场拒（下面第二节）。

## 单位

**本工具不做单位换算，只如实报字节里的数。** `MeshAsset.units`是声明的字段，
换算是调用方的事——一个工具悄悄把mm当成m换过去，是本仓最怕的那种"跑得通但全错"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

#: 二进制STL的字节形制。与`tests/cases/test_mesh_asset_integrity.py`的解析器同源——
#: 那17行是本仓已有的、被案例门验过的实现，本模块沿用它的形制而不是另发明一个。
_STL_HEADER_BYTES = 80
_STL_COUNT_BYTES = 4
_STL_TRIANGLE_BYTES = 50


class MeshReadError(ValueError):
    """一切失败关闭。**不猜格式、不容错、不返回部分结果。**"""


def parse_binary_stl(payload: bytes) -> list[tuple[float, float, float]]:
    """二进制STL → 顶点表。**长度不自洽即抛**，不按"能读多少读多少"处理。

    ASCII STL**当场拒**而不是转去解析：两种格式的浮点文本化路径不同，
    一个工具同时支持两条而不声明用的是哪条，产出的AABB就带一个说不清的来源。
    """

    #: **格式识别在结构校验之前**，次序是有理由的：反过来写时，
    #: 一份**短的**ASCII STL会先撞上长度检查，于是使用者拿到的诊断是
    #: "你的文件被截断了"，而真相是"你的文件是另一种格式"。
    #: 这条次序是被`tests/test_model_tools.py`那条必红用例逼出来的。
    if payload[:5].lower() == b"solid":
        raise MeshReadError(
            "this looks like an ASCII STL; convert it to the binary form first "
            "(this tool does not guess between the two)"
        )
    if len(payload) < _STL_HEADER_BYTES + _STL_COUNT_BYTES:
        raise MeshReadError("binary STL is shorter than its header")
    (count,) = struct.unpack_from("<I", payload, _STL_HEADER_BYTES)
    expected = _STL_HEADER_BYTES + _STL_COUNT_BYTES + _STL_TRIANGLE_BYTES * count
    if len(payload) != expected:
        raise MeshReadError(
            f"binary STL declares {count} triangles but is {len(payload)} bytes "
            f"(expected {expected})"
        )
    vertices: list[tuple[float, float, float]] = []
    for index in range(count):
        #: 每个三角形是"法向3f + 三个顶点各3f + 属性2字节"，跳过前12字节的法向。
        base = _STL_HEADER_BYTES + _STL_COUNT_BYTES + _STL_TRIANGLE_BYTES * index + 12
        for corner in range(3):
            vertices.append(struct.unpack_from("<3f", payload, base + 12 * corner))
    return vertices


def parse_obj(payload: bytes) -> list[tuple[float, float, float]]:
    """Wavefront OBJ的``v``行 → 顶点表。

    **只读``v``**：包盒由几何顶点定，纹理坐标（``vt``）与法向（``vn``）不参与。
    OBJ允许``v x y z [w]``的第四个分量（有理B样条的权），**本工具拒收带``w``的行**
    ——那说明这不是一个普通多边形网格，而包盒的含义要另说。
    """

    vertices: list[tuple[float, float, float]] = []
    for number, raw in enumerate(payload.decode("utf-8", errors="strict").splitlines(), 1):
        line = raw.strip()
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise MeshReadError(
                f"OBJ line {number}: expected 'v x y z' with exactly three coordinates, "
                f"got {len(parts) - 1}"
            )
        try:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError as error:
            raise MeshReadError(f"OBJ line {number}: {error}") from error
    if not vertices:
        raise MeshReadError("OBJ contains no 'v' vertex lines")
    return vertices


#: 后缀 → 解析器。**白名单，认不出即抛**（失败关闭，不按后缀猜）。
_PARSERS = {".stl": parse_binary_stl, ".obj": parse_obj}


def axis_aligned_bounds(
    vertices: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """顶点表 → ``(min, max)``。**逐轴取极值，不做任何放大或取整。**

    不给"安全余量"是刻意的：余量是**声明者**的决定（`envelope`语义），
    工具替他加一点，那道包含门就变成在验工具自己加的那个数。
    """

    if not vertices:
        raise MeshReadError("cannot bound an empty vertex set")
    axes = tuple(zip(*vertices, strict=True))
    return (
        tuple(min(axis) for axis in axes),  # type: ignore[return-value]
        tuple(max(axis) for axis in axes),  # type: ignore[return-value]
    )


def describe(path: Path) -> dict[str, object]:
    """一个网格文件 → 可粘进场景声明的字段。"""

    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise MeshReadError(
            f"unsupported mesh suffix {suffix!r}; this tool reads {sorted(_PARSERS)}"
        )
    payload = path.read_bytes()
    vertices = parser(payload)
    lower, upper = axis_aligned_bounds(vertices)
    return {
        "path_relative": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "vertex_count": len(vertices),
        "aabb_min": list(lower),
        "aabb_max": list(upper),
        #: **单位与凸性刻意留空**：两者都是声明者的判断，不是从字节里读得出来的。
        #: 留`None`比填一个默认值诚实——默认值会被当成"工具查过了"。
        "units": None,
        "convexity": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="网格资产 → 真实AABB与SHA-256（纯标准库，不进内核）"
    )
    parser.add_argument("mesh", type=Path, nargs="+", help="STL（二进制）或OBJ文件")
    arguments = parser.parse_args(argv)
    results = []
    for path in arguments.mesh:
        try:
            results.append(describe(path))
        except (MeshReadError, OSError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            return 1
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
