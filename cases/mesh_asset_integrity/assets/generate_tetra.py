#!/usr/bin/env python3
"""生成`tetra.stl`——本仓自带的合成网格资产（Chrono形制：生成脚本一并入库）。

为什么要合成资产：`examples/`那个KUKA连杆的STL在WII仓、不在本仓，
包盒保守性判据没有字节可查。所以仓内自带一个几百字节的确定性资产。

三条硬性质：

1. **纯标准库**，零依赖（本仓`dependencies = []`是承诺，案例侧也不破例）；
2. **确定性输出**：头部是固定ASCII、无时间戳、无路径、无随机数——
   同一份脚本在任何机器上产出逐字节相同的284字节；
3. **顶点全部是二进制小数**（±0.25的整数倍），float32存储无舍入，
   所以"真实包盒"是精确值，判据可以零容差。

四面体顶点（mm，刻意不对称、不居中、含负坐标）：

    v0 = (−3.5,  1.25, −2.0)
    v1 = (26.5,  1.25, −2.0)
    v2 = ( 1.5, 25.25, −2.0)
    v3 = ( 4.5,  7.25, 16.0)

真实包盒因此是 (−3.5, 1.25, −2.0) .. (26.5, 25.25, 16.0)。

用法：`python3 cases/mesh_asset_integrity/assets/generate_tetra.py`
"""

from __future__ import annotations

import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: 80字节固定头部。二进制STL的头不得以"solid"开头（那是ASCII STL的标志）。
HEADER = b"physics-engine synthetic tetra v1 units=mm generator=generate_tetra.py"

VERTICES = (
    (-3.5, 1.25, -2.0),
    (26.5, 1.25, -2.0),
    (1.5, 25.25, -2.0),
    (4.5, 7.25, 16.0),
)

#: 四个面，绕向使法向朝外（脚本自己验一遍，见`_outward_normal`）。
FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))


def _outward_normal(face: tuple[int, int, int]) -> tuple[float, float, float]:
    a, b, c = (VERTICES[index] for index in face)
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    length = sum(component * component for component in normal) ** 0.5
    unit = tuple(component / length for component in normal)
    centroid = tuple(sum(vertex[i] for vertex in VERTICES) / 4.0 for i in range(3))
    outward = sum(unit[i] * (a[i] - centroid[i]) for i in range(3))
    if outward <= 0.0:
        raise SystemExit(f"face {face} 的绕向使法向朝内——资产不诚实，先改绕向")
    return unit


def build() -> bytes:
    payload = bytearray(HEADER.ljust(80, b"\0"))
    payload += struct.pack("<I", len(FACES))
    for face in FACES:
        payload += struct.pack("<3f", *_outward_normal(face))
        for index in face:
            payload += struct.pack("<3f", *VERTICES[index])
        payload += struct.pack("<H", 0)
    return bytes(payload)


def main() -> int:
    data = build()
    expected_size = 84 + 50 * len(FACES)
    if len(data) != expected_size:
        raise SystemExit(f"STL长度{len(data)}不等于84+50×{len(FACES)}={expected_size}")
    (HERE / "tetra.stl").write_bytes(data)
    print(f"wrote tetra.stl, {len(data)} bytes, {len(FACES)} triangles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
