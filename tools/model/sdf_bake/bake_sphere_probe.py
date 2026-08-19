#!/usr/bin/env python3
"""烘一个解析球，与内核的解析采样逐项对比——甲3的验收②（决策0085第四节）。

**这个脚本跑在`tools/model/sdf_bake/.venv`里**（有point-cloud-utils），
但它import的`physics_engine`是**纯Python的内核**（走`PYTHONPATH`，不装）。
形制与`validation/run_comparison.py`逐字同源：
**两个环境、一份JSON**，内核那一侧永远不知道pcu存在。

    PYTHONPATH="${PWD}/src" tools/model/sdf_bake/.venv/bin/python \
        tools/model/sdf_bake/bake_sphere_probe.py \
        --out tools/model/sdf_bake/sphere_probe.report.json

## 三条腿，各自验的东西不同

**腿A｜逼近**：拿`mesh.triangulate_sphere`产的内接icosphere喂pcu，
把烘出来的值与`contact.field.sphere_distance_mm`（解析球）逐点比。
两者的差**几乎全部是三角化误差**（内接多面体不是球），所以判据是
"细分加密时这个差按阶降"——差不降就说明pcu那一侧有问题，而不是三角化。

**腿B｜符号**：逐点比符号。这是pcu被选中的**全部理由**所在
（0074第5.1节：广义缠绕数用体积判据定内外）。
判据：除去落在三角化误差带里的点，符号必须逐点一致。

**腿C｜脏网格**：把同一张网格的一个三角**绕向反过来**——
这正是0074第二节第3条实测到的真实件形态（非流形、绕向不一致）。
判据是一条对照：**内核的`mesh.mesh_mass_properties`当场拒**（它要闭合定向流形），
而**pcu照样给出正确的符号**。腿C是这条选型裁决唯一的直接证人；
没有它，"选pcu是因为网格脏"就只是一句转述。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.contact.field import sphere_distance_mm  # noqa: E402
from physics_engine.mesh import (  # noqa: E402
    MeshError,
    TriangleMesh,
    mesh_mass_properties,
    triangulate_sphere,
)
from physics_engine.shapes import Sphere  # noqa: E402

SPHERE_RADIUS_MM = 10.0
CENTRE_MM = (0.0, 0.0, 0.0)


def _arrays(mesh: TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(mesh.vertices_mm, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int32),
    )


def _query_points(count: int) -> np.ndarray:
    """球面附近一层壳上的查询点。**确定性**：用整数索引生成，不用随机数。

    半径在``[R − 1.5, R + 1.5]``之间扫，方向用黄金角螺旋——
    这两条都只依赖``count``，所以同一个``count``在任何机器上给同一串点。
    """

    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden * index
        shell = SPHERE_RADIUS_MM - 1.5 + 3.0 * ((index * 7) % count) / count
        points.append(
            (shell * radial * math.cos(angle), shell * radial * math.sin(angle), shell * z)
        )
    return np.asarray(points, dtype=np.float64)


def _analytic(points: np.ndarray) -> np.ndarray:
    return np.asarray(
        [sphere_distance_mm(tuple(point), CENTRE_MM, SPHERE_RADIUS_MM) for point in points],
        dtype=np.float64,
    )


def _leg_a_and_b(levels: tuple[int, ...], samples: int) -> dict:
    points = _query_points(samples)
    reference = _analytic(points)
    rows = []
    for level in levels:
        mesh = triangulate_sphere(Sphere(radius_mm=SPHERE_RADIUS_MM), subdivisions=level)
        vertices, faces = _arrays(mesh)
        baked, _face_ids, _bary = pcu.signed_distance_to_mesh(points, vertices, faces)
        deviation = np.abs(baked - reference)
        #: 内接多面体在球内，于是它的SDF在球外偏**大**、在球内偏**小**——
        #: 也就是``baked - reference >= 0``几乎处处。符号性质本身是一条判据：
        #: 数值噪声不会只往一边偏。
        signed = baked - reference
        band = float(np.max(deviation))
        sign_mismatch = int(
            np.count_nonzero((np.sign(baked) != np.sign(reference)) & (np.abs(reference) > band))
        )
        winding = pcu.triangle_soup_fast_winding_number(vertices, faces, points)
        #: **约定按实测取，不按文档取**——见报告里的``semantics``一节。
        winding_inside = winding > 0.5
        rows.append(
            {
                "subdivisions": level,
                "triangle_count": mesh.triangle_count,
                "max_abs_deviation_mm": float(np.max(deviation)),
                "mean_abs_deviation_mm": float(np.mean(deviation)),
                "min_signed_deviation_mm": float(np.min(signed)),
                "sign_mismatch_outside_the_deviation_band": sign_mismatch,
                "winding_sign_mismatch_outside_the_deviation_band": int(
                    np.count_nonzero(
                        (winding_inside != (reference < 0.0)) & (np.abs(reference) > band)
                    )
                ),
                "winding_at_the_centre": float(
                    np.atleast_1d(
                        pcu.triangle_soup_fast_winding_number(
                            vertices, faces, np.zeros((1, 3), dtype=np.float64)
                        )
                    )[0]
                ),
            }
        )
    return {
        "query_point_count": samples,
        "levels": rows,
        "max_abs_deviation_ratios": [
            rows[i]["max_abs_deviation_mm"] / rows[i + 1]["max_abs_deviation_mm"]
            for i in range(len(rows) - 1)
        ],
        "mean_abs_deviation_ratios": [
            rows[i]["mean_abs_deviation_mm"] / rows[i + 1]["mean_abs_deviation_mm"]
            for i in range(len(rows) - 1)
        ],
    }


def _leg_c(level: int, samples: int) -> dict:
    """脏网格：把一个三角的绕向反过来。**内核拒、pcu照样对。**"""

    clean = triangulate_sphere(Sphere(radius_mm=SPHERE_RADIUS_MM), subdivisions=level)
    triangles = list(clean.triangles)
    a, b, c = triangles[0]
    triangles[0] = (a, c, b)
    dirty = TriangleMesh(vertices_mm=clean.vertices_mm, triangles=tuple(triangles))

    kernel_refused = None
    try:
        mesh_mass_properties(dirty, density_kg_m3=2700.0)
    except MeshError as error:
        kernel_refused = str(error).split("——")[0].strip()

    points = _query_points(samples)
    reference = _analytic(points)
    vertices, faces = _arrays(dirty)
    winding = pcu.triangle_soup_fast_winding_number(vertices, faces, points)
    baked, _face_ids, _bary = pcu.signed_distance_to_mesh(points, vertices, faces)

    clean_vertices, clean_faces = _arrays(clean)
    clean_baked, _, _ = pcu.signed_distance_to_mesh(points, clean_vertices, clean_faces)

    band = float(np.max(np.abs(clean_baked - reference)))
    decisive = np.abs(reference) > band
    #: **整张网格全部反向**：这一档是选型裁决最直接的证人。
    #: 面法向投票法在这里必然全部判反；缠绕数法的``signed_distance_to_mesh``
    #: 实测**一个字节都不变**。
    flipped = np.asarray([(a, c, b) for a, b, c in clean.triangles], dtype=np.int32)
    flipped_baked, _, _ = pcu.signed_distance_to_mesh(points, clean_vertices, flipped)
    flipped_winding = pcu.triangle_soup_fast_winding_number(
        clean_vertices, flipped, points
    )
    clean_winding = pcu.triangle_soup_fast_winding_number(
        clean_vertices, clean_faces, points
    )

    return {
        "subdivisions": level,
        "kernel_refuses_the_dirty_mesh": kernel_refused is not None,
        "kernel_message": kernel_refused,
        "decisive_point_count": int(np.count_nonzero(decisive)),
        "max_abs_deviation_vs_clean_mesh_mm": float(np.max(np.abs(baked - clean_baked))),
        "sign_mismatch_outside_the_deviation_band": int(
            np.count_nonzero(np.sign(baked[decisive]) != np.sign(reference[decisive]))
        ),
        "winding_sign_mismatch_outside_the_deviation_band": int(
            np.count_nonzero(
                (winding[decisive] > 0.5) != (reference[decisive] < 0.0)
            )
        ),
        "all_triangles_flipped": {
            "max_abs_signed_distance_change_mm": float(
                np.max(np.abs(flipped_baked - clean_baked))
            ),
            "sign_mismatch_outside_the_deviation_band": int(
                np.count_nonzero(
                    np.sign(flipped_baked[decisive]) != np.sign(reference[decisive])
                )
            ),
            "winding_at_the_centre": float(
                np.atleast_1d(
                    pcu.triangle_soup_fast_winding_number(
                        clean_vertices, flipped, np.zeros((1, 3), dtype=np.float64)
                    )
                )[0]
            ),
            "winding_max_abs_sum_with_clean": float(
                np.max(np.abs(flipped_winding + clean_winding))
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="报告落盘路径（JSON）")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    arguments = parser.parse_args()

    import importlib.metadata as metadata

    report = {
        "schema": "tools/model/sdf_bake/sphere_probe.report",
        "schema_version": "1.0",
        "peer": {
            "package": "point-cloud-utils",
            "version": metadata.version("point-cloud-utils"),
            "license": "MIT",
            "numpy": metadata.version("numpy"),
        },
        "geometry": {
            "kind": "analytic sphere",
            "radius_mm": SPHERE_RADIUS_MM,
            "centre_mm": list(CENTRE_MM),
        },
        "semantics": {
            "triangle_soup_fast_winding_number": (
                "**文档写反了**。0.34.0的docstring原话是'positive for outside and "
                "negative for inside'，而实测返回的是**广义缠绕数本身**："
                "球心处1.00124、球外1.30e-05、球面外侧一点-8.34e-04。"
                "也就是内≈1、外≈0，判内外的阈值是**0.5**而不是符号。"
                "本仓一律按实测的约定用它——**同行库的文档不是金标，实测才是**"
                "（与cases/peer_fcl_distance那条'语义差异清单'同源）。"
            ),
            "signed_distance_to_mesh": (
                "**符号不来自面法向**：把整张网格的三角全部反向，"
                "返回的有符号距离最大只变**3.86e-07 mm**、符号一个点都没翻，"
                "而缠绕数从+1.00124整体翻到-1.00124。"
                "**那3.86e-07不是零**——pcu的fast winding number是带层次近似的，"
                "所以这里写的是实测量而不是'逐位不变'；"
                "把它说成逐位不变就是冒充一个没验过的性质。"
                "这正是0074第5.1节选它的那条性质的直接证据——"
                "内外由体积判据定，不靠面法向投票。"
                "**代价要一起写**：调用方**无法**通过反转网格来反转场的符号。"
            ),
        },
        "leg_a_and_b_approximation_and_sign": _leg_a_and_b(
            tuple(arguments.levels), arguments.samples
        ),
        "leg_c_dirty_mesh": _leg_c(3, arguments.samples),
    }

    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
