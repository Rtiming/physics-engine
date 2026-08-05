"""`case/rotated_aabb`的conformance门（轴7规则3）。

本文件不含Arvo公式，只把形状与位姿摆好、调`local_aabb_mm()`与`world_aabb_mm()`，
把结果交给清单比对。判据（含1e-9mm的世界盒容差与它的理由）全在
`cases/rotated_aabb/oracle.json`里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.oracles import load_manifest
from physics_engine.shapes import (
    Capsule,
    CollisionShape,
    FiniteCylinder,
    PosedBody,
    RoundedBox,
    SimBody,
    Sphere,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/rotated_aabb/oracle.json", root=ROOT)


def _shape(declaration: dict):
    fields = dict(declaration)
    kind = fields.pop("kind")
    if kind == "sphere":
        return Sphere(**fields)
    if kind == "capsule":
        return Capsule(
            point_a_mm=tuple(fields["point_a_mm"]),
            point_b_mm=tuple(fields["point_b_mm"]),
            radius_mm=fields["radius_mm"],
        )
    if kind == "rounded_box":
        return RoundedBox(
            half_extents_mm=tuple(fields["half_extents_mm"]),
            fillet_radius_mm=fields["fillet_radius_mm"],
        )
    if kind == "finite_cylinder":
        return FiniteCylinder(**fields)
    raise AssertionError(f"案例声明了测试不认识的形状种类：{kind}")


@pytest.mark.parametrize("case", MANIFEST.oracles, ids=[case.id for case in MANIFEST.oracles])
def test_world_aabb_matches_the_arvo_oracle(case):
    shape = _shape(case.inputs["shape"])
    posed = PosedBody(
        SimBody(body_id="body/subject", collision=CollisionShape(shape, "envelope")),
        translation_mm=tuple(case.inputs["translation_mm"]),
        rotation_xyzw=tuple(case.inputs["rotation_xyzw"]),
    )
    local_min, local_max = shape.local_aabb_mm()
    world_min, world_max = posed.world_aabb_mm()
    case.check_all(
        {
            "local_aabb_min_mm": list(local_min),
            "local_aabb_max_mm": list(local_max),
            "world_aabb_min_mm": list(world_min),
            "world_aabb_max_mm": list(world_max),
        }
    )


def test_world_box_encloses_every_transformed_corner():
    """保守性自洽：世界盒必须包住八个被变换的角点（Arvo式的存在理由）。

    这一条不读清单也成立——它是可证命题不是拟合数，所以放在这里当结构护栏：
    哪天有人把`world_aabb_mm`改成"贴合盒"，判据表红之前先在这里红。
    """

    for oracle in MANIFEST.oracles:
        shape = _shape(oracle.inputs["shape"])
        posed = PosedBody(
            SimBody(body_id="body/subject", collision=CollisionShape(shape, "envelope")),
            translation_mm=tuple(oracle.inputs["translation_mm"]),
            rotation_xyzw=tuple(oracle.inputs["rotation_xyzw"]),
        )
        low, high = shape.local_aabb_mm()
        world_low, world_high = posed.world_aabb_mm()
        for cx in (low[0], high[0]):
            for cy in (low[1], high[1]):
                for cz in (low[2], high[2]):
                    point = posed.transform_point_mm((cx, cy, cz))
                    for axis in range(3):
                        assert world_low[axis] - 1.0e-9 <= point[axis] <= world_high[axis] + 1.0e-9
