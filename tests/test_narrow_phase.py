"""narrow phase第一片的**诚实性**门：可信度分级与降级行为。

原先本文件里的手算判据（球-球侵入5.0、球-胶囊2.0、旋转胶囊0.5、平行3.0、
交错2.0）已按轴7规则3搬进`cases/segment_distance/oracle.json`，由
`tests/cases/test_segment_distance.py`对拍——判据写死在测试里，验的是
"测试和内核是不是同一个人写的"，不是内核对不对。搬过去之后那批判据
多了五条退化分支、生成器身份与自指哈希。

留在这里的是**没有金标数值**的那一类事实：哪一族给什么可信度、
不支持的族会不会冒充精确。这类判据不属于oracle清单（它不是物理量的值），
属于接口诚实性（AGENTS.md本仓纪律第四条）。
"""

from __future__ import annotations

from physics_engine.collision import BroadPhaseCollisionQuery
from physics_engine.shapes import (
    CollisionShape,
    FiniteCylinder,
    PosedBody,
    SimBody,
    Sphere,
)


def _body(name: str, shape, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return PosedBody(
        SimBody(body_id=f"body/{name}", collision=CollisionShape(shape, "fitted")),
        translation_mm=translation,
        rotation_xyzw=rotation,
    )


def test_unsupported_pair_stays_broad_phase_honestly():
    """圆柱族没有narrow phase实现——事件必须自报`broad_phase`且不给穿透值。

    "不知道就说不知道"：冒充一个`penetration_mm`比不给更糟，因为调用方
    没有办法分辨哪个数是算出来的、哪个是编的。
    """

    sphere = _body("sph", Sphere(radius_mm=30.0))
    roller = _body(
        "rol", FiniteCylinder(radius_mm=45.0, half_width_mm=9.0), translation=(40.0, 0.0, 0.0)
    )
    events = BroadPhaseCollisionQuery((sphere, roller)).check_state()
    assert len(events) == 1
    assert events[0].confidence == "broad_phase"
    assert events[0].penetration_mm is None


def test_supported_pair_never_reports_broad_phase_confidence():
    """反向：球/胶囊族一旦报事件，可信度只能是`narrow_phase`。

    这条守的是降级不许反向发生——精确可算的对被降级成"大概撞了"，
    调用方会以为引擎能力比实际弱，同样是不诚实。
    """

    a = _body("a", Sphere(radius_mm=10.0))
    b = _body("b", Sphere(radius_mm=10.0), translation=(15.0, 0.0, 0.0))
    events = BroadPhaseCollisionQuery((a, b)).check_state()
    assert [event.confidence for event in events] == ["narrow_phase"]
    assert events[0].penetration_mm is not None
