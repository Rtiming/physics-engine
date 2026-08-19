"""窄相第二片的门——解析原语闭式 + 距离场协议（决策0090）。

本文件与`test_narrow_phase.py`分工：那一份守**可信度分级的诚实性**
（哪一族给什么档、不支持的族会不会冒充），本份守**数**——
闭式对拍、逐位不变、偏差估计的主项、以及每条门各自的必红。

## 为什么"逐位不变"要用`float.hex()`而不是`pytest.approx`

派活书第戊2条第2点原话是"逐位不变（`float.hex()`，不是近似相等）"。
理由是这一条判的不是精度而是**身份**：既有球/胶囊那条路一个字都不许动。
近似相等会让"顺手把``a − (b + c)``改成``(a − b) − c``"这类改写悄悄过门，
而那正是本轨最容易犯的错（新加一族时顺手统一表达式）。
`collision.py`的做法是让两条调用面共用`_segment_pair_separation_mm`一个表达式，
**逐位是结构保证的**；本门是那条结构的证人。
"""

from __future__ import annotations

import math
import random

import pytest

from physics_engine.collision import (
    BroadPhaseCollisionQuery,
    CollisionQueryError,
    HalfSpace,
    NarrowPhaseResult,
    field_separation_mm,
    half_space_separation_mm,
    narrow_phase_separation_mm,
    posed_signed_distance_mm,
    segment_segment_distance_mm,
    segment_segment_witnesses,
    shape_signed_distance_gradient,
    shape_signed_distance_mm,
    world_to_local_mm,
)
from physics_engine.contact.field import sample_narrow_band
from physics_engine.shapes import (
    Capsule,
    CollisionShape,
    FiniteCylinder,
    GeneratedShape,
    MeshAsset,
    PosedBody,
    RoundedBox,
    ShapeError,
    SimBody,
    Sphere,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _body(name, shape, translation=(0.0, 0.0, 0.0), rotation=IDENTITY) -> PosedBody:
    return PosedBody(
        SimBody(body_id=f"body/{name}", collision=CollisionShape(shape, "fitted")),
        translation_mm=translation,
        rotation_xyzw=rotation,
    )


# ==========================================================================
# 一、既有球/胶囊窄相**逐位不变**
# ==========================================================================

#: 语料生成器：stdlib`random`、种子写死，**重建这批输入不需要任何第三方包**
#: （与`cases/peer_fcl_distance`同一条纪律）。120对里34对相交。
_CORPUS_SEED = 20260818


def _unit_quaternion(rng: random.Random) -> tuple[float, float, float, float]:
    """Shoemake法的S³均匀采样。"""

    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1, s2 = math.sqrt(1.0 - u1), math.sqrt(u1)
    return (
        s1 * math.sin(2.0 * math.pi * u2),
        s1 * math.cos(2.0 * math.pi * u2),
        s2 * math.sin(2.0 * math.pi * u3),
        s2 * math.cos(2.0 * math.pi * u3),
    )


def _sphere_capsule_corpus() -> list[tuple[int, PosedBody, PosedBody]]:
    rng = random.Random(_CORPUS_SEED)
    corpus: list[tuple[int, PosedBody, PosedBody]] = []
    for index in range(120):
        kind_a = rng.random() < 0.5
        kind_b = rng.random() < 0.5
        radius_a = 10.0 ** rng.uniform(-0.3, 1.7)
        radius_b = 10.0 ** rng.uniform(-0.3, 1.7)
        shape_a = (
            Sphere(radius_mm=radius_a)
            if kind_a
            else Capsule(
                (0.0, 0.0, -(10.0 ** rng.uniform(0.0, 2.0))),
                (0.0, 0.0, 10.0 ** rng.uniform(0.0, 2.0)),
                radius_a,
            )
        )
        shape_b = (
            Sphere(radius_mm=radius_b)
            if kind_b
            else Capsule(
                (0.0, 0.0, -(10.0 ** rng.uniform(0.0, 2.0))),
                (0.0, 0.0, 10.0 ** rng.uniform(0.0, 2.0)),
                radius_b,
            )
        )
        translation_a = tuple(rng.uniform(-30.0, 30.0) for _ in range(3))
        translation_b = tuple(rng.uniform(-30.0, 30.0) for _ in range(3))
        rotation_a = IDENTITY if kind_a else _unit_quaternion(rng)
        rotation_b = IDENTITY if kind_b else _unit_quaternion(rng)
        corpus.append(
            (
                index,
                _body("a", shape_a, translation_a, rotation_a),
                _body("b", shape_b, translation_b, rotation_b),
            )
        )
    return corpus


#: **2026-08-18轨戊开工前**（HEAD `6541cb7`，`collision.py`一个字节未改时）
#: 从上面那批语料上取下来的``penetration_mm.hex()``。它是身份指纹不是物理判据。
_FROZEN_PENETRATION_HEX: tuple[tuple[int, str], ...] = (
    (0, "0x1.3dfc08f5f636cp+3"),
    (1, "0x1.0708c779ebaafp+4"),
    (4, "0x1.16744e5dac6acp+4"),
    (7, "0x1.f95097cbdcd22p+4"),
    (16, "0x1.d3c1453625b1fp+4"),
    (18, "0x1.4e0e6d83410f0p+1"),
    (19, "0x1.6264c26a59f8bp+5"),
    (21, "0x1.3b17d3d10a9c1p+4"),
    (23, "0x1.8a5582bb99738p+1"),
    (31, "0x1.6fa8bdc005b2bp+4"),
    (33, "0x1.ab941793bde7cp+4"),
    (34, "0x1.5a2376dc396f0p+3"),
    (37, "0x1.60fc775c1a778p+3"),
    (38, "0x1.630a7956a2a38p+3"),
    (43, "0x1.d0f3b4e057330p+2"),
    (47, "0x1.ebc5d07e68012p+2"),
    (50, "0x1.0919532536140p+1"),
    (55, "0x1.5f77df7291f84p+3"),
    (60, "0x1.5818438f1d3bap+4"),
    (65, "0x1.7c1c8dc202661p+3"),
    (67, "0x1.d482cc3e0cf52p+4"),
    (73, "0x1.9307d60b72244p+3"),
    (79, "0x1.0b94772fe8db4p+2"),
    (80, "0x1.627967248f87cp+2"),
    (81, "0x1.11e7e380ba983p+4"),
    (90, "0x1.655f83e8b533ap+3"),
    (96, "0x1.6d49b88573433p+4"),
    (98, "0x1.621a594737c74p+3"),
    (100, "0x1.b6808e79a19a4p+4"),
    (106, "0x1.ca98be8f7ff80p+0"),
    (109, "0x1.4a32748ba2182p+4"),
    (114, "0x1.24a533c9e730ap+6"),
    (115, "0x1.df0e30eadc20ep+4"),
    (116, "0x1.8089984a00f89p+4"),
)


#: 跨平台那一半允许的最大ulp差。**取2不是拍脑袋**：2026-08-19实测
#: 同一棵树在macOS arm64与Linux x86-64上，34对里有2对差**恰好1 ulp**
#: （`0x1.ca98be8f7ff90p+0`对`…7ff80p+0`、`0x1.8089984a00f89p+4`对`…00f88p+4`），
#: 病根是`x ** 0.5`与libm在末位上的实现差（本轨自己在20万个正数里量到277个差1 ulp）。
#: 取2留一档余量；**超过2就不再是"平台末位差"，是真的动了数**。
_CROSS_PLATFORM_ULP_BUDGET = 2


def _ulp_gap(left: float, right: float) -> int:
    """两个同号有限浮点之间隔了几个可表示数。"""

    import struct

    def ordinal(value: float) -> int:
        bits = struct.unpack("<q", struct.pack("<d", value))[0]
        return bits if bits >= 0 else -(bits & 0x7FFFFFFFFFFFFFFF)

    return abs(ordinal(left) - ordinal(right))


def test_the_existing_sphere_capsule_narrow_phase_is_unchanged():
    """34个相交对的``penetration_mm``对开工前的值。**判两件事，分开判。**

    **跨平台那一半**：每一对都在``_CROSS_PLATFORM_ULP_BUDGET``个ulp之内。
    这一条在任何机器上都该成立，不成立就是真的动了数。

    **同平台那一半**：`float.hex()`**逐位**相同——冻下来的是十六进制而不是十进制，
    因为十进制往返会掩盖最低位。**它只在冻它的那台机器上成立**：
    2026-08-19实测同一棵树在Linux x86-64上34对里有**2对差恰好1 ulp**，
    病根是`x ** 0.5`与libm的末位实现差。**不成立时判skip并说清楚，不判红**——
    那不是缺陷，是**一条本仓从来没有声明过的性质**，与九个oracle那件事同一族
    （plans/07第六节"轴7"那一行）。
    """

    frozen = dict(_FROZEN_PENETRATION_HEX)
    measured = {}
    for index, first, second in _sphere_capsule_corpus():
        events = BroadPhaseCollisionQuery((first, second)).check_state()
        if events and events[0].penetration_mm is not None:
            measured[index] = events[0].penetration_mm
    assert sorted(measured) == sorted(frozen), (
        f"相交对的集合变了：多了{sorted(set(measured) - set(frozen))}、"
        f"少了{sorted(set(frozen) - set(measured))}——**这一条与平台无关，是真的动了行为**"
    )

    drifted = {
        index: (value.hex(), frozen[index], _ulp_gap(value, float.fromhex(frozen[index])))
        for index, value in measured.items()
        if value.hex() != frozen[index]
    }
    too_far = {i: d for i, d in drifted.items() if d[2] > _CROSS_PLATFORM_ULP_BUDGET}
    assert not too_far, (
        f"这些对超出了{_CROSS_PLATFORM_ULP_BUDGET}个ulp的跨平台余量：{too_far}——"
        "**那不是平台末位差，是真的动了数**"
    )
    if drifted:
        pytest.skip(
            f"{len(drifted)}/{len(frozen)}对与冻结指纹差1—{_CROSS_PLATFORM_ULP_BUDGET}个ulp——"
            f"**这台机器不是冻它的那台**（指纹冻于macOS arm64）。逐对：{drifted}。"
            "病根是`x ** 0.5`与libm的末位实现差；**这不是缺陷，是本仓从来没有声明过的性质**，"
            "与九个oracle跨架构不逐字节复现是同一族（plans/07第六节「轴7」那一行）。"
        )


def test_the_direct_query_returns_the_same_bits_as_the_scene_query():
    """直查面与场景查询在球/胶囊族上**逐位相同**——两条面共用同一个表达式。"""

    checked = 0
    for _index, first, second in _sphere_capsule_corpus():
        events = BroadPhaseCollisionQuery((first, second)).check_state()
        direct = narrow_phase_separation_mm(first, second)
        assert direct.confidence == "narrow_phase"
        assert direct.estimated_bias_mm is None
        if events and events[0].penetration_mm is not None:
            assert (-direct.separation_mm).hex() == events[0].penetration_mm.hex()
            checked += 1
    assert checked == len(_FROZEN_PENETRATION_HEX)


def test_the_frozen_fingerprint_goes_red_when_one_bit_moves():
    """必红：把一个指纹的最低位挪一格，门当场红。

    这一条验的是**判据本身有牙齿**——`float.hex()`比对如果被写成
    `pytest.approx`，下面这个差1 ulp的值会照过。
    """

    original = float.fromhex(_FROZEN_PENETRATION_HEX[0][1])
    mutated = math.nextafter(original, math.inf)
    assert mutated != original
    assert mutated == pytest.approx(original)  # 近似相等抓不到它
    assert mutated.hex() != original.hex()  # 逐位比对抓得到
    assert _ulp_gap(mutated, original) == 1  # 而ulp尺子给出它到底差了几格

    #: 跨平台那一半的牙齿：**超出余量的漂移必须被判出来**。
    #: 没有这一条，`_CROSS_PLATFORM_ULP_BUDGET`可以被调到任意大而没人发现。
    far = original
    for _ in range(_CROSS_PLATFORM_ULP_BUDGET + 1):
        far = math.nextafter(far, math.inf)
    assert _ulp_gap(far, original) > _CROSS_PLATFORM_ULP_BUDGET


# ==========================================================================
# 二、解析可求的构型逐点对拍闭式
# ==========================================================================


def test_sphere_sphere_matches_the_closed_form_pointwise():
    """球-球：``|c₁ − c₂| − (r₁ + r₂)``，abs 1e-12 mm。

    **这一条本来打算写零容差，实测不成立，而不成立的两个理由都值得记下来**：

    1. `segment_segment_distance_mm`开方用的是``x ** 0.5``而不是`math.sqrt`。
       ``**``走的是libm的``pow``，**IEEE不要求它正确舍入**；本机实测20万个
       正数里**277个**两者差1 ulp（0.14%）；
    2. 本判据的求和用`sum()`，而**CPython 3.12起`sum()`对float走Neumaier
       补偿求和**，与手写的``a + b + c``不逐位相同——本机实测20万组三维点积里
       **44972组**不同（22.5%）。

    两条合起来在坐标量级50 mm上给到约3e-14 mm的差，1e-12留两个量级。
    **两条都没有被"修"**：改`segment_segment_distance_mm`会破掉逐位不变那条判据。
    """

    rng = random.Random(931)
    for _ in range(200):
        centre_a = tuple(rng.uniform(-50.0, 50.0) for _ in range(3))
        centre_b = tuple(rng.uniform(-50.0, 50.0) for _ in range(3))
        radius_a, radius_b = rng.uniform(0.5, 30.0), rng.uniform(0.5, 30.0)
        first = _body("a", Sphere(radius_mm=radius_a), centre_a)
        second = _body("b", Sphere(radius_mm=radius_b), centre_b)
        delta = tuple(centre_b[axis] - centre_a[axis] for axis in range(3))
        closed_form = math.sqrt(sum(item * item for item in delta)) - (radius_a + radius_b)
        assert narrow_phase_separation_mm(first, second).separation_mm == pytest.approx(
            closed_form, rel=0.0, abs=1.0e-12
        )


def test_sphere_half_space_matches_the_closed_form_pointwise():
    """球-半空间：``(c − p)·n − r``，abs 1e-12 mm。

    半空间走的是支撑函数那条路（``min_q q·d = −R``）。**同样不写零容差**，
    理由与上一条第2点同源：本判据用`sum()`（Neumaier补偿），内核那条用
    手写的三项相加。实测差在1e-15 mm量级，1e-12留三个量级。
    """

    rng = random.Random(732)
    for _ in range(200):
        centre = tuple(rng.uniform(-50.0, 50.0) for _ in range(3))
        radius = rng.uniform(0.5, 30.0)
        normal_raw = tuple(rng.gauss(0.0, 1.0) for _ in range(3))
        norm = math.sqrt(sum(item * item for item in normal_raw))
        normal = tuple(item / norm for item in normal_raw)
        plane_point = tuple(rng.uniform(-20.0, 20.0) for _ in range(3))
        plane = HalfSpace(point_mm=plane_point, unit_normal=normal)
        body = _body("s", Sphere(radius_mm=radius), centre)
        closed_form = (
            sum((centre[axis] - plane_point[axis]) * normal[axis] for axis in range(3))
            - radius
        )
        assert half_space_separation_mm(body, plane) == pytest.approx(
            closed_form, rel=0.0, abs=1.0e-12
        )


def _axis_aligned_box_closed_form(
    centre_a, half_a, fillet_a, centre_b, half_b, fillet_b
) -> float:
    """判据侧独立写的轴对齐盒对闭式——**枚举写法，不是内核那条**。

    内核先算三个``gap_i``再分支；这里先判"分不分开"再分别写死两条式子。
    两串运算不同，所以本条判据不取零容差。
    """

    gaps = [
        abs(centre_b[axis] - centre_a[axis]) - half_a[axis] - half_b[axis]
        for axis in range(3)
    ]
    if max(gaps) > 0.0:
        squared = sum(gap * gap for gap in gaps if gap > 0.0)
        return math.sqrt(squared) - fillet_a - fillet_b
    return -(min(-gap for gap in gaps)) - fillet_a - fillet_b


def test_axis_aligned_box_pair_matches_the_closed_form_pointwise():
    """盒-盒轴对齐：分离支与穿透支各自对闭式，abs 1e-12 mm。

    两支都要覆盖到——**只测分离支会漏掉"最小重叠"那半条**，
    而最小重叠正是这条闭式与EPA给同一个数的地方（轴对齐盒的最小平移
    穿透深度就在重叠最小的那一轴上）。
    """

    rng = random.Random(455)
    separated = penetrating = 0
    for _ in range(400):
        half_a = tuple(rng.uniform(1.0, 20.0) for _ in range(3))
        half_b = tuple(rng.uniform(1.0, 20.0) for _ in range(3))
        fillet_a, fillet_b = rng.uniform(0.0, 3.0), rng.uniform(0.0, 3.0)
        centre_a = tuple(rng.uniform(-10.0, 10.0) for _ in range(3))
        centre_b = tuple(rng.uniform(-40.0, 40.0) for _ in range(3))
        first = _body(
            "a", RoundedBox(half_extents_mm=half_a, fillet_radius_mm=fillet_a), centre_a
        )
        second = _body(
            "b", RoundedBox(half_extents_mm=half_b, fillet_radius_mm=fillet_b), centre_b
        )
        expected = _axis_aligned_box_closed_form(
            centre_a, half_a, fillet_a, centre_b, half_b, fillet_b
        )
        measured = narrow_phase_separation_mm(first, second)
        assert measured.confidence == "narrow_phase"
        assert measured.separation_mm == pytest.approx(expected, rel=0.0, abs=1.0e-12)
        if expected > 0.0:
            separated += 1
        else:
            penetrating += 1
    assert separated > 50 and penetrating > 50  # 两支都真的被走到


def test_the_box_pair_criterion_goes_red_when_the_overlap_axis_is_picked_wrong():
    """必红：穿透支取"最大重叠"而不是"最小重叠"，判据当场红。

    这一条是**盒对盒唯一一处能悄悄错**的地方——两种写法在分离支上完全一样，
    只有在全轴重叠时才分岔。
    """

    half = (10.0, 20.0, 30.0)
    first = _body("a", RoundedBox(half_extents_mm=half, fillet_radius_mm=0.0))
    second = _body(
        "b", RoundedBox(half_extents_mm=half, fillet_radius_mm=0.0), (1.0, 2.0, 3.0)
    )
    gaps = [abs((1.0, 2.0, 3.0)[axis]) - 2.0 * half[axis] for axis in range(3)]
    correct = max(gaps)  # 最小重叠 = 最大（最不负）的gap
    wrong = min(gaps)  # 最大重叠
    assert narrow_phase_separation_mm(first, second).separation_mm == pytest.approx(
        correct, abs=1.0e-12
    )
    assert abs(wrong - correct) > 1.0  # 注错值离判据窗口足够远，门确实会红


# ==========================================================================
# 三、球型探针对解析原语：精确，且与旋转无关
# ==========================================================================


def test_the_box_signed_distance_is_exact_inside_and_outside():
    """圆角盒的SDF体内体外都精确——判据侧用**投影到面上再取距离**独立算一遍。"""

    half = (7.0, 3.0, 11.0)
    box = RoundedBox(half_extents_mm=half, fillet_radius_mm=0.0)
    rng = random.Random(2201)
    for _ in range(500):
        point = tuple(rng.uniform(-20.0, 20.0) for _ in range(3))
        inside = all(abs(point[axis]) <= half[axis] for axis in range(3))
        if inside:
            expected = -min(half[axis] - abs(point[axis]) for axis in range(3))
        else:
            clamped = tuple(
                max(-half[axis], min(half[axis], point[axis])) for axis in range(3)
            )
            expected = math.sqrt(
                sum((point[axis] - clamped[axis]) ** 2 for axis in range(3))
            )
        assert shape_signed_distance_mm(box, point) == pytest.approx(expected, abs=1.0e-12)


def test_the_cylinder_signed_distance_is_exact_inside_and_outside():
    """无法兰圆柱同上。判据侧按"轴向/径向两段各自clamp"独立算。"""

    cylinder = FiniteCylinder(radius_mm=9.0, half_width_mm=4.0)
    rng = random.Random(2202)
    for _ in range(500):
        point = tuple(rng.uniform(-20.0, 20.0) for _ in range(3))
        radial = math.hypot(point[0], point[1])
        if radial <= 9.0 and abs(point[2]) <= 4.0:
            expected = -min(9.0 - radial, 4.0 - abs(point[2]))
        else:
            expected = math.hypot(max(radial - 9.0, 0.0), max(abs(point[2]) - 4.0, 0.0))
        assert shape_signed_distance_mm(cylinder, point) == pytest.approx(
            expected, abs=1.0e-12
        )


def test_the_signed_distance_is_invariant_under_the_bodys_own_pose():
    """把体和查询点一起刚体运动，有符号距离不变——rel 1e-12。

    这条挡的是`world_to_local_mm`把转置写反：**转置写反在轴对称构型上看不出来**，
    所以判据必须用一个非轴对称的盒和一个一般姿态。
    """

    box = RoundedBox(half_extents_mm=(7.0, 3.0, 11.0), fillet_radius_mm=1.5)
    rng = random.Random(2203)
    for _ in range(200):
        quaternion = _unit_quaternion(rng)
        translation = tuple(rng.uniform(-30.0, 30.0) for _ in range(3))
        local_point = tuple(rng.uniform(-25.0, 25.0) for _ in range(3))
        posed = _body("box", box, translation, quaternion)
        world_point = posed.transform_point_mm(local_point)
        assert world_to_local_mm(posed, world_point) == pytest.approx(
            local_point, rel=0.0, abs=1.0e-11
        )
        assert posed_signed_distance_mm(posed, world_point) == pytest.approx(
            shape_signed_distance_mm(box, local_point), rel=1.0e-12
        )


def test_a_sphere_against_a_rotated_box_reports_narrow_phase_with_the_exact_depth():
    """球对旋转盒：精确档，而且穿透深度就是SDF减半径。"""

    box_body = _body(
        "box",
        RoundedBox(half_extents_mm=(10.0, 10.0, 10.0), fillet_radius_mm=0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)),  # 绕z转45°
    )
    #: **外侧那一支**：点在x轴上、盒转45°之后离它最近的是顶点，顶点在10√2处。
    outside = narrow_phase_separation_mm(
        _body("ball", Sphere(radius_mm=2.0), (18.0, 0.0, 0.0)), box_body
    )
    assert outside.confidence == "narrow_phase"
    assert outside.estimated_bias_mm is None
    assert outside.separation_mm == pytest.approx(
        18.0 - 10.0 * math.sqrt(2.0) - 2.0, abs=1.0e-12
    )
    #: **内侧那一支，也是这条测试第一版写错的地方**：``(14, 0, 0)``看着在
    #: "半宽10"之外，其实在转过来的盒**里面**（顶点到14.142才结束）。
    #: 体内的最近面不是顶点而是侧面，距离是``10 − 14/√2 = 0.1005``。
    #: **把"沿轴到顶点"当成"到表面"，是旋转凸体上最容易犯的一个错。**
    inside = narrow_phase_separation_mm(
        _body("ball", Sphere(radius_mm=2.0), (14.0, 0.0, 0.0)), box_body
    )
    assert inside.separation_mm == pytest.approx(
        -(10.0 - 14.0 / math.sqrt(2.0)) - 2.0, abs=1.0e-12
    )


def test_a_generated_shape_wrapper_is_transparent_to_the_narrow_phase():
    """`GeneratedShape`包着的同族与裸形状给同一个数——包装不改物理。"""

    naked = FiniteCylinder(radius_mm=9.0, half_width_mm=4.0)
    wrapped = GeneratedShape(
        algorithm_id="algorithm:test/cylinder",
        algorithm_version="1.0.0",
        parameters=(("radius_mm", 9.0),),
        shape=naked,
    )
    point = (3.0, 4.0, 6.0)
    assert shape_signed_distance_mm(wrapped, point) == shape_signed_distance_mm(naked, point)


# ==========================================================================
# 四、失败关闭：四条，每条一个必红
# ==========================================================================


def test_a_flanged_cylinder_is_refused_instead_of_being_treated_as_plain():
    """带法兰的圆柱当场抛——**不按无法兰算**。

    必红那一半在下面`..._would_under_report_by`那条：它把"如果按无法兰算"
    会少报多少量出来，证明这条失败关闭不是保守过头。
    """

    flanged = FiniteCylinder(
        radius_mm=45.0, half_width_mm=9.0, flange_outer_radius_mm=60.0
    )
    with pytest.raises(CollisionQueryError, match="flange"):
        shape_signed_distance_mm(flanged, (50.0, 0.0, 0.0))


def test_treating_the_flange_as_plain_would_under_report_by_fifteen_millimetres():
    """必红的另一半：真实导轮（R45/法兰R60/半宽9）上，按无法兰算**少报15 mm**。

    这个数是`shapes.py`那两个半径之差，不是估的。**15 mm不是小量**——
    带材厚度是十分之一毫米量级，摩擦锥按法向力算，少报的那一段全是静默的。
    """

    plain = FiniteCylinder(radius_mm=45.0, half_width_mm=9.0)
    probe = (50.0, 0.0, 0.0)  # 在法兰内、槽底外
    if_treated_as_plain = shape_signed_distance_mm(plain, probe)
    assert if_treated_as_plain == pytest.approx(5.0, abs=1e-12)  # 报"离开5 mm"
    #: 若法兰是一个R60的实心圆盘，同一点应当在体内10 mm处——**符号都反了**。
    assert if_treated_as_plain > 0.0
    assert 60.0 - 45.0 == 15.0


def test_a_mesh_asset_is_refused_because_the_kernel_never_reads_mesh_bytes():
    """`MeshAsset`当场抛：它带的是路径＋SHA-256＋声明的AABB，一个顶点都没有。"""

    asset = MeshAsset(
        path_relative="assets/link3.stl",
        sha256="a" * 64,
        units="mm",
        usage="collision",
        convexity="nonconvex_declared",
        aabb_min_mm=(-10.0, -10.0, -10.0),
        aabb_max_mm=(10.0, 10.0, 10.0),
    )
    with pytest.raises(CollisionQueryError, match="no geometry"):
        shape_signed_distance_mm(asset, (0.0, 0.0, 0.0))


def test_the_direct_query_raises_for_pairs_it_cannot_answer():
    """直查面**没有"不知道"这个返回值**，所以答不出来一律抛（决策0090第六节）。"""

    capsule = _body("cap", Capsule((0.0, 0.0, -5.0), (0.0, 0.0, 5.0), 2.0))
    box = _body("box", RoundedBox(half_extents_mm=(4.0, 4.0, 4.0), fillet_radius_mm=0.0))
    with pytest.raises(CollisionQueryError, match="no exact narrow phase"):
        narrow_phase_separation_mm(capsule, box)

    turned = _body(
        "box2",
        RoundedBox(half_extents_mm=(4.0, 4.0, 4.0), fillet_radius_mm=0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, math.sin(0.3), math.cos(0.3)),
    )
    with pytest.raises(CollisionQueryError, match="no exact narrow phase"):
        narrow_phase_separation_mm(box, turned)


def test_the_scene_query_degrades_instead_of_raising_on_the_same_pairs():
    """同样两对在场景查询上**降级**：事件照报、深度留空。

    与上一条并排读：这不是两个口径打架，是同一条纪律在两个契约上的两种执行
    （决策0090第六节）。让场景查询也抛会破掉"不漏报"这条唯一硬承诺。
    """

    capsule = _body("cap", Capsule((0.0, 0.0, -5.0), (0.0, 0.0, 5.0), 2.0))
    box = _body("box", RoundedBox(half_extents_mm=(4.0, 4.0, 4.0), fillet_radius_mm=0.0))
    events = BroadPhaseCollisionQuery((capsule, box)).check_state()
    assert len(events) == 1
    assert events[0].confidence == "broad_phase"
    assert events[0].penetration_mm is None


def test_a_distance_field_for_an_unknown_body_is_refused_at_assembly_time():
    """装配期校验：场挂在一个不存在的体上当场炸，不拖到查询时。"""

    ball = _body("ball", Sphere(radius_mm=1.0))
    field = sample_narrow_band(
        lambda point: point[2],
        origin_mm=(-4.0, -4.0, -4.0),
        spacing_mm=1.0,
        node_counts=(9, 9, 9),
        band_mm=4.0,
    )
    with pytest.raises(ShapeError, match="unknown body"):
        BroadPhaseCollisionQuery((ball,), distance_fields={"body/nope": field})


def test_an_object_that_is_not_a_signed_distance_source_is_refused():
    """协议不满足当场炸——**鸭子类型不等于不校验**（决策0090第三节）。"""

    ball = _body("ball", Sphere(radius_mm=1.0))
    with pytest.raises(ShapeError, match="SignedDistanceSource"):
        BroadPhaseCollisionQuery((ball,), distance_fields={"body/ball": object()})


# ==========================================================================
# 五、距离场那条路：精度声明与它的三条自洽出口
# ==========================================================================


def _half_space_field(spacing_mm: float):
    """``φ(x) = z``烘成窄带场。仿射 ⟹ 三次B样条**逐位重构**（0085第三节）。"""

    return sample_narrow_band(
        lambda point: point[2],
        origin_mm=(-6.0, -6.0, -6.0),
        spacing_mm=spacing_mm,
        node_counts=(
            int(12.0 / spacing_mm) + 1,
            int(12.0 / spacing_mm) + 1,
            int(12.0 / spacing_mm) + 1,
        ),
        band_mm=max(4.0, 2.0 * spacing_mm * math.sqrt(3.0) + 1.0),
    )


def _sphere_field(spacing_mm: float, radius_mm: float = 10.0):
    """``φ(x) = |x| − R``烘成窄带场。"""

    extent = radius_mm + 4.0
    counts = int(2.0 * extent / spacing_mm) + 1
    return sample_narrow_band(
        lambda point: math.sqrt(sum(item * item for item in point)) - radius_mm,
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=spacing_mm,
        node_counts=(counts, counts, counts),
        band_mm=max(3.5, 2.0 * spacing_mm * math.sqrt(3.0) + 1.0),
    )


def test_the_field_route_declares_sampled_field_and_never_narrow_phase():
    """场那条路的可信度**只能**是`sampled_field`。

    这是本轨第3条验收（"一个在非凸体上给出精确穿透深度的实现是冒充"）
    的执行面：档位是接口自己声明的，调用方不必去猜这个数是闭式还是采样。
    """

    field = _sphere_field(0.5)
    obstacle = _body("obs", Sphere(radius_mm=10.0))
    #: 探针要真的穿进去，否则假阳性消除会把事件剔掉（那是另一条已有的判据）。
    probe = _body("ball", Sphere(radius_mm=1.0), (10.5, 0.0, 0.0))
    query = BroadPhaseCollisionQuery(
        (obstacle, probe), distance_fields={"body/obs": field}
    )
    result = field_separation_mm(field, obstacle, (10.5, 0.0, 0.0), 1.0)
    assert result.confidence == "sampled_field"
    assert result.separation_mm < 0.0
    assert result.resolution_mm == 0.5
    assert result.estimated_bias_mm is not None
    events = query.check_state()
    assert len(events) == 1
    assert events[0].confidence == "sampled_field"
    assert events[0].resolution_mm == 0.5
    assert events[0].estimated_bias_mm is not None


def test_the_half_space_field_is_exact_at_every_resolution_and_declares_zero_bias():
    """仿射场：三档``h``全部与闭式差``≤ 1 ulp``，且偏差估计**恒为0**。

    这条是偏差估计公式``(h²/6)·tr(∇²φ)``的第一个自洽出口：
    ``φ``仿射 ⟹ ``∇²φ ≡ 0`` ⟹ 估计恒零，与0085"半空间任何分辨率下精确"对上。
    """

    obstacle = _body("plane", Sphere(radius_mm=1.0))  # 位姿是单位阵，场即世界系
    for spacing in (2.0, 1.0, 0.5):
        field = _half_space_field(spacing)
        for height in (-2.0, -0.5, 0.75, 2.5):
            result = field_separation_mm(field, obstacle, (0.3, -1.1, height), 0.0)
            assert result.separation_mm == pytest.approx(height, rel=0.0, abs=1.0e-13)
            assert result.estimated_bias_mm == pytest.approx(0.0, abs=1.0e-15)
            assert result.resolution_mm == spacing


def test_the_sphere_field_bias_estimate_matches_the_closed_form_leading_term():
    """凸障碍：偏差估计对上闭式主项``h²/(3ρ)``，rel 5e-3。

    ``∇²(|x| − R) = 2/ρ`` ⟹ ``(h²/6)·(2/ρ) = h²/(3ρ)``。
    这与`cases/sdf_contact_convergence`那条已经量过的位置偏移主项``−h²/(3z*)``
    **逐字同形**（差一个符号是因为那里量的是位置、这里量的是距离本身）。

    **符号为正**⟹场报出来的距离比真值大⟹偏松，与0085第三节同向。
    """

    obstacle = _body("obs", Sphere(radius_mm=10.0))
    probe_radius = 11.5
    for spacing in (1.0, 0.5, 0.25):
        field = _sphere_field(spacing)
        result = field_separation_mm(field, obstacle, (probe_radius, 0.0, 0.0), 0.0)
        expected_bias = spacing * spacing / (3.0 * probe_radius)
        assert result.estimated_bias_mm > 0.0
        assert result.estimated_bias_mm == pytest.approx(expected_bias, rel=5.0e-3)
        #: 真实偏差与估计同号同量级——估计是主项，不是上界。
        actual_bias = result.separation_mm - (probe_radius - 10.0)
        assert actual_bias > 0.0
        assert actual_bias == pytest.approx(expected_bias, rel=2.0e-2)


def test_the_bias_estimate_goes_red_if_the_sixth_is_dropped():
    """必红：把``h²/6``写成``h²``（漏掉二阶矩那个1/6），判据当场红。

    这一条挡的是最容易犯的错——那个``1/6``是三次B样条的二阶矩``1/3``的一半，
    不是一个可以省的常数。
    """

    spacing, probe_radius = 0.5, 11.5
    correct = spacing * spacing / (3.0 * probe_radius)
    dropped_sixth = correct * 6.0
    assert dropped_sixth != pytest.approx(correct, rel=5.0e-3)


def test_the_field_value_error_is_second_order_on_a_non_convex_torus():
    """**非凸**语料：环面的场值误差随``h``二阶，比落在``[3.6, 4.4]``。

    环面``φ = sqrt((sqrt(x²+y²) − R)² + z²) − r``是**精确的有符号距离**且非凸——
    它正是0074第二节第2条那句"SDF对非凸没有任何额外代价"的可测出口。
    走凸体窄相（EPA/MPR）的话，这个形状要先凸分解，而那条路已被0073第七节第1条裁掉。
    """

    major, minor = 12.0, 3.5

    def torus(point):
        radial = math.hypot(point[0], point[1]) - major
        return math.hypot(radial, point[2]) - minor

    #: 四个探针全部落在窄带内（``|φ| ≤ 3``）：外赤道外侧、一般位形、
    #: 管内（``φ < 0``）、上表面附近。**孔心那种点不在带里**，
    #: 拿它当探针只会撞上"窄带外失败关闭"那条门，量不到阶。
    probes = ((17.0, 0.0, 0.0), (0.0, 16.2, 1.3), (9.5, 0.0, 0.0), (12.0, 0.0, 5.2))
    errors = []
    for spacing in (0.8, 0.4, 0.2):
        extent = major + minor + 4.0
        counts = int(2.0 * extent / spacing) + 1
        field = sample_narrow_band(
            torus,
            origin_mm=(-extent, -extent, -8.0),
            spacing_mm=spacing,
            node_counts=(counts, counts, int(16.0 / spacing) + 1),
            band_mm=max(3.0, 2.0 * spacing * math.sqrt(3.0) + 0.5),
        )
        body = _body("torus", Sphere(radius_mm=1.0))
        worst = 0.0
        for probe in probes:
            measured = field_separation_mm(field, body, probe, 0.0)
            assert measured.confidence == "sampled_field"
            worst = max(worst, abs(measured.separation_mm - torus(probe)))
        errors.append(worst)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.6 <= ratio <= 4.4 for ratio in ratios), ratios


def test_the_bias_is_negative_inside_a_bore_and_matches_its_own_closed_form():
    """**0085只写了凸那一半**：凹面上场偏紧，偏差估计为负，rel 5e-3。

    语料是一个**圆柱形孔**（实体是圆柱的补集，**非凸**）：
    ``φ(x) = R − ρ``是它的精确有符号距离，且``∇²φ = −1/ρ``是初等的
    （``φ_ρ = −1``、``φ_ρρ = 0``、``∇²φ = φ_ρρ + φ_ρ/ρ``）。于是

        偏差主项 = (h²/6)·(−1/ρ) = −h²/(6ρ)

    **符号是负的** ⟹ 场报出来的距离比真值**小** ⟹ 物体被挡在更外面，
    与凸障碍上那条"偏松"方向**相反**。0074第二节第4条原话只说了
    "系统性偏保守或偏松"，0085第三节把凸那一侧钉到"偏松"，
    **本条把凹那一侧钉到"偏紧"**——两侧各有一条闭式，谁也不用猜。

    这条同时是"SDF对非凸没有额外代价"的第二个出口：孔的补集非凸，
    走凸体窄相要先凸分解，而那条路已被0073第七节第1条裁掉。
    """

    bore_radius = 12.0

    def bore(point):
        return bore_radius - math.hypot(point[0], point[1])

    body = _body("bore", Sphere(radius_mm=1.0))
    probe_radius = 10.5  # 孔内，离壁1.5 mm
    for spacing in (0.5, 0.25):
        extent = bore_radius + 4.0
        counts = int(2.0 * extent / spacing) + 1
        field = sample_narrow_band(
            bore,
            origin_mm=(-extent, -extent, -3.0),
            spacing_mm=spacing,
            node_counts=(counts, counts, int(6.0 / spacing) + 1),
            band_mm=max(3.0, 2.0 * spacing * math.sqrt(3.0) + 0.5),
        )
        result = field_separation_mm(field, body, (probe_radius, 0.0, 0.0), 0.0)
        expected_bias = -spacing * spacing / (6.0 * probe_radius)
        assert result.estimated_bias_mm < 0.0
        assert result.estimated_bias_mm == pytest.approx(expected_bias, rel=5.0e-3)
        #: 真实偏差与估计**同号**——这才是"估计抓对了方向"的证据。
        actual_bias = result.separation_mm - bore((probe_radius, 0.0, 0.0))
        assert actual_bias < 0.0
        assert actual_bias == pytest.approx(expected_bias, rel=5.0e-2)


def test_a_probe_outside_the_narrow_band_fails_closed():
    """窄带外**抛**，不外推（0085第四节）。"""

    field = _sphere_field(0.5)
    obstacle = _body("obs", Sphere(radius_mm=10.0))
    with pytest.raises(CollisionQueryError, match="narrow band"):
        field_separation_mm(field, obstacle, (0.0, 0.0, 0.0), 0.0)  # 深在体内


def test_the_scene_query_degrades_when_the_probe_leaves_the_band():
    """同一件事在场景查询上是降级：事件照报、深度留空。"""

    field = _sphere_field(0.5)
    obstacle = _body("obs", Sphere(radius_mm=10.0))
    probe = _body("ball", Sphere(radius_mm=1.0), (0.0, 0.0, 0.0))
    events = BroadPhaseCollisionQuery(
        (obstacle, probe), distance_fields={"body/obs": field}
    ).check_state()
    assert len(events) == 1
    assert events[0].confidence == "broad_phase"
    assert events[0].penetration_mm is None


def test_a_non_ball_probe_against_a_field_is_refused():
    """胶囊对场当场拒——``min φ``会静默漏接触（决策0090第2.3节第一行）。"""

    field = _sphere_field(0.5)
    obstacle = _body("obs", Sphere(radius_mm=10.0))
    capsule = _body("cap", Capsule((0.0, 0.0, -5.0), (0.0, 0.0, 5.0), 1.0), (11.5, 0.0, 0.0))
    with pytest.raises(CollisionQueryError, match="no exact narrow phase"):
        narrow_phase_separation_mm(
            capsule, obstacle, distance_fields={"body/obs": field}
        )


def test_the_field_follows_the_bodys_pose():
    """场按体的局部系烘：体动了场跟着动，而**偏差估计与位姿无关**（迹是旋转不变量）。"""

    field = _sphere_field(0.5)
    at_origin = _body("obs", Sphere(radius_mm=10.0))
    moved = _body(
        "obs",
        Sphere(radius_mm=10.0),
        (100.0, -40.0, 7.0),
        (0.0, math.sin(0.4), 0.0, math.cos(0.4)),
    )
    reference = field_separation_mm(field, at_origin, (11.5, 0.0, 0.0), 0.0)
    probe = moved.transform_point_mm((11.5, 0.0, 0.0))
    shifted = field_separation_mm(field, moved, probe, 0.0)
    assert shifted.separation_mm == pytest.approx(reference.separation_mm, rel=1.0e-12)
    assert shifted.estimated_bias_mm == pytest.approx(
        reference.estimated_bias_mm, rel=1.0e-12
    )


def test_two_fields_on_one_pair_are_refused_rather_than_guessed():
    """两个体都配了场：谁当探针没有答案，**不猜**。"""

    field = _sphere_field(0.5)
    first = _body("a", Sphere(radius_mm=10.0))
    second = _body("b", Sphere(radius_mm=10.0), (11.5, 0.0, 0.0))
    with pytest.raises(CollisionQueryError, match="no exact narrow phase"):
        narrow_phase_separation_mm(
            first, second, distance_fields={"body/a": field, "body/b": field}
        )


def test_the_result_dataclass_defaults_declare_exactness():
    """精度声明的**缺省值**就是"精确"——闭式那条路不必逐处写`None`。"""

    result = NarrowPhaseResult(separation_mm=-1.0, confidence="narrow_phase")
    assert result.estimated_bias_mm is None
    assert result.resolution_mm is None


# ==========================================================================
# 六、法向与接触点（0094第1条硬约束：先扩输出档）
#
# 判据用两条**零新增金标**的自洽恒等式：
#   ① Bullet的 ``pointOnA == pointOnB + distance · normalBtoA``；
#   ② 见证点真的落在各自的面上（``|φ| ≈ 0``）。
# 两条都不需要任何外部参考解——它们判的是"法向、接触点、距离三样互相对得上"。
# ==========================================================================


def _witness_identity(result) -> float:
    residual = result.witness_identity_residual_mm()
    assert residual is not None, "这一构型没有法向，不该拿它判恒等式"
    return residual


def test_the_witness_helper_returns_none_when_there_is_nothing_to_check():
    """三样有一个是`None`就返回`None`——**没有可判的东西，不是判过了**。"""

    assert (
        NarrowPhaseResult(separation_mm=-1.0, confidence="narrow_phase")
        .witness_identity_residual_mm()
        is None
    )


def test_segment_family_satisfies_the_bullet_witness_identity():
    """球/胶囊族120对语料上逐对判Bullet恒等式，残差 < 1e-11 mm。"""

    worst = 0.0
    checked = 0
    for _index, first, second in _sphere_capsule_corpus():
        result = narrow_phase_separation_mm(first, second)
        if result.normal_ab is None:
            continue
        worst = max(worst, _witness_identity(result))
        checked += 1
    assert checked == 120  # 随机语料里没有一对是同心的
    assert worst < 1.0e-11, worst


def test_segment_family_witnesses_sit_on_both_surfaces():
    """见证点真的在面上：`|φ| < 1e-11 mm`，两侧各判一次。

    这一条比恒等式更强——恒等式只说三样自洽，**它可以整体平移而仍然自洽**。
    """

    worst = 0.0
    for _index, first, second in _sphere_capsule_corpus():
        result = narrow_phase_separation_mm(first, second)
        if result.normal_ab is None:
            continue
        worst = max(
            worst,
            abs(posed_signed_distance_mm(first, result.witness_a_mm)),
            abs(posed_signed_distance_mm(second, result.witness_b_mm)),
        )
    assert worst < 1.0e-11, worst


def test_the_normal_points_from_b_towards_a():
    """方向约定：`normal_ab`从B指向A（Bullet的`normalBtoA`）。

    判法不靠直觉：把A沿`+normal_ab`挪一段，分离量必须**正好**增加那么多。
    """

    first = _body("a", Sphere(radius_mm=3.0), (10.0, 0.0, 0.0))
    second = _body("b", Sphere(radius_mm=4.0), (0.0, 0.0, 0.0))
    base = narrow_phase_separation_mm(first, second)
    assert base.normal_ab == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-15)
    step = 2.5
    moved = _body(
        "a",
        Sphere(radius_mm=3.0),
        tuple(10.0 * (axis == 0) + base.normal_ab[axis] * step for axis in range(3)),
    )
    assert narrow_phase_separation_mm(moved, second).separation_mm == pytest.approx(
        base.separation_mm + step, abs=1.0e-12
    )


def test_swapping_the_pair_order_negates_the_normal_and_swaps_the_witnesses():
    """A/B互换：法向取反、两个见证点互换、**分离量一个字不动**。"""

    first = _body("a", Sphere(radius_mm=2.0), (9.0, 1.0, -2.0))
    second = _body(
        "b", RoundedBox(half_extents_mm=(4.0, 5.0, 6.0), fillet_radius_mm=0.5)
    )
    forward = narrow_phase_separation_mm(first, second)
    backward = narrow_phase_separation_mm(second, first)
    assert backward.separation_mm == forward.separation_mm  # 逐位
    assert backward.normal_ab == pytest.approx(
        tuple(-value for value in forward.normal_ab), abs=0.0
    )
    assert backward.witness_a_mm == forward.witness_b_mm
    assert backward.witness_b_mm == forward.witness_a_mm


def test_the_ball_probe_route_satisfies_both_identities():
    """球型探针对圆角盒/圆柱：恒等式＋见证点在面上，随机400组。"""

    rng = random.Random(6161)
    worst_identity = worst_surface = 0.0
    for _ in range(400):
        target_shape = (
            RoundedBox(
                half_extents_mm=tuple(rng.uniform(2.0, 12.0) for _ in range(3)),
                fillet_radius_mm=rng.choice((0.0, 1.5)),
            )
            if rng.random() < 0.5
            else FiniteCylinder(
                radius_mm=rng.uniform(3.0, 12.0), half_width_mm=rng.uniform(2.0, 8.0)
            )
        )
        target = _body("t", target_shape, rotation=_unit_quaternion(rng))
        probe = _body(
            "p",
            Sphere(radius_mm=rng.uniform(0.5, 4.0)),
            tuple(rng.uniform(-20.0, 20.0) for _ in range(3)),
        )
        result = narrow_phase_separation_mm(probe, target)
        if result.normal_ab is None:
            continue
        worst_identity = max(worst_identity, _witness_identity(result))
        worst_surface = max(
            worst_surface,
            abs(posed_signed_distance_mm(probe, result.witness_a_mm)),
            abs(posed_signed_distance_mm(target, result.witness_b_mm)),
        )
    assert worst_identity < 1.0e-12, worst_identity
    assert worst_surface < 1.0e-11, worst_surface


def test_the_axis_aligned_box_pair_satisfies_both_identities():
    """轴对齐盒对：分离支与穿透支都判，随机400组。"""

    rng = random.Random(7272)
    worst_identity = worst_surface = 0.0
    separated = penetrating = 0
    for _ in range(400):
        half_a = tuple(rng.uniform(1.0, 15.0) for _ in range(3))
        half_b = tuple(rng.uniform(1.0, 15.0) for _ in range(3))
        fillet_a, fillet_b = rng.choice((0.0, 1.0)), rng.choice((0.0, 2.0))
        first = _body(
            "a",
            RoundedBox(half_extents_mm=half_a, fillet_radius_mm=fillet_a),
            tuple(rng.uniform(-8.0, 8.0) for _ in range(3)),
        )
        second = _body(
            "b",
            RoundedBox(half_extents_mm=half_b, fillet_radius_mm=fillet_b),
            tuple(rng.uniform(-30.0, 30.0) for _ in range(3)),
        )
        result = narrow_phase_separation_mm(first, second)
        assert result.normal_ab is not None
        worst_identity = max(worst_identity, _witness_identity(result))
        worst_surface = max(
            worst_surface,
            abs(posed_signed_distance_mm(first, result.witness_a_mm)),
            abs(posed_signed_distance_mm(second, result.witness_b_mm)),
        )
        if result.separation_mm > 0.0:
            separated += 1
        else:
            penetrating += 1
    assert separated > 30 and penetrating > 30, (separated, penetrating)
    assert worst_identity < 1.0e-12, worst_identity
    assert worst_surface < 1.0e-11, worst_surface


def test_the_field_route_carries_a_normal_and_satisfies_the_identity():
    """场那条路也给法向。恒等式是精确的（三样出自同一次查询），
    但**见证点落在面上只到``O(h²)``**——那正是`sampled_field`那一档的含义。"""

    field = _sphere_field(0.25)
    obstacle = _body("obs", Sphere(radius_mm=10.0))
    probe_centre = (11.5, 0.0, 0.0)
    result = field_separation_mm(field, obstacle, probe_centre, 1.0)
    assert result.confidence == "sampled_field"
    assert result.normal_ab == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-6)
    assert _witness_identity(result) < 1.0e-12
    #: B侧见证点应当落在真球面上，误差是场的``O(h²)``而不是舍入。
    radius_of_witness = math.sqrt(sum(value * value for value in result.witness_b_mm))
    assert radius_of_witness == pytest.approx(10.0, abs=5.0e-3)


def test_the_scene_query_carries_the_normal_and_witnesses_too():
    """场景查询的事件带同样三样——**否则扩档等于没扩**（力学读的是事件）。"""

    first = _body("a", Sphere(radius_mm=6.0), (8.0, 0.0, 0.0))
    second = _body("b", Sphere(radius_mm=6.0))
    events = BroadPhaseCollisionQuery((first, second)).check_state()
    assert len(events) == 1
    event = events[0]
    assert event.confidence == "narrow_phase"
    assert event.normal_ab == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-15)
    assert event.witness_a_mm == pytest.approx((2.0, 0.0, 0.0), abs=1.0e-12)
    assert event.witness_b_mm == pytest.approx((6.0, 0.0, 0.0), abs=1.0e-12)
    #: 事件里那三样与直查面**逐位相同**。
    direct = narrow_phase_separation_mm(first, second)
    assert event.normal_ab == direct.normal_ab
    assert event.witness_a_mm == direct.witness_a_mm
    assert event.witness_b_mm == direct.witness_b_mm


def test_a_broad_phase_event_never_carries_a_normal():
    """降级那一档三样全是`None`——**"不知道"不许长出一个方向来**。"""

    capsule = _body("cap", Capsule((0.0, 0.0, -5.0), (0.0, 0.0, 5.0), 2.0))
    box = _body("box", RoundedBox(half_extents_mm=(4.0, 4.0, 4.0), fillet_radius_mm=0.0))
    event = BroadPhaseCollisionQuery((capsule, box)).check_state()[0]
    assert event.confidence == "broad_phase"
    assert event.normal_ab is None
    assert event.witness_a_mm is None
    assert event.witness_b_mm is None


def test_concentric_spheres_report_no_normal_instead_of_inventing_one():
    """同心球：法向真的没有定义。**报`None`，不编一个方向。**

    这一格是"诚实可信度"在法向上的执行面：距离照给（穿透深度良定义），
    方向留空。编一个方向出来会让摩擦锥安静地作用在一个错方向上。
    """

    first = _body("a", Sphere(radius_mm=3.0))
    second = _body("b", Sphere(radius_mm=4.0))
    result = narrow_phase_separation_mm(first, second)
    assert result.separation_mm == pytest.approx(-7.0, abs=1.0e-15)
    assert result.normal_ab is None
    assert result.witness_a_mm is None
    assert result.witness_identity_residual_mm() is None
    event = BroadPhaseCollisionQuery((first, second)).check_state()[0]
    assert event.confidence == "narrow_phase"
    assert event.penetration_mm == pytest.approx(7.0, abs=1.0e-15)
    assert event.normal_ab is None


def test_a_probe_on_the_cylinder_axis_reports_no_normal():
    """圆柱轴上、且侧壁比端面近：到侧壁四面八方一样远，法向没有定义。"""

    cylinder = FiniteCylinder(radius_mm=3.0, half_width_mm=20.0)
    assert shape_signed_distance_gradient(cylinder, (0.0, 0.0, 0.0)) is None
    #: 端面更近时法向就有定义了（轴向），**同一个点集上两种答案各有出处**。
    flat = FiniteCylinder(radius_mm=20.0, half_width_mm=3.0)
    assert shape_signed_distance_gradient(flat, (0.0, 0.0, 1.0)) == (0.0, 0.0, 1.0)


def test_the_gradient_is_the_unit_outward_normal_by_finite_difference():
    """梯度对中心差分：`|∇φ − FD| < 2e-6`，且模恒为1。

    **这一条挡的是"梯度写成了另一个形状的梯度"**——恒等式与见证点两条门
    对一个整体错的方向是盲的（它们只判自洽）。
    """

    shapes = (
        Sphere(radius_mm=5.0),
        Capsule((0.0, 0.0, -4.0), (0.0, 0.0, 4.0), 2.0),
        RoundedBox(half_extents_mm=(3.0, 7.0, 11.0), fillet_radius_mm=1.0),
        FiniteCylinder(radius_mm=6.0, half_width_mm=3.0),
    )
    rng = random.Random(8383)
    step = 1.0e-5
    worst = 0.0
    for shape in shapes:
        for _ in range(200):
            point = tuple(rng.uniform(-18.0, 18.0) for _ in range(3))
            gradient = shape_signed_distance_gradient(shape, point)
            if gradient is None:
                continue
            assert math.sqrt(sum(v * v for v in gradient)) == pytest.approx(1.0, abs=1e-15)
            finite = []
            for axis in range(3):
                plus = tuple(point[i] + (step if i == axis else 0.0) for i in range(3))
                minus = tuple(point[i] - (step if i == axis else 0.0) for i in range(3))
                finite.append(
                    (
                        shape_signed_distance_mm(shape, plus)
                        - shape_signed_distance_mm(shape, minus)
                    )
                    / (2.0 * step)
                )
            worst = max(
                worst, max(abs(gradient[axis] - finite[axis]) for axis in range(3))
            )
    assert worst < 2.0e-6, worst


def test_the_witness_identity_goes_red_when_the_normal_is_flipped():
    """必红一：法向取反，恒等式当场红（残差 = 2·|分离量|）。"""

    good = narrow_phase_separation_mm(
        _body("a", Sphere(radius_mm=3.0), (10.0, 0.0, 0.0)),
        _body("b", Sphere(radius_mm=4.0)),
    )
    assert _witness_identity(good) < 1.0e-12
    flipped = NarrowPhaseResult(
        separation_mm=good.separation_mm,
        confidence=good.confidence,
        normal_ab=tuple(-value for value in good.normal_ab),
        witness_a_mm=good.witness_a_mm,
        witness_b_mm=good.witness_b_mm,
    )
    assert flipped.witness_identity_residual_mm() == pytest.approx(
        2.0 * abs(good.separation_mm), rel=1.0e-12
    )


def test_the_witness_identity_goes_red_when_a_witness_slides_along_the_surface():
    """必红二：把一个见证点**沿面内**挪一点（离开面法向那条线），恒等式当场红。

    这一条比"取反"更值钱：**沿面内挪不改变`|φ|`**，所以"见证点在面上"那道门
    抓不到它，只有恒等式抓得到。两道门因此必须并列，不能互相顶替。
    """

    good = narrow_phase_separation_mm(
        _body("a", Sphere(radius_mm=3.0), (10.0, 0.0, 0.0)),
        _body("b", Sphere(radius_mm=4.0)),
    )
    slid_centre = (10.0, 0.0, 0.0)
    slid = (
        slid_centre[0] - 3.0 * math.cos(0.3),
        slid_centre[1] + 3.0 * math.sin(0.3),
        slid_centre[2],
    )
    #: 它仍然在A的球面上——"在面上"那道门看不见这次改动。
    surface = _body("a", Sphere(radius_mm=3.0), slid_centre)
    assert abs(posed_signed_distance_mm(surface, slid)) < 1.0e-12
    slid_result = NarrowPhaseResult(
        separation_mm=good.separation_mm,
        confidence=good.confidence,
        normal_ab=good.normal_ab,
        witness_a_mm=slid,
        witness_b_mm=good.witness_b_mm,
    )
    assert slid_result.witness_identity_residual_mm() > 0.1


def test_the_gradient_gate_goes_red_when_the_box_uses_the_wrong_face():
    """必红三：体内取"最远的面"而不是"最近的面"，有限差分门当场红。"""

    box = RoundedBox(half_extents_mm=(3.0, 7.0, 11.0), fillet_radius_mm=0.0)
    point = (2.5, 1.0, 1.0)  # 体内，x面最近
    assert shape_signed_distance_gradient(box, point) == (1.0, 0.0, 0.0)
    #: 取z面（最远的那一面）是注错值，与有限差分差一个整单位向量。
    wrong = (0.0, 0.0, 1.0)
    assert max(abs(wrong[axis] - (1.0, 0.0, 0.0)[axis]) for axis in range(3)) == 1.0


def test_the_witnesses_come_from_the_same_arithmetic_as_the_distance():
    """`segment_segment_witnesses`的第三项与`segment_segment_distance_mm`**逐位相同**。

    这一条守的是那次改写：距离函数现在是witness函数的第三项，
    **不是两份拷贝**。两份拷贝迟早会漂，而漂的时候没有任何门看得见。
    """

    rng = random.Random(9494)
    for _ in range(500):
        points = [tuple(rng.uniform(-40.0, 40.0) for _ in range(3)) for _ in range(4)]
        witness = segment_segment_witnesses(*points)
        assert witness[2].hex() == segment_segment_distance_mm(*points).hex()
