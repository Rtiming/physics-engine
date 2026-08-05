"""同行库对拍的算子层：输入生成、两侧适配、逐点比对。

**身份声明**：本文件住在`validation/`，它import`fcl`；`src/physics_engine/`
永远不import本文件，本文件也永远不进`dependencies`（0015张力B、决策0025）。

## 一、两侧的语义映射（对拍成立的前提，全部实测确定，勿凭印象改）

| 项 | 本仓 | FCL（python-fcl 0.7.0.11） |
|---|---|---|
| 胶囊定义 | `Capsule(point_a_mm, point_b_mm, radius_mm)`，中轴线段给任意两端点 | `fcl.Capsule(radius, lz)`，中轴线段固定为z轴上`[-lz/2, +lz/2]`，**总长`lz+2r`** |
| 四元数 | `rotation_xyzw`（标量在**后**） | `fcl.Transform(q, t)`的`q`是**wxyz**（标量在**前**）。实测：`setQuatRotation([cos45,sin45,0,0])`得绕x轴+90度的旋转矩阵 |
| 分离量 | 一个数：`segment_segment_distance_mm(...) − (r1+r2)`，相交时为负 | **没有单一对应物**，见第二节 |
| 接触判据 | `separation < 0`才报事件（**恰好相切不报**） | `fcl.collide`把恰好相切判为碰撞（`d ≤ 0`闭区间） |

## 二、FCL的三条路不是同一个算子（本次对拍最贵的一条发现）

1. `distance(enable_signed_distance=False)`——走**解析特化**。分离构型精确；
   球-球/球-胶囊相交时返回**哨兵`-1.0`**（不是深度！）；胶囊-胶囊相交时
   反而返回**真的负距离**（该对有带符号的解析实现）。
2. `distance(enable_signed_distance=True)`——走libccd。它**不只**在相交时接管，
   连分离构型的精度也一起降到绝对约1e-6（实测：胶囊端对端真值4.0，
   `signed=False`得4.0，`signed=True`得4.000000847465477；该绝对误差
   **与世界尺度无关**，在1e-3到1e3的五个量级上都是约1e-6）。
   **更糟的是它会大错**：见`KNOWN_FCL_DEFECTS`。
3. `collide(..., enable_contact=True)`的`contact.penetration_depth`——走
   解析接触生成。球-球与球-胶囊上与仓内Fraction精确解逐位或1e-13级吻合，
   **但胶囊-胶囊上会大错**（见`KNOWN_FCL_DEFECTS`第2条）。

因此本对拍的路由规则**只看FCL自己的返回值**，不看我们的数（否则对拍是循环的）：
第1条不是哨兵就用它（覆盖全部分离/擦边，外加胶囊-胶囊相交），是哨兵才退到第3条
（球-球与球-胶囊的相交）。第2条全程记录但**不作判据**——它是被验的对象之一。
哨兵歧义（真值恰为−1.0mm）在此规则下无害：那种情况下第3条也≈−1.0，两条路同答案。

## 三、输入生成为什么用stdlib的`random`

`numpy.random`会让"重放这批输入"本身依赖同行环境。用`random.Random(seed)`，
任何一个裸Python 3.11+都能重生成同一批输入，同行库只负责给它那一列数。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from physics_engine.collision import (
    BroadPhaseCollisionQuery,
    segment_segment_distance_mm,
)
from physics_engine.shapes import (
    Capsule,
    CollisionShape,
    PosedBody,
    SimBody,
    Sphere,
    Vector3,
)

PairClass = Literal["sphere_sphere", "sphere_capsule", "capsule_capsule"]
Band = Literal["separated", "grazing", "penetrating"]

PAIR_CLASSES: tuple[PairClass, ...] = ("sphere_sphere", "sphere_capsule", "capsule_capsule")
BANDS: tuple[Band, ...] = ("separated", "grazing", "penetrating")

#: 各band的目标分离量区间（mm，几何量按本仓mm制）。
#: `grazing`不是凑数——它是两侧接触判据分歧唯一可能的发生地，必须被采到。
BAND_TARGETS: dict[Band, tuple[float, float]] = {
    "separated": (1.0e-2, 3.0e2),
    "grazing": (1.0e-7, 1.0e-2),
    "penetrating": (-1.0, -1.0e-3),  # 实际下界按两体半径缩放，见`_sample_target`
}

#: FCL在unsigned路径上表示"相交"的哨兵值。它**不是距离**。
FCL_COLLISION_SENTINEL = -1.0

#: 已查实的两处同行缺陷。写进每份产物，与案例页"已知失效清单"第1、2条同源。
#: **它们被记录，但不作判据**——把同行的bug钉成门，下个版本修好了我们反而会红
#: （与spec/08规则1"实测数不作金标"同源）。
KNOWN_FCL_DEFECTS = (
    "1) fcl.distance(enable_signed_distance=True)（libccd路）会返回大错的距离，"
    "且不限于相交构型：本案例2700组里109组偏差>1e-5mm、663组>1e-6mm，最大3.42e-3mm；"
    "最坏一条是分离构型，相对误差45.6%。GST_LIBCCD与GST_INDEP给出逐位相同的错值，换solver无效。"
    " 2) fcl.collide().penetration_depth在胶囊-胶囊上会大错：900组里128组偏差>1e-5mm，最大2.11mm；"
    "而同一批输入下FCL自己的distance(unsigned)是对的——FCL在此自相矛盾。"
    "该算子在球-球（偏差0.0）与球-胶囊（1.09e-13mm）上精确。"
    " 两处都用fractions.Fraction精确算过真值，本仓是与真值一致的那一方。"
)


@dataclass(frozen=True)
class ShapeSpec:
    """一个被对拍的形状：本仓与FCL都能无损构造的最小描述。

    `half_length_mm == 0.0`表示球。FCL的`Capsule(r, 0)`与`Sphere(r)`几何等价，
    但我们**不**用它冒充球——球走`fcl.Sphere`，两家各用各的原语才是独立对拍。
    """

    radius_mm: float
    half_length_mm: float

    @property
    def is_sphere(self) -> bool:
        return self.half_length_mm == 0.0

    def local_segment(self) -> tuple[Vector3, Vector3]:
        h = self.half_length_mm
        return ((0.0, 0.0, -h), (0.0, 0.0, h))


@dataclass(frozen=True)
class Placement:
    translation_mm: Vector3
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class Sample:
    index: int
    pair_class: PairClass
    band: Band
    shape_a: ShapeSpec
    shape_b: ShapeSpec
    place_a: Placement
    place_b: Placement


# ---------------------------------------------------------------- 本仓侧


def sim_body(body_id: str, spec: ShapeSpec) -> SimBody:
    shape = (
        Sphere(radius_mm=spec.radius_mm)
        if spec.is_sphere
        else Capsule(
            point_a_mm=spec.local_segment()[0],
            point_b_mm=spec.local_segment()[1],
            radius_mm=spec.radius_mm,
        )
    )
    return SimBody(body_id=body_id, collision=CollisionShape(shape=shape, direction="fitted"))


def posed_body(body_id: str, spec: ShapeSpec, place: Placement) -> PosedBody:
    return PosedBody(
        body=sim_body(body_id, spec),
        translation_mm=place.translation_mm,
        rotation_xyzw=place.rotation_xyzw,
    )


def world_segment(spec: ShapeSpec, place: Placement) -> tuple[Vector3, Vector3]:
    posed = posed_body("body/probe", spec, place)
    a, b = spec.local_segment()
    return posed.transform_point_mm(a), posed.transform_point_mm(b)


def our_separation_mm(
    shape_a: ShapeSpec, place_a: Placement, shape_b: ShapeSpec, place_b: Placement
) -> float:
    """本仓侧的被验量：中轴线段距离减两半径。相交时为负。"""

    a0, a1 = world_segment(shape_a, place_a)
    b0, b1 = world_segment(shape_b, place_b)
    return segment_segment_distance_mm(a0, a1, b0, b1) - (shape_a.radius_mm + shape_b.radius_mm)


def our_event(
    shape_a: ShapeSpec, place_a: Placement, shape_b: ShapeSpec, place_b: Placement
) -> tuple[bool, float | None]:
    """`BroadPhaseCollisionQuery`侧的被验量：报不报事件、报多深。"""

    query = BroadPhaseCollisionQuery(
        (
            posed_body("body/a", shape_a, place_a),
            posed_body("body/b", shape_b, place_b),
        )
    )
    events = query.check_state()
    if not events:
        return False, None
    event = events[0]
    return event.confidence == "narrow_phase", event.penetration_mm


# ---------------------------------------------------------------- 输入生成


def _unit_quaternion_xyzw(rng: random.Random) -> tuple[float, float, float, float]:
    """S^3上均匀的单位四元数（Shoemake法），xyzw次序。"""

    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1, s2 = math.sqrt(1.0 - u1), math.sqrt(u1)
    x = s1 * math.sin(2.0 * math.pi * u2)
    y = s1 * math.cos(2.0 * math.pi * u2)
    z = s2 * math.sin(2.0 * math.pi * u3)
    w = s2 * math.cos(2.0 * math.pi * u3)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def _unit_direction(rng: random.Random) -> Vector3:
    while True:
        v = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if norm > 1.0e-6:
            return (v[0] / norm, v[1] / norm, v[2] / norm)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _shape_for(rng: random.Random, want_capsule: bool) -> ShapeSpec:
    radius = _log_uniform(rng, 0.5, 50.0)
    half_length = _log_uniform(rng, 1.0, 200.0) if want_capsule else 0.0
    return ShapeSpec(radius_mm=radius, half_length_mm=half_length)


def _sample_target(rng: random.Random, band: Band, shape_a: ShapeSpec, shape_b: ShapeSpec) -> float:
    low, high = BAND_TARGETS[band]
    if band == "penetrating":
        # 侵入不能超过两体能重叠的量，否则二分无解。
        deepest = 0.9 * min(shape_a.radius_mm, shape_b.radius_mm)
        return -_log_uniform(rng, min(-high, deepest * 0.999), deepest)
    return _log_uniform(rng, low, high)


def _place_to_target(
    shape_a: ShapeSpec,
    place_a: Placement,
    shape_b: ShapeSpec,
    rotation_b: tuple[float, float, float, float],
    direction: Vector3,
    target_mm: float,
) -> Placement | None:
    """沿`direction`平移B，二分到本仓分离量≈`target_mm`。

    **这只是摆位启发式，不是被判的数**：摆完之后FCL独立地对同一批坐标算它自己
    那一列。即便本仓实现有错，摆位也只是偏离目标band，对拍本身照样成立。

    可行性：两形状都关于自身原点对称，故s=0是分离量沿任意射线的全局最小；
    凸集的符号距离是平移的凸函数，因此分离量在s≥0上非降，二分有效。

    实现上先把两条中轴线段各自转好（旋转与s无关），循环里只做平移——
    与逐次构造`PosedBody`数学上同一，但快一个量级。
    """

    a0, a1 = world_segment(shape_a, place_a)
    base_b = Placement(translation_mm=(0.0, 0.0, 0.0), rotation_xyzw=rotation_b)
    rb0, rb1 = world_segment(shape_b, base_b)
    radius_sum = shape_a.radius_mm + shape_b.radius_mm
    origin = place_a.translation_mm

    def translation_at(scale: float) -> Vector3:
        return (
            origin[0] + direction[0] * scale,
            origin[1] + direction[1] * scale,
            origin[2] + direction[2] * scale,
        )

    def separation_at(scale: float) -> float:
        t = translation_at(scale)
        b0 = (rb0[0] + t[0], rb0[1] + t[1], rb0[2] + t[2])
        b1 = (rb1[0] + t[0], rb1[1] + t[1], rb1[2] + t[2])
        return segment_segment_distance_mm(a0, a1, b0, b1) - radius_sum

    if separation_at(0.0) > target_mm:
        return None  # 目标比最深重叠还深，换一组输入
    lo, hi = 0.0, 1.0
    for _ in range(80):
        if separation_at(hi) >= target_mm:
            break
        hi *= 2.0
    else:  # pragma: no cover - 半径上界50mm下不可能走到
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if separation_at(mid) < target_mm:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1.0e-15 * max(1.0, hi):
            break
    return Placement(translation_mm=translation_at(hi), rotation_xyzw=rotation_b)


def generate_samples(seed: int, per_cell: int) -> tuple[Sample, ...]:
    """固定种子生成`3类 × 3band × per_cell`组输入。纯stdlib，可离线重生成。"""

    rng = random.Random(seed)
    samples: list[Sample] = []
    index = 0
    for pair_class in PAIR_CLASSES:
        want_a = pair_class == "capsule_capsule"
        want_b = pair_class in ("sphere_capsule", "capsule_capsule")
        for band in BANDS:
            produced = 0
            attempts = 0
            while produced < per_cell:
                attempts += 1
                if attempts > per_cell * 50 + 100:  # pragma: no cover - 防御性
                    raise RuntimeError(f"sample generation stalled: {pair_class}/{band}")
                shape_a = _shape_for(rng, want_a)
                shape_b = _shape_for(rng, want_b)
                place_a = Placement(
                    translation_mm=(
                        rng.uniform(-500.0, 500.0),
                        rng.uniform(-500.0, 500.0),
                        rng.uniform(-500.0, 500.0),
                    ),
                    rotation_xyzw=_unit_quaternion_xyzw(rng),
                )
                rotation_b = _unit_quaternion_xyzw(rng)
                direction = _unit_direction(rng)
                target = _sample_target(rng, band, shape_a, shape_b)
                place_b = _place_to_target(
                    shape_a, place_a, shape_b, rotation_b, direction, target
                )
                if place_b is None:
                    continue
                samples.append(
                    Sample(
                        index=index,
                        pair_class=pair_class,
                        band=band,
                        shape_a=shape_a,
                        shape_b=shape_b,
                        place_a=place_a,
                        place_b=place_b,
                    )
                )
                index += 1
                produced += 1
    return tuple(samples)


# ---------------------------------------------------------------- 故意注错

#: 轴7规则6/plans-02"每道门要有它必须红的输入"的执行体。
#: 每一条都是**同行对拍能抓到、而仓内自洽测试抓不到**的那类错。
FAULTS: dict[str, str] = {
    "none": "不注错——正常对拍",
    "quaternion_order": "把本仓的xyzw当wxyz用：仓内自洽测试全绿，对拍必红",
    "radius_sum": "分离量只减一个半径：闭式公式的经典漏项",
    "capsule_half_length": "把胶囊半长当全长：与FCL的lz语义映射错位的经典形",
}


def _apply_fault(fault: str, sample: Sample) -> Sample:
    """注错**只改本仓这一侧**看到的输入，FCL那一侧始终拿原始输入。"""

    if fault == "quaternion_order":
        def shift(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            return (q[3], q[0], q[1], q[2])

        return Sample(
            index=sample.index,
            pair_class=sample.pair_class,
            band=sample.band,
            shape_a=sample.shape_a,
            shape_b=sample.shape_b,
            place_a=Placement(sample.place_a.translation_mm, shift(sample.place_a.rotation_xyzw)),
            place_b=Placement(sample.place_b.translation_mm, shift(sample.place_b.rotation_xyzw)),
        )
    if fault == "capsule_half_length":
        def double(spec: ShapeSpec) -> ShapeSpec:
            return ShapeSpec(spec.radius_mm, spec.half_length_mm * 2.0)

        return Sample(
            index=sample.index,
            pair_class=sample.pair_class,
            band=sample.band,
            shape_a=double(sample.shape_a),
            shape_b=double(sample.shape_b),
            place_a=sample.place_a,
            place_b=sample.place_b,
        )
    return sample


# ---------------------------------------------------------------- FCL适配


def _fcl_geometry(fcl_module, spec: ShapeSpec):  # type: ignore[no-untyped-def]
    if spec.is_sphere:
        return fcl_module.Sphere(spec.radius_mm)
    return fcl_module.Capsule(spec.radius_mm, 2.0 * spec.half_length_mm)


def fcl_object(fcl_module, numpy_module, spec: ShapeSpec, place: Placement):  # type: ignore[no-untyped-def]
    """本仓xyzw → FCL wxyz的唯一转换点。"""

    x, y, z, w = place.rotation_xyzw
    transform = fcl_module.Transform(
        numpy_module.array([w, x, y, z], dtype=float),
        numpy_module.array(place.translation_mm, dtype=float),
    )
    return fcl_module.CollisionObject(_fcl_geometry(fcl_module, spec), transform)


def peer_readings(fcl_module, numpy_module, sample: Sample) -> dict:
    """一次取齐FCL的四个读数——它们来自**三条不同的算子**，不可互相冒充。"""

    object_a = fcl_object(fcl_module, numpy_module, sample.shape_a, sample.place_a)
    object_b = fcl_object(fcl_module, numpy_module, sample.shape_b, sample.place_b)

    unsigned = fcl_module.distance(
        object_a,
        object_b,
        fcl_module.DistanceRequest(enable_nearest_points=False, enable_signed_distance=False),
        fcl_module.DistanceResult(),
    )
    signed = fcl_module.distance(
        object_a,
        object_b,
        fcl_module.DistanceRequest(enable_nearest_points=False, enable_signed_distance=True),
        fcl_module.DistanceResult(),
    )
    result = fcl_module.CollisionResult()
    contacts = fcl_module.collide(
        object_a,
        object_b,
        fcl_module.CollisionRequest(num_max_contacts=4, enable_contact=True),
        result,
    )
    depth = (
        max(contact.penetration_depth for contact in result.contacts)
        if result.contacts
        else None
    )
    return {
        "unsigned": unsigned,
        "signed": signed,
        "collides": bool(contacts),
        "contact_depth": depth,
    }


def compare(fcl_module, numpy_module, samples: tuple[Sample, ...], fault: str) -> list[dict]:
    """逐点对拍。每条记录带够复现的形状参数与两侧的**原始**返回值。"""

    if fault not in FAULTS:
        raise ValueError(f"unknown fault: {fault}")
    rows: list[dict] = []
    for sample in samples:
        faulted = _apply_fault(fault, sample)
        ours = our_separation_mm(
            faulted.shape_a, faulted.place_a, faulted.shape_b, faulted.place_b
        )
        if fault == "radius_sum":
            ours += faulted.shape_b.radius_mm
        event, penetration = our_event(
            faulted.shape_a, faulted.place_a, faulted.shape_b, faulted.place_b
        )
        peer = peer_readings(fcl_module, numpy_module, sample)

        # 路由**只看FCL自己的返回值**，不看我们的数——否则对拍是循环的。
        # 规则：unsigned不是哨兵就用它（分离/擦边全部、外加胶囊-胶囊相交，
        # 该对FCL有带符号的解析实现）；是哨兵才退到接触深度。
        # 哨兵歧义（真值恰为−1.0mm）在此规则下无害：那种情况下接触深度
        # 也≈−1.0，两条路给同一个答案。
        if peer["unsigned"] != FCL_COLLISION_SENTINEL:
            operator = "distance.unsigned"
            peer_value = peer["unsigned"]
        else:
            operator = "collide.penetration_depth"
            peer_value = -peer["contact_depth"] if peer["contact_depth"] is not None else None

        deviation = abs(ours - peer_value) if peer_value is not None else None
        scale = max(abs(ours), abs(peer_value)) if peer_value is not None else 0.0
        signed_deviation = abs(ours - peer["signed"])
        depth_deviation = (
            abs(ours + peer["contact_depth"]) if peer["contact_depth"] is not None else None
        )
        rows.append(
            {
                "index": sample.index,
                "pair_class": sample.pair_class,
                "band": sample.band,
                "radius_a_mm": sample.shape_a.radius_mm,
                "radius_b_mm": sample.shape_b.radius_mm,
                "half_length_a_mm": sample.shape_a.half_length_mm,
                "half_length_b_mm": sample.shape_b.half_length_mm,
                "ours_separation_mm": ours,
                "ours_event": event,
                "ours_penetration_mm": penetration,
                "peer_operator": operator,
                "peer_separation_mm": peer_value,
                "peer_unsigned_mm": peer["unsigned"],
                "peer_signed_mm": peer["signed"],
                "peer_contact_depth_mm": peer["contact_depth"],
                "peer_collides": peer["collides"],
                "abs_deviation_mm": deviation,
                "rel_deviation": (deviation / scale) if (deviation is not None and scale > 0.0)
                else 0.0,
                "signed_path_abs_deviation_mm": signed_deviation,
                "contact_depth_abs_deviation_mm": depth_deviation,
            }
        )
    return rows


def _new_cell(operator: str) -> dict:
    return {
        "count": 0,
        "peer_operator": operator,
        "max_abs_deviation_mm": 0.0,
        "max_rel_deviation": 0.0,
        "predicate_disagreements": 0,
        "missing_peer_value": 0,
    }


def summarise(rows: list[dict]) -> dict:
    """按(类, band)聚合。判据读的是本函数的输出，不是逐行数据。"""

    cells: dict[str, dict] = {}
    for row in rows:
        key = f"{row['pair_class']}/{row['band']}"
        cell = cells.setdefault(key, _new_cell(row["peer_operator"]))
        cell["count"] += 1
        if row["peer_operator"] != cell["peer_operator"]:
            cell["peer_operator"] = "mixed"
        if row["abs_deviation_mm"] is None:
            cell["missing_peer_value"] += 1
        else:
            cell["max_abs_deviation_mm"] = max(
                cell["max_abs_deviation_mm"], row["abs_deviation_mm"]
            )
            cell["max_rel_deviation"] = max(cell["max_rel_deviation"], row["rel_deviation"])
        if row["ours_event"] != row["peer_collides"]:
            cell["predicate_disagreements"] += 1

    measured = [r for r in rows if r["abs_deviation_mm"] is not None]
    signed_devs = [r["signed_path_abs_deviation_mm"] for r in rows]
    overall = {
        "count": len(rows),
        "max_abs_deviation_mm": max((r["abs_deviation_mm"] for r in measured), default=0.0),
        "max_rel_deviation": max((r["rel_deviation"] for r in measured), default=0.0),
        "predicate_disagreements": sum(1 for r in rows if r["ours_event"] != r["peer_collides"]),
        "missing_peer_value": len(rows) - len(measured),
    }
    #: 被记录但**不作判据**的两列：FCL那两条已查实会大错的路。
    #: 它们进产物是为了让"同行也会错"这件事有据可查，不是为了当判据
    #: ——把同行的缺陷钉成门，下一个版本就会莫名其妙地红（spec/08规则1）。
    def _depth_devs(pair_class: str) -> list[float]:
        return [
            r["contact_depth_abs_deviation_mm"]
            for r in rows
            if r["pair_class"] == pair_class and r["contact_depth_abs_deviation_mm"] is not None
        ]

    observed = {
        "peer_signed_path_max_abs_deviation_mm": max(signed_devs, default=0.0),
        "peer_signed_path_over_1e_5_mm": sum(1 for d in signed_devs if d > 1.0e-5),
        "peer_signed_path_over_1e_6_mm": sum(1 for d in signed_devs if d > 1.0e-6),
        "peer_contact_depth_max_abs_deviation_mm": {
            pair_class: max(_depth_devs(pair_class), default=0.0) for pair_class in PAIR_CLASSES
        },
        "peer_contact_depth_over_1e_5_mm": {
            pair_class: sum(1 for d in _depth_devs(pair_class) if d > 1.0e-5)
            for pair_class in PAIR_CLASSES
        },
    }
    return {"cells": dict(sorted(cells.items())), "overall": overall, "observed": observed}
