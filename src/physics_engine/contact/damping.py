"""线性法向阻尼：恢复系数与阻尼比的双向换算，以及dashpot耗散项。

``restitution_from_damping_ratio``与``step_response_overshoot``（`drives`）
**不是同一个式子**——两者都含``exp(−ζΦ/√(1−ζ²))``，但``Φ``一个取``2·acos(ζ)``
（0052第一节的截断约定）、一个取``π``；ζ=0.5处差1.83倍，**只在ζ=0重合**。

拆分自原`contact.py`（2026-08-17）——**函数体逐字节未动**。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.contact.layout import NORMAL_UNIT_TOLERANCE
from physics_engine.contact.penalty import PenaltyNormalContact
from physics_engine.energies import DISSIPATION, EnergyContext, Vector
from physics_engine.state import State


def _force_zero_contact_time_factor(damping_ratio: float) -> float:
    """返回``ω0·t_c``；定义域覆盖欠阻尼、临界与过阻尼。"""

    if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
        raise ContactError(
            f"damping ratio must be finite and nonnegative: {damping_ratio!r}"
        )
    if damping_ratio == 1.0:
        return 2.0
    if damping_ratio < 1.0:
        root = math.sqrt((1.0 - damping_ratio) * (1.0 + damping_ratio))
        return 2.0 * math.acos(damping_ratio) / root
    root = math.sqrt(damping_ratio - 1.0) * math.sqrt(damping_ratio + 1.0)
    return 2.0 * math.acosh(damping_ratio) / root


def restitution_from_damping_ratio(damping_ratio: float) -> float:
    """合力归零分离约定下的线性弹簧-dashpot恢复系数。"""

    factor = _force_zero_contact_time_factor(damping_ratio)
    return math.exp(-damping_ratio * factor)


def damping_ratio_from_restitution(restitution: float) -> float:
    """反解``0 < e ≤ 1``对应的唯一有限阻尼比；过阻尼同样允许。"""

    if not math.isfinite(restitution) or not (0.0 < restitution <= 1.0):
        raise ContactError(
            f"restitution must be finite and in (0, 1], got {restitution!r}; "
            "e=0 only occurs in the infinite-damping limit"
        )
    if restitution == 1.0:
        return 0.0
    lower, upper = 0.0, 1.0
    while restitution_from_damping_ratio(upper) > restitution:
        upper *= 2.0
        if not math.isfinite(upper):
            raise ContactError(
                f"restitution {restitution!r} requires a damping ratio beyond float range"
            )
    #: e(ζ)在[0,∞)严格单调；固定120次让结果跨平台确定，不用容差提前退出。
    for _ in range(120):
        middle = 0.5 * (lower + upper)
        if restitution_from_damping_ratio(middle) > restitution:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


@dataclass(frozen=True)
class LinearDashpotParameters:
    """目标恢复系数派生出的接触动力学参数（mm-N-kg-s单位制）。"""

    restitution: float
    damping_ratio: float
    stiffness_n_per_mm: float
    effective_mass_kg: float
    damping_n_s_per_mm: float
    omega0_rad_per_s: float
    contact_duration_s: float
    stability_rate_per_s: float


def linear_dashpot_parameters(
    *, stiffness_n_per_mm: float, effective_mass_kg: float, restitution: float
) -> LinearDashpotParameters:
    """从``(k, m_eff, e)``派生dashpot系数、接触时长与显式稳定率。"""

    if not math.isfinite(stiffness_n_per_mm) or stiffness_n_per_mm <= 0.0:
        raise ContactError(
            f"penalty stiffness must be positive and finite: {stiffness_n_per_mm!r}"
        )
    if not math.isfinite(effective_mass_kg) or effective_mass_kg <= 0.0:
        raise ContactError(
            f"effective mass must be positive and finite: {effective_mass_kg!r}"
        )
    damping_ratio = damping_ratio_from_restitution(restitution)
    omega0 = math.sqrt(1000.0 * stiffness_n_per_mm / effective_mass_kg)
    damping = 2.0 * damping_ratio * effective_mass_kg * omega0 / 1000.0
    duration = _force_zero_contact_time_factor(damping_ratio) / omega0
    if damping_ratio <= 1.0:
        stability_rate = omega0
    else:
        root = math.sqrt(damping_ratio - 1.0) * math.sqrt(damping_ratio + 1.0)
        stability_rate = (damping_ratio + root) * omega0
    return LinearDashpotParameters(
        restitution=restitution,
        damping_ratio=damping_ratio,
        stiffness_n_per_mm=stiffness_n_per_mm,
        effective_mass_kg=effective_mass_kg,
        damping_n_s_per_mm=damping,
        omega0_rad_per_s=omega0,
        contact_duration_s=duration,
        stability_rate_per_s=stability_rate,
    )


@dataclass(frozen=True)
class LinearNormalDashpot:
    """半空间与球-球的线性法向dashpot，按**合力归零**截断。

    本项只给耗散力；弹簧势能仍由``PenaltyNormalContact``或
    ``PenaltySphereContact``给。先算弹簧压缩力``N_s``，再令
    ``N = max(0, N_s − c·g_dot)``，本项贡献``N−N_s``。因此出射阶段允许
    dashpot抵消弹簧，但总接触力永不成为拉力。
    """

    name: str = "normal_dashpot"
    kind: ClassVar[Literal["dissipation"]] = DISSIPATION
    #: (节点, 面上一点, 外法向, 弹簧刚度N/mm, 阻尼N·s/mm, 半径mm)
    planes: tuple[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            float,
        ],
        ...,
    ] = ()
    #: (节点i, 节点j, 半径和mm, 弹簧刚度N/mm, 阻尼N·s/mm)
    sphere_pairs: tuple[tuple[int, int, float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.planes and not self.sphere_pairs:
            raise ContactError("normal_dashpot needs at least one plane or sphere pair")
        for node, point, normal, stiffness, damping, radius in self.planes:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"dashpot node index must be a nonnegative int: {node!r}")
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"dashpot plane point must be a finite 3-vector: {point!r}")
            if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
                raise ContactError(f"dashpot normal must be a finite 3-vector: {normal!r}")
            norm = math.sqrt(sum(value * value for value in normal))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(f"dashpot normal must be a unit vector (|n| = {norm!r})")
            self._validate_coefficients(stiffness, damping, radius)
        for i, j, radii_sum, stiffness, damping in self.sphere_pairs:
            for node in (i, j):
                if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                    raise ContactError(
                        f"dashpot sphere node index must be a nonnegative int: {node!r}"
                    )
            if i == j:
                raise ContactError(f"a sphere cannot damp contact with itself: node {i}")
            self._validate_coefficients(stiffness, damping, radii_sum)

    @classmethod
    def _from_validated_parts(
        cls,
        *,
        planes: tuple[
            tuple[
                int,
                tuple[float, float, float],
                tuple[float, float, float],
                float,
                float,
                float,
            ],
            ...,
        ],
        sphere_pairs: tuple[tuple[int, int, float, float, float], ...],
    ) -> LinearNormalDashpot:
        """由同包内已验证装配层构造；调用方承担全部``__post_init__``不变量。"""

        term = object.__new__(cls)
        object.__setattr__(term, "name", "normal_dashpot")
        object.__setattr__(term, "planes", planes)
        object.__setattr__(term, "sphere_pairs", sphere_pairs)
        return term

    @staticmethod
    def _validate_coefficients(stiffness: float, damping: float, radius: float) -> None:
        if not math.isfinite(stiffness) or stiffness <= 0.0:
            raise ContactError(f"dashpot stiffness must be positive: {stiffness!r}")
        if not math.isfinite(damping) or damping <= 0.0:
            raise ContactError(f"dashpot damping must be positive: {damping!r}")
        if not math.isfinite(radius) or radius < 0.0:
            raise ContactError(f"dashpot radius must be finite and nonnegative: {radius!r}")

    def node_index_bound(self) -> int:
        indices = [plane[0] for plane in self.planes]
        for i, j, _, _, _ in self.sphere_pairs:
            indices.extend((i, j))
        return max(indices) + 1

    @staticmethod
    def _damping_magnitude(
        *, gap_mm: float, gap_rate_mm_per_s: float, stiffness: float, damping: float
    ) -> float:
        if gap_mm >= 0.0:
            return 0.0
        spring = -stiffness * gap_mm
        total = max(0.0, spring - damping * gap_rate_mm_per_s)
        return total - spring

    def force_and_power(
        self, state: State, velocity: Sequence[float], context: EnergyContext
    ) -> tuple[Vector, float]:
        if len(velocity) != len(state.vector):
            raise ContactError("dashpot velocity and state vector must have the same length")
        force = [0.0] * len(state.vector)
        power = 0.0
        for node, point, normal, stiffness, damping, radius in self.planes:
            gap = PenaltyNormalContact._gap_mm(state.vector, node, point, normal, radius)
            base = 3 * node
            gap_rate = sum(velocity[base + axis] * normal[axis] for axis in range(3))
            magnitude = self._damping_magnitude(
                gap_mm=gap,
                gap_rate_mm_per_s=gap_rate,
                stiffness=stiffness,
                damping=damping,
            )
            for axis in range(3):
                force[base + axis] += magnitude * normal[axis]
            power += max(0.0, -magnitude * gap_rate)

        for i, j, radii_sum, stiffness, damping in self.sphere_pairs:
            delta = tuple(
                state.vector[3 * j + axis] - state.vector[3 * i + axis]
                for axis in range(3)
            )
            length = math.sqrt(sum(value * value for value in delta))
            if length == 0.0:
                raise ContactError(
                    f"spheres {i} and {j} share a centre — dashpot direction is undefined"
                )
            direction = tuple(value / length for value in delta)
            gap = length - radii_sum
            gap_rate = sum(
                (velocity[3 * j + axis] - velocity[3 * i + axis]) * direction[axis]
                for axis in range(3)
            )
            magnitude = self._damping_magnitude(
                gap_mm=gap,
                gap_rate_mm_per_s=gap_rate,
                stiffness=stiffness,
                damping=damping,
            )
            for axis in range(3):
                component = magnitude * direction[axis]
                force[3 * i + axis] -= component
                force[3 * j + axis] += component
            power += max(0.0, -magnitude * gap_rate)
        return tuple(force), power
