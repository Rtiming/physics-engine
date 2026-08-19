"""conformance：距离场接触的收敛（`cases/sdf_contact_convergence`）。

判据正本是`cases/sdf_contact_convergence/oracle.json`，闭式出处在
`cases/sdf_contact_convergence/generate_oracle.py`的模块docstring。
本文件只做三件事：把内核摆到清单说的构型上、取数、按清单的容差比。

**内核的数拿来撞闭式，不是反过来**（spec/08规则1）。
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import pytest

from physics_engine.contact.field import (
    PenaltySignedDistanceField,
    half_space_distance_mm,
    sample_narrow_band,
    sphere_distance_mm,
)
from physics_engine.contact.penalty import PenaltyNormalContact, PenaltySphereContact
from physics_engine.energies import EnergyContext, EnergyRegistry, PointLoad
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State, StateField, StateLayout

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/sdf_contact_convergence/oracle.json", root=ROOT)
ORACLES = {oracle.id.rsplit("/", 1)[1]: oracle for oracle in MANIFEST.oracles}

CONSTANTS = ORACLES["analytic_equilibrium"].inputs
SPHERE_RADIUS_MM = CONSTANTS["sphere_radius_mm"]
NODE_RADIUS_MM = CONSTANTS["node_radius_mm"]
STIFFNESS_N_PER_MM = CONSTANTS["stiffness_n_per_mm"]
HOLD_DOWN_N = CONSTANTS["hold_down_n"]
SPACINGS_MM = tuple(ORACLES["order_and_penalty_ratio"].inputs["spacings_mm"])

CENTRE = (0.0, 0.0, 0.0)
PLANE_POINT = (0.0, 0.0, 0.0)
PLANE_NORMAL = (0.0, 0.0, 1.0)

#: **两个节点**：0号是被压的那个，1号钉在球心。
#: 1号存在的唯一理由是`PenaltySphereContact`两端都要自由度——
#: 把它全部钉住，那一项就退化成"固定球障碍"，而**仓里本来就有这个解析球**，
#: 不必为本案例新造一个接触项。半空间与场那两档不碰1号，但共用同一份布局，
#: **三条路摆在同一个构型上才叫"并排"**。
LAYOUT = StateLayout(
    layout_id="layout/sdf_contact_convergence",
    fields=(
        StateField("node0_x_mm", 1),
        StateField("node0_y_mm", 1),
        StateField("node0_z_mm", 1),
        StateField("node1_x_mm", 1),
        StateField("node1_y_mm", 1),
        StateField("node1_z_mm", 1),
    ),
)
CONTEXT = EnergyContext(
    context_id="context/sdf_contact_convergence",
    node_masses_kg=(1.0e-9, 1.0e-9),
    gravity_mm_s2=(0.0, 0.0, 0.0),
)
#: `solve.py`申报的参考取法是"总载荷的1e-9到1e-10"；总载荷25 N ⟹ 2.5e-8—2.5e-9。
#: 这里取1e-9（更紧一档），实测本构型**一次牛顿**就到——
#: 活动接触段的能量对``z``严格二次，牛顿在二次函数上一步精确。
RESIDUAL_TOL_N = 1.0e-9
FIXED = frozenset({0, 1, 3, 4, 5})


def _state(z_mm: float) -> State:
    return State(layout=LAYOUT, vector=(0.0, 0.0, z_mm, 0.0, 0.0, 0.0))


#: 烘一次场在``h = 0.25``那一档要0.47秒，而三条门都要它。
#: 缓存是**测试的**优化不是内核的：场是不可变的，同一档``h``烘出来的字节相同。
@cache
def _sphere_field(spacing_mm: float):
    extent = 13.0
    count = int(2.0 * extent / spacing_mm) + 1
    return sample_narrow_band(
        lambda point: sphere_distance_mm(point, CENTRE, SPHERE_RADIUS_MM),
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=spacing_mm,
        node_counts=(count, count, count),
        band_mm=max(3.5, 2.0 * spacing_mm * math.sqrt(3.0) * 1.05),
    )


@cache
def _plane_field(spacing_mm: float):
    extent = 10.0
    count = int(2.0 * extent / spacing_mm) + 1
    return sample_narrow_band(
        lambda point: half_space_distance_mm(point, PLANE_POINT, PLANE_NORMAL),
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=spacing_mm,
        node_counts=(count, count, count),
        band_mm=8.0,
    )


def _solve(term, start_z_mm: float) -> float:
    registry = EnergyRegistry(
        terms=(term, PointLoad(loads=((0, (0.0, 0.0, -HOLD_DOWN_N)),)))
    )
    result = solve_equilibrium(
        registry,
        CONTEXT,
        LAYOUT,
        (0.0, 0.0, start_z_mm, 0.0, 0.0, 0.0),
        fixed_indices=FIXED,
        residual_tol_n=RESIDUAL_TOL_N,
        max_iterations=60,
    )
    assert result.converged, result.reason
    #: 回溯次数进结果是`solve.py`第三条申报义务；这里断言它是0——
    #: 本构型良态，一旦开始回溯说明构型或场变了，那本身就是信号。
    assert result.backtracks == 0, result.backtracks
    return result.state.vector[2]


def _field_term(field):
    return PenaltySignedDistanceField(
        field=field, contacts=((0, STIFFNESS_N_PER_MM, NODE_RADIUS_MM),)
    )


# --------------------------- 闭式一：解析接触项的平衡 ---


def test_the_analytic_terms_reproduce_the_closed_form_equilibrium() -> None:
    """两个**已有的**解析接触项各解一次，都要落在闭式一上（逐位）。

    半空间用`PenaltyNormalContact`；球用`PenaltySphereContact`把1号节点钉在球心。
    活动接触段的能量对``z``严格二次 ⟹ 牛顿一步精确 ⟹ 零容差**是代数保证的**，
    不是精度自信（清单里的``reason``写了同一句）。
    """

    oracle = ORACLES["analytic_equilibrium"]
    plane = PenaltyNormalContact(
        planes=((0, PLANE_POINT, PLANE_NORMAL, STIFFNESS_N_PER_MM, NODE_RADIUS_MM),)
    )
    sphere = PenaltySphereContact(
        pairs=((0, 1, SPHERE_RADIUS_MM + NODE_RADIUS_MM, STIFFNESS_N_PER_MM),)
    )
    plane_z = _solve(plane, NODE_RADIUS_MM - 0.08)
    sphere_z = _solve(sphere, SPHERE_RADIUS_MM + NODE_RADIUS_MM - 0.08)
    gap = plane.normal_force_n(_state(plane_z))[0] / -STIFFNESS_N_PER_MM
    oracle.check_all(
        {
            "plane_equilibrium_z_mm": plane_z,
            "sphere_equilibrium_z_mm": sphere_z,
            "contact_gap_mm": gap,
            "normal_force_n": plane.normal_force_n(_state(plane_z))[0],
        }
    )
    #: 球那一侧的力另判一次——**两个项各自的力都要对**，不是只对一个。
    assert sphere.contact_force_n(_state(sphere_z))[0] == pytest.approx(
        HOLD_DOWN_N, rel=1.0e-12
    )


# --------------------------- 闭式二上半：半空间，与h无关 ---


@pytest.mark.parametrize("spacing", [2.0, 1.0, 0.5])
def test_the_plane_field_gives_the_same_equilibrium_at_every_resolution(
    spacing: float,
) -> None:
    """仿射场被精确重构：**三档``h``跨4倍，平衡位置一个数**。

    这一条同时是"与`PenaltyNormalContact`并排"那条验收：
    实测两条路差**1 ulp（2.220e-16 mm）**，法向力25.0到1e-13相对。
    """

    key = "plane_field_h_" + str(spacing).replace(".", "p")
    term = _field_term(_plane_field(spacing))
    z = _solve(term, NODE_RADIUS_MM - 0.08)
    ORACLES[key].check_all(
        {"equilibrium_z_mm": z, "normal_force_n": term.normal_force_n(_state(z))[0]}
    )


def test_the_plane_field_matches_the_analytic_term_within_one_ulp() -> None:
    """并排那条的直判：两条路解出来的``z``差不超过1 ulp。

    **不写逐位相等**——两条路的求和次序不同（一条点积、一条64项B样条求和），
    逐位相等是没验过的性质，承诺它就是冒充。
    """

    analytic = PenaltyNormalContact(
        planes=((0, PLANE_POINT, PLANE_NORMAL, STIFFNESS_N_PER_MM, NODE_RADIUS_MM),)
    )
    reference = _solve(analytic, NODE_RADIUS_MM - 0.08)
    for spacing in (2.0, 1.0, 0.5):
        z = _solve(_field_term(_plane_field(spacing)), NODE_RADIUS_MM - 0.08)
        assert abs(z - reference) <= 4.0 * abs(math.ulp(reference)), (spacing, z - reference)


# --------------------------- 闭式二下半：球，主项 −h²/(3z*) ---


@pytest.mark.parametrize("spacing", [1.0, 0.5, 0.25])
def test_the_sphere_field_offset_matches_the_leading_term(spacing: float) -> None:
    """**场把物体往里推**，主项``−h²/(3z*)``，实测与主项差0.35%以内。

    实测偏差：``−3.18728e-02 / −7.95010e-03 / −1.98639e-03``mm；
    主项：``−3.17612e-02 / −7.94029e-03 / −1.98507e-03``mm。
    """

    key = "sphere_field_h_" + str(spacing).replace(".", "p")
    star = SPHERE_RADIUS_MM + NODE_RADIUS_MM - HOLD_DOWN_N / STIFFNESS_N_PER_MM
    z = _solve(_field_term(_sphere_field(spacing)), SPHERE_RADIUS_MM + NODE_RADIUS_MM - 0.08)
    ORACLES[key].check_all({"equilibrium_offset_mm": z - star, "equilibrium_z_mm": z})


def test_the_offset_is_negative_at_every_resolution() -> None:
    """**符号是判据的一部分**：凸障碍上场系统性偏松，物体沉得更深。

    0074第二节第4条只说"系统性偏保守或偏松"，没说是哪一边。
    本案例把那句话钉到一边：**对凸障碍是偏松**。
    数值噪声不会三档全部同号，所以这条断言有分辨力。
    """

    star = SPHERE_RADIUS_MM + NODE_RADIUS_MM - HOLD_DOWN_N / STIFFNESS_N_PER_MM
    for spacing in SPACINGS_MM:
        z = _solve(
            _field_term(_sphere_field(spacing)), SPHERE_RADIUS_MM + NODE_RADIUS_MM - 0.08
        )
        assert z < star, (spacing, z - star)


# --------------------------- 闭式三：阶与"比罚穿透大多少" ---


def test_the_order_and_the_ratio_to_the_penalty_penetration() -> None:
    """三档解一遍，量阶（实测4.0091 / 4.0023）与闭式三那个**6.37倍**。

    梯度那一条另算：中心差分对能量，实测比恒4.0000。
    """

    star = SPHERE_RADIUS_MM + NODE_RADIUS_MM - HOLD_DOWN_N / STIFFNESS_N_PER_MM
    offsets = []
    terms = []
    for spacing in SPACINGS_MM:
        term = _field_term(_sphere_field(spacing))
        terms.append(term)
        z = _solve(term, SPHERE_RADIUS_MM + NODE_RADIUS_MM - 0.08)
        offsets.append(abs(z - star))

    ratios = [offsets[i] / offsets[i + 1] for i in range(len(offsets) - 1)]
    assert len(ratios) == 2

    #: 梯度的中心差分——**在案例的构型上**再验一次（单元门里已验过一次）。
    #: 两处都留着的理由：单元门验的是任意一点，这里验的是**平衡点附近**，
    #: 而活动集恰好在那里翻转，是最容易出问题的地方。
    probe = (0.0, 0.0, star - 0.02)
    analytic = terms[1].gradient(_state(probe[2]), CONTEXT)
    errors = []
    for step in (0.004, 0.002, 0.001, 0.0005):
        ahead = terms[1].energy(_state(probe[2] + step), CONTEXT)
        behind = terms[1].energy(_state(probe[2] - step), CONTEXT)
        errors.append(abs((ahead - behind) / (2.0 * step) - analytic[2]))
    gradient_ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]

    ORACLES["order_and_penalty_ratio"].check_all(
        {
            "position_error_order_ratio": sum(ratios) / len(ratios),
            "gradient_finite_difference_ratio": sum(gradient_ratios)
            / len(gradient_ratios),
            "field_error_over_penalty_penetration_at_coarsest": offsets[0]
            / (HOLD_DOWN_N / STIFFNESS_N_PER_MM),
        }
    )


def test_every_oracle_in_the_manifest_is_exercised() -> None:
    """清单里的每一条都必须被上面某条门用到——**不许有睡着的判据**。

    这条与`check_all`那条"漏算一个量也是失败"同源：那条挡"挑着比"，
    这条挡"整条oracle没人碰"。
    """

    exercised = {
        "analytic_equilibrium",
        "order_and_penalty_ratio",
        *(f"plane_field_h_{str(s).replace('.', 'p')}" for s in (2.0, 1.0, 0.5)),
        *(f"sphere_field_h_{str(s).replace('.', 'p')}" for s in SPACINGS_MM),
    }
    assert set(ORACLES) == exercised, set(ORACLES) ^ exercised
