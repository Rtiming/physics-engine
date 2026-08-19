"""`contact_dynamics`那两条上限声明的门——决策0087的丙1与丙2。

本文件判四类，一类都不能少：

1. **迁移率谱对得上手推的闭式**（球的``1/m + R²/I``、箱底四角的``N/m + N·h²/I_yy``
   与偏航的``Σr_⊥²/I_zz``）——上限的分母是被算出来的，不是被声明的；
2. **两条上限与`rigidbody`／`contact_pipeline`共用同一个稳定区半径**，
   从声明字符串里抠出来对拍（0083注错第M4轮抓到的洞：那个数没有任何东西钉着，
   改松了不会红）；
3. **回归判据：声明要能预言0082第五节那次真实发散**——
   `k_t = 0.03`、`dt = 4e-4`上声明说不许，而实跑当场发散；`dt = 2e-4`上声明说可以，
   实跑稳定。**两侧都判**，只判一侧的门挡不住"永远说不许"的实现；
4. **必红那一类：声明说安全而实际发散**。判据是一条**双向**的对拍——
   把上限故意放松（当年那个"只算偏航模态"的式子就是这样漏的），
   它就会把一个真的会发散的构型放行，而本文件的第3类当场红。
"""

from __future__ import annotations

import math
import re

import pytest

from physics_engine.contact_dynamics import (
    CONTACT_DYNAMICS_STEP_BOUND,
    EXPLICIT_EULER_STABILITY_RADIUS,
    RK4_STABILITY_RADIUS,
    TANGENTIAL_COEFFICIENT_BOUND,
    ContactDynamicsError,
    box_corner_points_mm,
    contact_dynamics_step_bound,
    contact_mode_spectrum,
    creep_resolution_lower_bound,
    governing_assembly_step_bound,
    sphere_plane_step_bound,
    support_points_plane_callbacks,
    tangential_coefficient_window,
    tightest_step_bound,
)
from physics_engine.rigidbody import (
    EXPLICIT_EULER_BODY,
    RK4_BODY,
    RigidBodyInertia,
    integrate_free_flight,
    make_state,
)
from physics_engine.shapes import RoundedBox

# ---------------------------------------------------------------------------
# 两条案例的**声明输入**，逐字取自它们的oracle清单
# ---------------------------------------------------------------------------

BOX_HALF = (5.0, 5.0, 10.0)
BOX_MASS = 0.05
BOX_KN = 50.0
BOX_CN = 0.01
BOX_KT = 0.03
BOX_MU = 1.2
GRAVITY = 9810.0
BOX_THETA = math.radians(20.0)

BALL_RADIUS = 10.0
BALL_MASS = 1.0
BALL_KN = 5.0e5
BALL_KT = 50.0
BALL_INERTIA = 0.4 * BALL_MASS * BALL_RADIUS * BALL_RADIUS


def _box_inertia() -> RigidBodyInertia:
    return RigidBodyInertia.from_shape(
        RoundedBox(half_extents_mm=BOX_HALF, fillet_radius_mm=0.0), mass_kg=BOX_MASS
    )


def _ball_inertia() -> tuple[tuple[float, ...], ...]:
    return (
        (BALL_INERTIA, 0.0, 0.0),
        (0.0, BALL_INERTIA, 0.0),
        (0.0, 0.0, BALL_INERTIA),
    )


# ---------------------------------------------------------------------------
# 一、迁移率谱对手推闭式
# ---------------------------------------------------------------------------


def test_the_sphere_mobility_is_the_textbook_contact_point_mobility() -> None:
    """单点球的切向迁移率必须是``1/m + R²/I``——教科书上的接触点迁移率。

    **这条闭式是独立推出来的**（接触点速度``v + ω×r``对切向力的响应），
    不是把实现再抄一遍：实心球``I = (2/5)mR²``于是它等于``3.5/m``。
    """

    spectrum = contact_mode_spectrum(
        support_points_body_mm=((0.0, 0.0, -BALL_RADIUS),),
        mass_kg=BALL_MASS,
        inertia_body_kg_mm2=_ball_inertia(),
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    expected = 1.0 / BALL_MASS + BALL_RADIUS**2 / BALL_INERTIA
    assert expected == pytest.approx(3.5, rel=0, abs=0)
    assert spectrum.tangential_mobility_per_kg == pytest.approx(expected, rel=1e-14)
    #: 法向那一支只有平动：球心沿法向被压，杆臂平行于力，力矩恒为零。
    assert spectrum.normal_mobility_per_kg == pytest.approx(1.0 / BALL_MASS, rel=1e-14)


def test_the_box_bottom_face_mobility_is_translation_plus_pitch_not_yaw_alone() -> None:
    """箱底四角的最紧切向模态是**平动＋俯仰耦合**``N/m + N·h²/I_yy``，
    **不是**0082登记的那条偏航式子``Σr_⊥²/I_zz``。

    两条都在谱里，而**偏航那条不是最大的**：实测272 vs 240（1/kg），
    偏航低13.3%。0082第五节与案例页第四节第2条记的`λh = 2.88`用的正是偏航那一条，
    **它漏掉了更紧的那个模态**——这条门就是那个漏洞的回归判据。
    """

    inertia = _box_inertia()
    half_w, half_d, half_h = BOX_HALF
    spectrum = contact_mode_spectrum(
        support_points_body_mm=box_corner_points_mm(BOX_HALF)[:4],
        mass_kg=BOX_MASS,
        inertia_body_kg_mm2=inertia.inertia_body_kg_mm2,
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    inertia_yy = inertia.inertia_body_kg_mm2[1][1]
    inertia_zz = inertia.inertia_body_kg_mm2[2][2]
    pitch = 4.0 / BOX_MASS + 4.0 * half_h**2 / inertia_yy
    yaw = 4.0 * (half_w**2 + half_d**2) / inertia_zz
    assert spectrum.tangential_mobility_per_kg == pytest.approx(pitch, rel=1e-12)
    assert spectrum.tangential_mobilities_per_kg[2] == pytest.approx(yaw, rel=1e-12)
    assert yaw < pitch, "偏航若反而更紧，这条门的前提就变了，0082那条式子要重估"
    assert yaw / pitch == pytest.approx(240.0 / 272.0, rel=1e-12)


def test_the_box_normal_mobility_carries_the_rocking_mode_the_pipeline_cannot_see() -> None:
    """法向族里除了整体沉浮``N/m``，还有一个**摇摆**模态``N·w²/I_yy``。

    `contact_pipeline`那条上限只有前者（它没有杆臂）。摇摆这一条是
    "力矩装配看到的模态"最直白的一个例子：法向罚力在两侧不等时产生回复力矩。
    """

    inertia = _box_inertia()
    half_w, _, _ = BOX_HALF
    spectrum = contact_mode_spectrum(
        support_points_body_mm=box_corner_points_mm(BOX_HALF)[:4],
        mass_kg=BOX_MASS,
        inertia_body_kg_mm2=inertia.inertia_body_kg_mm2,
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    heave = 4.0 / BOX_MASS
    rocking = 4.0 * half_w**2 / inertia.inertia_body_kg_mm2[1][1]
    assert spectrum.normal_mobilities_per_kg[0] == pytest.approx(heave, rel=1e-12)
    assert spectrum.normal_mobilities_per_kg[1] == pytest.approx(rocking, rel=1e-12)


def test_the_attitude_free_supremum_bounds_every_oriented_value() -> None:
    """姿态无关那一支必须**大于等于**任何姿态上的精确值。

    它的依据是``P ⪯ I₃``，于是`Σ JᵀPJ ⪯ Σ JᵀJ`。扫一圈法向实测一遍——
    一个把``P = I₃``写成``P = n̂n̂ᵀ``的实现在单条法向上看着也对。
    """

    inertia = _box_inertia()
    points = box_corner_points_mm(BOX_HALF)[:4]
    free = contact_mode_spectrum(
        support_points_body_mm=points,
        mass_kg=BOX_MASS,
        inertia_body_kg_mm2=inertia.inertia_body_kg_mm2,
    )
    assert free.plane_normal_body is None
    supremum = free.tangential_mobility_per_kg
    for index in range(12):
        angle = index * math.pi / 6.0
        for normal in (
            (math.cos(angle), math.sin(angle), 0.0),
            (0.0, math.cos(angle), math.sin(angle)),
            (math.sin(angle), 0.0, math.cos(angle)),
        ):
            oriented = contact_mode_spectrum(
                support_points_body_mm=points,
                mass_kg=BOX_MASS,
                inertia_body_kg_mm2=inertia.inertia_body_kg_mm2,
                plane_normal_body=normal,
            )
            assert oriented.tangential_mobility_per_kg <= supremum * (1.0 + 1e-12)
            assert oriented.normal_mobility_per_kg <= supremum * (1.0 + 1e-12)
    #: 松多少也要报出来——13%是本仓那只箱子的实测值，不是一个普适常数。
    assert supremum / 272.0 == pytest.approx(1.1305, rel=1e-3)


# ---------------------------------------------------------------------------
# 二、常量与声明
# ---------------------------------------------------------------------------


def test_the_stability_radius_is_the_same_number_rigidbody_already_declared() -> None:
    """**必红**：把2.785改成别的数，本门必须红。

    0083注错第M4轮抓到过这个洞：那个常量没有任何东西钉着，改成2.0之后
    全套门都绿——**而那种错只会让上限变松，不会变红**，是最坏的一类。
    这条从`rigidbody`的声明**字符串里把数字抠出来**再对拍，
    于是它同时钉住"三条上限的分子是同一个数"这件事本身。
    """

    def declared_radius(text: str) -> float:
        match = re.search(r"h\s*<\s*([0-9.]+)\s*/\s*\|ω\|_max", text)
        assert match is not None, f"这条声明里读不出稳定区半径：{text!r}"
        return float(match.group(1))

    assert declared_radius(RK4_BODY.declaration.step_bound) == RK4_STABILITY_RADIUS
    assert (
        declared_radius(EXPLICIT_EULER_BODY.declaration.step_bound)
        == EXPLICIT_EULER_STABILITY_RADIUS
    )


def test_the_two_radii_agree_with_the_pipeline_copy() -> None:
    """本模块与`contact_pipeline`各写了一份稳定区半径。**两份不许漂。**

    分成两个模块各写一份不是重复：两条上限的**分母**不同，分子必须相同。
    没有这条门，改一处忘一处的后果同样是"上限变松、不会变红"。
    """

    from physics_engine import contact_pipeline

    assert contact_pipeline.RK4_STABILITY_RADIUS == RK4_STABILITY_RADIUS
    assert (
        contact_pipeline.EXPLICIT_EULER_STABILITY_RADIUS
        == EXPLICIT_EULER_STABILITY_RADIUS
    )


def test_the_declaration_says_the_two_bounds_are_independent_and_the_tighter_one_wins() -> None:
    """**必红**：声明里删掉"独立／更紧"那半句，本门红。

    形制照`contact_pipeline`那条同名的门：一条上限如果没有写明它只挡了一半物理，
    读的人会以为它是全部——research/17第五节抓到的正是这个错。
    """

    for text in (CONTACT_DYNAMICS_STEP_BOUND, TANGENTIAL_COEFFICIENT_BOUND):
        assert "独立" in text or "同一个不等式" in text, text
    assert "更紧" in CONTACT_DYNAMICS_STEP_BOUND
    assert "力矩装配" in CONTACT_DYNAMICS_STEP_BOUND
    assert "governing_assembly_step_bound" in CONTACT_DYNAMICS_STEP_BOUND
    #: `k_t`那条要写明两种症状是同一条判据，否则它读起来还是"两条不相干的边界"。
    assert "颤振" in TANGENTIAL_COEFFICIENT_BOUND
    assert "发散" in TANGENTIAL_COEFFICIENT_BOUND
    assert "下限" in TANGENTIAL_COEFFICIENT_BOUND


# ---------------------------------------------------------------------------
# 三、法向族与`contact_pipeline`同形：单点无杆臂时逐字退化
# ---------------------------------------------------------------------------


def test_a_single_point_with_no_lever_degenerates_to_the_pipeline_formula() -> None:
    """杆臂为零时本模块的``ω0``与``ζ``必须与`contact_pipeline`那条**逐字重合**。

    这是"两条上限同形不同分母"这句话的执行体：分母一样时两条式子必须给同一个数。
    取``r`极小而不是零：零杆臂让`I`那一块彻底解耦，仍能对上，但那样就没验到
    杆臂那一路的代码；这里取`r = 0`是因为退化点正是要判的那个点。
    """

    from physics_engine.contact_pipeline import contact_stiffness_step_bound

    mass, stiffness, damping = 0.4, 3.0e4, 12.0
    mine = contact_dynamics_step_bound(
        support_points_body_mm=((0.0, 0.0, 0.0),),
        mass_kg=mass,
        inertia_body_kg_mm2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        normal_stiffness_n_per_mm=stiffness,
        tangential_stiffness_n_per_mm=0.0,
        normal_damping_n_s_per_mm=damping,
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    theirs = contact_stiffness_step_bound(
        stiffness_n_per_mm=stiffness,
        effective_mass_kg=mass,
        damping_n_s_per_mm=damping,
    )
    assert mine.normal_omega0_rad_per_s == pytest.approx(
        theirs.omega0_rad_per_s, rel=4e-16
    )
    assert mine.normal_damping_ratio == pytest.approx(theirs.damping_ratio, rel=4e-16)
    assert mine.normal_stability_rate_per_s == pytest.approx(
        theirs.stability_rate_per_s, rel=4e-16
    )
    assert mine.step_bound_s == pytest.approx(theirs.step_bound_s, rel=4e-16)


def test_the_overdamped_branch_is_not_dropped() -> None:
    """**必红**：把过阻尼那一支省掉（速率恒取`ω0`），本门红。

    阻尼越大最快模态越快——"加阻尼总是更稳"是错的。
    """

    common = {
        "support_points_body_mm": ((0.0, 0.0, -2.0),),
        "mass_kg": 1.0,
        "inertia_body_kg_mm2": ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        "normal_stiffness_n_per_mm": 10.0,
        "tangential_stiffness_n_per_mm": 0.0,
        "plane_normal_body": (0.0, 0.0, 1.0),
    }
    light = contact_dynamics_step_bound(normal_damping_n_s_per_mm=0.0, **common)
    heavy = contact_dynamics_step_bound(normal_damping_n_s_per_mm=2.0, **common)
    assert light.normal_damping_ratio < 1.0 < heavy.normal_damping_ratio
    assert heavy.normal_stability_rate_per_s > light.normal_stability_rate_per_s
    ratio = heavy.normal_damping_ratio
    predicted = (ratio + math.sqrt(ratio * ratio - 1.0)) * heavy.normal_omega0_rad_per_s
    assert heavy.normal_stability_rate_per_s == pytest.approx(predicted, rel=1e-14)
    assert heavy.step_bound_s < light.step_bound_s


def test_the_bound_reports_which_family_governs() -> None:
    """**必红**：不报出处（或永远报同一个）本门红。两族各当一次家。"""

    inertia = _box_inertia()
    points = box_corner_points_mm(BOX_HALF)[:4]
    common = {
        "support_points_body_mm": points,
        "mass_kg": BOX_MASS,
        "inertia_body_kg_mm2": inertia.inertia_body_kg_mm2,
        "plane_normal_body": (0.0, 0.0, 1.0),
    }
    tangential = contact_dynamics_step_bound(
        normal_stiffness_n_per_mm=BOX_KN, tangential_stiffness_n_per_mm=BOX_KT, **common
    )
    assert tangential.governed_by == "tangential_damping"
    normal = contact_dynamics_step_bound(
        normal_stiffness_n_per_mm=1.0e6,
        tangential_stiffness_n_per_mm=1.0e-6,
        **common,
    )
    assert normal.governed_by == "normal_stiffness"
    assert normal.step_bound_s == pytest.approx(
        RK4_STABILITY_RADIUS / normal.normal_stability_rate_per_s, rel=1e-15
    )


def test_governing_assembly_step_bound_takes_the_tighter_one_and_names_it() -> None:
    """"取更紧的那个"这句话的执行体。三种情况各判一次。"""

    contact = governing_assembly_step_bound(
        contact_assembly_bound_s=1.0e-4, rotational_mode_bound_s=1.0e-3
    )
    assert contact.governed_by == "contact_assembly"
    #: **数字也要判**：注错第C6轮抓到的空门——把`min`写成`max`时，
    #: 只判`governed_by`的门全绿，而返回的上限是**更松**的那一条。
    #: "报出处"与"报对数"是两件事，两件都要有门。
    assert contact.step_bound_s == 1.0e-4
    rotational = governing_assembly_step_bound(
        contact_assembly_bound_s=1.0e-3, rotational_mode_bound_s=1.0e-4
    )
    assert rotational.governed_by == "rotational_mode"
    assert rotational.step_bound_s == 1.0e-4
    both = governing_assembly_step_bound(
        contact_assembly_bound_s=2.0e-4, rotational_mode_bound_s=2.0e-4
    )
    assert both.governed_by == "both"
    assert both.step_bound_s == 2.0e-4


def test_the_window_endpoint_convention_is_closed_on_both_sides() -> None:
    """**必红**：把`admits`的端点判反（`>`写成`>=`），本门红。

    注错第C17轮抓到的空门：所有别的门用的都是"离端点还有距离"的`k_t`，
    于是端点怎么判**一条门都看不见**。而`TangentialCoefficientWindow`的docstring
    明写着上界本身是允许的那一档——**一条只写在文档里、没有门看着的口径会漂**。
    """

    window = tangential_coefficient_window(
        support_points_body_mm=((0.0, 0.0, -1.0),),
        mass_kg=1.0,
        inertia_body_kg_mm2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        step_s=1.0e-4,
        plane_normal_body=(0.0, 0.0, 1.0),
        required_tangential_force_n=1.0,
        allowed_creep_mm_per_s=1.0,
    )
    assert window.admits(window.upper_bound_n_s_per_mm)
    assert not window.admits(math.nextafter(window.upper_bound_n_s_per_mm, math.inf))
    assert window.lower_bound_n_s_per_mm is not None
    assert window.admits(window.lower_bound_n_s_per_mm)
    assert not window.admits(math.nextafter(window.lower_bound_n_s_per_mm, -math.inf))


# ---------------------------------------------------------------------------
# 四、支承组：候选池不是"全部声明过的点"
# ---------------------------------------------------------------------------


def test_scanning_all_eight_corners_as_one_group_is_too_tight_by_a_measured_amount() -> None:
    """八个角当一组比按面分组紧56%——**那会把一条跑得好好的案例判成不稳定**。

    翻倒案例的失稳侧声明的正是八个角（翻过去之后贴地的是另一个面），
    而八个角永远不可能同时贴同一个平面。这条门把"分组是形制的一部分"钉住。
    """

    inertia = _box_inertia()
    corners = box_corner_points_mm(BOX_HALF)
    faces = tuple(
        tuple(corner for corner in corners if (corner[axis] > 0.0) is positive)
        for axis in range(3)
        for positive in (False, True)
    )
    kwargs = {
        "mass_kg": BOX_MASS,
        "inertia_body_kg_mm2": inertia.inertia_body_kg_mm2,
        "normal_stiffness_n_per_mm": BOX_KN,
        "tangential_stiffness_n_per_mm": BOX_KT,
        "normal_damping_n_s_per_mm": BOX_CN,
    }
    grouped = tightest_step_bound(faces, **kwargs)
    lumped = contact_dynamics_step_bound(support_points_body_mm=corners, **kwargs)
    assert lumped.step_bound_s < grouped.step_bound_s
    assert grouped.step_bound_s / lumped.step_bound_s == pytest.approx(1.5609, rel=1e-4)
    #: 定上限的是底面／顶面那一组（`z`那一对），不是侧面。
    assert grouped.governing_group_index in (4, 5)
    assert len(grouped.bounds) == 6
    #: 八角一组的上限比案例实际用的`dt = 2e-4`还紧——**这就是"太紧"的代价**。
    assert lumped.step_bound_s < 2.0e-4 < grouped.step_bound_s


# ---------------------------------------------------------------------------
# 五、`k_t`窗口：回归判据 —— 声明要能预言0082那次发散
# ---------------------------------------------------------------------------


def test_the_kt_window_predicts_the_measured_divergence_of_the_tipping_case() -> None:
    """**回归判据**：0082第五节实测`k_t = 0.03`、`dt = 4e-4`当场发散。

    声明必须**在两侧都对**：`4e-4`上说不许、`2e-4`上说可以。
    只判一侧的门挡不住一个"永远说不许"的实现。
    """

    inertia = _box_inertia()
    points = box_corner_points_mm(BOX_HALF)[:4]
    common = {
        "support_points_body_mm": points,
        "mass_kg": BOX_MASS,
        "inertia_body_kg_mm2": inertia.inertia_body_kg_mm2,
        "plane_normal_body": (0.0, 0.0, 1.0),
    }
    coarse = tangential_coefficient_window(step_s=4.0e-4, **common)
    fine = tangential_coefficient_window(step_s=2.0e-4, **common)
    assert not coarse.admits(BOX_KT), (
        f"声明放行了0082实测发散的那组：上界{coarse.upper_bound_n_s_per_mm!r}"
    )
    assert fine.admits(BOX_KT)
    assert coarse.upper_bound_n_s_per_mm == pytest.approx(0.0255974, rel=1e-5)
    assert fine.upper_bound_n_s_per_mm == pytest.approx(0.0511949, rel=1e-5)
    #: 案例声明的`k_t`比粗档上限高17%——**那17%就是那次发散**。
    assert BOX_KT / coarse.upper_bound_n_s_per_mm == pytest.approx(1.172, rel=1e-3)


def test_the_two_point_group_really_is_allowed_the_coarse_step_the_case_uses() -> None:
    """二分那一组用的正是`dt = 4e-4`，而四点组不许——**分辨力就在这里**。

    一条只看`k_t`与`dt`、不看支承点几何的上限，会把这两组判成同一个答案，
    而实测它们差一倍（两点组`σ = 136`、四点组`σ = 272`，单位1/kg）。
    """

    inertia = _box_inertia()
    two_points = ((-BOX_HALF[0], 0.0, -BOX_HALF[2]), (BOX_HALF[0], 0.0, -BOX_HALF[2]))
    window = tangential_coefficient_window(
        support_points_body_mm=two_points,
        mass_kg=BOX_MASS,
        inertia_body_kg_mm2=inertia.inertia_body_kg_mm2,
        step_s=4.0e-4,
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    assert window.admits(BOX_KT)
    assert window.tangential_mobility_per_kg == pytest.approx(136.0, rel=1e-12)


def test_the_kt_window_admits_the_rolling_ball_and_rejects_its_chatter_region() -> None:
    """滚球那条：`k_t = 50`在窗口里，实测进颤振区的`5e3`与`5e5`不在。

    **两条案例共用同一条判据**——0082说"翻倒那条是显式稳定率"、
    滚球案例说"那条是颤振区"，而它们是同一个`λh > 稳定区半径`。
    """

    window = tangential_coefficient_window(
        support_points_body_mm=((0.0, 0.0, -BALL_RADIUS),),
        mass_kg=BALL_MASS,
        inertia_body_kg_mm2=_ball_inertia(),
        step_s=1.0e-6,
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    assert window.upper_bound_n_s_per_mm == pytest.approx(795.714, rel=1e-5)
    assert window.admits(BALL_KT)
    assert not window.admits(5.0e3)
    assert not window.admits(5.0e5)


def test_the_creep_lower_bound_can_make_the_window_empty_and_that_is_the_answer() -> None:
    """窗口空掉是一条**真结论**：这个步长下既想稳又想粘住办不到，要改的是`h`。

    翻倒那条案例在`dt = 4e-4`上正是空的，而案例最后的办法也正是把步长压到`2e-4`。
    蠕滑那一侧的输入取案例自己声明的稳态蠕滑（`k_t = 0.03`上1.542 mm/s，
    于是要传的切向力是`2·k_t·v ≈ 0.0925 N`），允许蠕滑取2 mm/s。
    """

    inertia = _box_inertia()
    points = box_corner_points_mm(BOX_HALF)[:4]
    required = 2.0 * BOX_KT * 1.542040
    common = {
        "support_points_body_mm": points,
        "mass_kg": BOX_MASS,
        "inertia_body_kg_mm2": inertia.inertia_body_kg_mm2,
        "plane_normal_body": (0.0, 0.0, 1.0),
        "required_tangential_force_n": required,
        "allowed_creep_mm_per_s": 2.0,
    }
    coarse = tangential_coefficient_window(step_s=4.0e-4, **common)
    fine = tangential_coefficient_window(step_s=2.0e-4, **common)
    assert coarse.is_empty, (
        f"上界{coarse.upper_bound_n_s_per_mm!r}、下界{coarse.lower_bound_n_s_per_mm!r}"
    )
    assert not fine.is_empty
    assert coarse.lower_bound_n_s_per_mm == pytest.approx(0.046261, rel=1e-4)
    assert creep_resolution_lower_bound(
        required_tangential_force_n=required, allowed_creep_mm_per_s=2.0
    ) == pytest.approx(coarse.lower_bound_n_s_per_mm, rel=0, abs=0)


def test_the_creep_bound_needs_both_inputs_or_neither() -> None:
    """只给一个等于让本函数替你猜另一个——**没有一个普适的『够小』**。"""

    common = {
        "support_points_body_mm": ((0.0, 0.0, -1.0),),
        "mass_kg": 1.0,
        "inertia_body_kg_mm2": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "step_s": 1.0e-4,
    }
    with pytest.raises(ContactDynamicsError, match="要么都给"):
        tangential_coefficient_window(required_tangential_force_n=1.0, **common)
    with pytest.raises(ContactDynamicsError, match="要么都给"):
        tangential_coefficient_window(allowed_creep_mm_per_s=1.0, **common)
    assert tangential_coefficient_window(**common).lower_bound_n_s_per_mm is None


# ---------------------------------------------------------------------------
# 六、失败关闭
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"support_points_body_mm": ()}, "空的"),
        ({"mass_kg": 0.0}, "mass_kg"),
        ({"mass_kg": float("nan")}, "mass_kg"),
        ({"normal_stiffness_n_per_mm": 0.0}, "normal_stiffness"),
        ({"normal_stiffness_n_per_mm": -1.0}, "normal_stiffness"),
        ({"tangential_stiffness_n_per_mm": -1.0}, "tangential_stiffness"),
        ({"normal_damping_n_s_per_mm": -1.0}, "normal_damping"),
        ({"stability_radius": 0.0}, "stability_radius"),
        (
            {"inertia_body_kg_mm2": ((1.0, 2.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
            "不对称",
        ),
        (
            {"inertia_body_kg_mm2": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
            "正定",
        ),
        ({"inertia_body_kg_mm2": ((0.0, 0.0, 0.0),) * 3}, "全零"),
    ],
)
def test_the_bound_fails_closed_on_bad_inputs(kwargs, pattern) -> None:
    """**不返回"尽力而为"的上限**——一个错的上限看起来像一道门。"""

    base = {
        "support_points_body_mm": ((0.0, 0.0, -1.0),),
        "mass_kg": 1.0,
        "inertia_body_kg_mm2": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "normal_stiffness_n_per_mm": 100.0,
        "tangential_stiffness_n_per_mm": 1.0,
    }
    with pytest.raises(ContactDynamicsError, match=pattern):
        contact_dynamics_step_bound(**{**base, **kwargs})


def test_the_sphere_entry_refuses_an_anisotropic_inertia() -> None:
    """球那一档的整套形制建立在"它是球"上——非各向同性当场失败关闭。"""

    with pytest.raises(ContactDynamicsError, match="各向同性"):
        sphere_plane_step_bound(
            radius_mm=1.0,
            mass_kg=1.0,
            inertia_body_kg_mm2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)),
            normal_stiffness_n_per_mm=1.0,
            tangential_stiffness_n_per_mm=1.0,
        )


def test_the_sphere_entry_agrees_with_the_general_one() -> None:
    """球那一档只是通用那条的一个便利入口，两条必须给同一个数。"""

    convenient = sphere_plane_step_bound(
        radius_mm=BALL_RADIUS,
        mass_kg=BALL_MASS,
        inertia_body_kg_mm2=_ball_inertia(),
        normal_stiffness_n_per_mm=BALL_KN,
        tangential_stiffness_n_per_mm=BALL_KT,
    )
    general = contact_dynamics_step_bound(
        support_points_body_mm=((0.0, 0.0, -BALL_RADIUS),),
        mass_kg=BALL_MASS,
        inertia_body_kg_mm2=_ball_inertia(),
        normal_stiffness_n_per_mm=BALL_KN,
        tangential_stiffness_n_per_mm=BALL_KT,
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    assert convenient.step_bound_s == general.step_bound_s
    #: 滚球那组声明输入上是**切向**族当家：`ω0 = √(1000·5e5·1) = 22360`，
    #: 而切向速率是`1000·50·3.5 = 175000`——**将近8倍**。
    #: 案例页第四节第1条把`k_t = 50`说成"被实测逼出来的"，本行给出它的机理。
    assert convenient.governed_by == "tangential_damping"
    assert convenient.normal_omega0_rad_per_s == pytest.approx(22360.68, rel=1e-6)
    assert convenient.tangential_stability_rate_per_s == pytest.approx(175000.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 七、必红那一类：**声明说安全而实际发散**判不出来的反例
# ---------------------------------------------------------------------------


def _frame(theta: float):
    return (
        (math.sin(theta), 0.0, math.cos(theta)),
        (math.cos(theta), 0.0, -math.sin(theta)),
    )


def _settled_box_start(theta: float, tangential: float):
    """闭式静态位形，与`tests/cases/test_box_tipping_threshold.py`同一组式子。"""

    weight = BOX_MASS * GRAVITY / 1000.0
    half_w, _, half_h = BOX_HALF
    mean = weight * math.cos(theta) / 4.0
    half = weight * half_h * math.sin(theta) / (4.0 * half_w)
    creep = (weight * math.sin(theta) - 2.0 * BOX_MU * (mean - half)) / (2.0 * tangential)
    delta_up, delta_down = (mean - half) / BOX_KN, (mean + half) / BOX_KN
    beta = math.asin((delta_down - delta_up) / (2.0 * half_w))
    height = half_h * math.cos(beta) - 0.5 * (delta_up + delta_down)
    normal, downhill = _frame(theta)
    alpha = theta + beta
    return make_state(
        position_mm=tuple(height * axis for axis in normal),
        velocity_mm_per_s=tuple(creep * axis for axis in downhill),
        attitude_xyzw=(0.0, math.sin(0.5 * alpha), 0.0, math.cos(0.5 * alpha)),
    )


def _run_box(tangential: float, dt_s: float, steps: int) -> float:
    """跑一趟，交出**横坡**角速度分量的末态量级。

    判横坡分量而不是判倾角：本构型对横坡是对称的，那两个分量在稳定时
    恒在舍入量级（案例判到1e-10）。**它是发散最早、最干净的可观测量。**
    """

    normal, _ = _frame(BOX_THETA)
    force, torque = support_points_plane_callbacks(
        support_points_body_mm=box_corner_points_mm(BOX_HALF)[:4],
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=normal,
        normal_stiffness_n_per_mm=BOX_KN,
        tangential_stiffness_n_per_mm=tangential,
        friction_coefficient=BOX_MU,
        gravity_world_n=(0.0, 0.0, -BOX_MASS * GRAVITY / 1000.0),
        normal_damping_n_s_per_mm=BOX_CN,
    )
    final, _diagnostics = integrate_free_flight(
        RK4_BODY,
        state=_settled_box_start(BOX_THETA, tangential),
        inertia=_box_inertia(),
        dt_s=dt_s,
        steps=steps,
        force_world_n=force,
        torque_body_nmm=torque,
    )
    return max(abs(final.vector[6]), abs(final.vector[8]))


def test_must_be_red_a_loosened_bound_would_wave_through_a_run_that_really_diverges() -> None:
    """**必红的那一条**：声明说安全、实跑发散，这里当场抓住。

    做法是**真跑**：
    * 声明拒绝的那组（`k_t = 0.03`、`dt = 4e-4`）实测末态横坡角速度到**1e-2量级**；
    * 声明放行的那组（同`k_t`、`dt = 2e-4`）实测仍在**1e-14量级**。

    两者差12个数量级，中间没有任何"要不要算发散"的判断余地。
    **一个把上限算松了的实现**（例如只算偏航模态那条`Σr_⊥²/I_zz`，
    它给的上限是0.029、放行0.03）会在第一条上说"安全"，
    而这条门读的是真实积分的结果，**它不听声明的**。
    """

    inertia = _box_inertia()
    points = box_corner_points_mm(BOX_HALF)[:4]
    common = {
        "support_points_body_mm": points,
        "mass_kg": BOX_MASS,
        "inertia_body_kg_mm2": inertia.inertia_body_kg_mm2,
        "plane_normal_body": (0.0, 0.0, 1.0),
    }
    coarse = tangential_coefficient_window(step_s=4.0e-4, **common)
    fine = tangential_coefficient_window(step_s=2.0e-4, **common)

    diverged = _run_box(BOX_KT, 4.0e-4, 1500)
    settled = _run_box(BOX_KT, 2.0e-4, 1500)
    assert diverged > 1.0e-3, f"这一组本该发散，实测{diverged!r}——回归判据的前提塌了"
    assert settled < 1.0e-10, f"这一组本该稳定，实测{settled!r}"
    assert not coarse.admits(BOX_KT)
    assert fine.admits(BOX_KT)

    #: **把上限故意放松到只算偏航模态**——那正是0082登记的那条式子。
    #: 它会放行那个真的发散的组，于是"声明说安全而实际发散"这件事在这里可判。
    yaw_only = 4.0 * (BOX_HALF[0] ** 2 + BOX_HALF[1] ** 2) / inertia.inertia_body_kg_mm2[2][2]
    loosened = RK4_STABILITY_RADIUS / (1000.0 * 4.0e-4 * yaw_only)
    assert loosened > coarse.upper_bound_n_s_per_mm
    assert loosened > BOX_KT * 0.96, (
        "偏航那条式子给的上限本该几乎放行0.03（实测0.0290 vs 0.03），"
        "这条门的反例前提变了就要重写"
    )


def test_the_declared_bound_is_conservative_and_by_how_much() -> None:
    """上限是**保守的**，而保守多少要被量出来写下来。

    实测发散起点在`k_t ≈ 0.02831`（`λh ≈ 3.08`），声明的上限是0.02560
    （`λh = 2.785`）——**保守10.6%**。理由写在决策0087第三节：
    上坡侧那一对支承点的切向力已经饱和在摩擦锥上，那两点不再是线性阻尼器。

    **这条门判的是"保守"这个方向**（声明的上限不许比实测起点还松），
    不判那10.6%这个数——一个数值细节上的漂移不该让它红。
    """

    inertia = _box_inertia()
    window = tangential_coefficient_window(
        support_points_body_mm=box_corner_points_mm(BOX_HALF)[:4],
        mass_kg=BOX_MASS,
        inertia_body_kg_mm2=inertia.inertia_body_kg_mm2,
        step_s=4.0e-4,
        plane_normal_body=(0.0, 0.0, 1.0),
    )
    just_inside = window.upper_bound_n_s_per_mm * 0.999
    assert window.admits(just_inside)
    assert _run_box(just_inside, 4.0e-4, 1500) < 1.0e-10, (
        "声明放行的最大那个`k_t`实跑就发散了——上限不再是保守的，那是最坏的一类错"
    )
