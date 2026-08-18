"""三球金字塔含力矩平衡那一支：``μc = 2 − √3``（决策0079，兑现0063第五节第1步）。

金标来自`cases/three_sphere_pyramid_rotational/generate_oracle.py`的
**`Q(√3)`精确有理算术**——不是把research/15的数抄下来，是在仓内重算一遍。

本文件同时装着**斜面那一半**：球在斜面上的约束反力矩``W·R·sinα``（闭式），
以及"放开转动就没有静平衡"这条定性判据。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from physics_engine.contact.penalty import PenaltyNormalContact, PenaltySphereContact
from physics_engine.energies import (
    EnergyContext,
    EnergyRegistry,
    PointLoad,
    UniformGravity,
)
from physics_engine.rotation import (
    MaterialPoint,
    MaterialPointStickSpring,
    RotationStickCoupling,
    StickSpring,
    build_rigid_body_layout,
)
from physics_engine.solve import (
    solve_equilibrium,
    tangent_stiffness_is_positive_definite,
)

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "three_sphere_pyramid_rotational"

RADIUS_MM = 10.0
MASS_KG = 1.5
GRAVITY_MM_S2 = 9810.0
WEIGHT_N = MASS_KG * GRAVITY_MM_S2 / 1000.0
SQRT3 = math.sqrt(3.0)


@pytest.fixture(scope="module")
def oracles() -> dict:
    document = json.loads((CASE / "oracle.json").read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in document["oracles"]}


def _residual_tolerance(stiffness: float) -> float:
    """**绝对残差的可达地板随罚刚度线性上升**，所以容差必须跟着刚度走。

    实测地板约``7e-15·k``（力的绝对噪声 = 刚度 × 位置的舍入，而位置是``O(10 mm)``）。
    取``4e-14·k``约6倍余量；低刚度端用``1e-8``托底，那里地板由载荷尺度而非刚度定。
    **写成一个有名字的函数而不是一个魔数**：它是本案例最容易被下一个人调松的地方。
    """

    return max(1.0e-8, 4.0e-14 * stiffness)


def solve_pyramid(stiffness: float, *, pin_top_spin: bool = True, max_iterations: int = 60):
    """三球金字塔的准静态解。三体各三个平面自由度（``x``、``z``、``θ_y``）。

    ## 为什么钉住顶球自旋（**规范固定**，不是改问题）

    三个球全都可以自转时，切线刚度有一个一维零空间：**整座金字塔沿地面整体侧滚**。
    9个自由度、4个法向 + 4个切向共8条独立约束 ⟹ 差一条；这与research/15第3.2节
    "9方程/8未知、精确秩8"是同一件事的对偶说法。实测约化Hessian条件数**7.86e17**
    （最小特征值8.73e-10、最大6.86e8）——数值上就是奇异的。

    **零模上重力不做功**（纯水平平移），所以平衡解**存在**、只是不唯一：
    它是一条平衡线而不是一个点。钉住顶球``θ_y``挑掉线上的一个点，
    **而"这不改问题"是可判的**：钉住处的约束反力矩必须恰为``0.0``。
    两条断言分别在`test_the_whole_assembly_rolling_sideways_is_a_zero_mode`
    与`test_the_top_sphere_moment_is_exactly_redundant`。
    """

    ground_penetration = 1.5 * WEIGHT_N / stiffness
    sphere_penetration = 0.5 * WEIGHT_N / stiffness
    base_z = RADIUS_MM - ground_penetration
    positions = (
        0.0, 0.0, base_z + SQRT3 * RADIUS_MM - 2.0 * sphere_penetration / SQRT3,
        RADIUS_MM, 0.0, base_z,
        -RADIUS_MM, 0.0, base_z,
    )
    layout = build_rigid_body_layout(
        layout_id="layout/three_sphere_pyramid_rotational",
        node_count=3,
        rotating_bodies=(0, 1, 2),
    )
    initial = layout.initial_vector(positions)

    normal_right = (-0.5, 0.0, SQRT3 / 2.0)
    normal_left = (0.5, 0.0, SQRT3 / 2.0)
    up = (0.0, 0.0, 1.0)
    springs = (
        StickSpring(
            first=MaterialPoint(
                0, tuple(-RADIUS_MM * v for v in normal_right), layout.rotation_base(0)
            ),
            second=MaterialPoint(
                1, tuple(RADIUS_MM * v for v in normal_right), layout.rotation_base(1)
            ),
            normal=normal_right, stiffness_n_per_mm=stiffness, anchor_mm=(0.0, 0.0, 0.0),
        ),
        StickSpring(
            first=MaterialPoint(
                0, tuple(-RADIUS_MM * v for v in normal_left), layout.rotation_base(0)
            ),
            second=MaterialPoint(
                2, tuple(RADIUS_MM * v for v in normal_left), layout.rotation_base(2)
            ),
            normal=normal_left, stiffness_n_per_mm=stiffness, anchor_mm=(0.0, 0.0, 0.0),
        ),
        StickSpring(
            first=MaterialPoint(1, (0.0, 0.0, -RADIUS_MM), layout.rotation_base(1)),
            normal=up, stiffness_n_per_mm=stiffness, anchor_mm=(RADIUS_MM, 0.0, 0.0),
        ),
        StickSpring(
            first=MaterialPoint(2, (0.0, 0.0, -RADIUS_MM), layout.rotation_base(2)),
            normal=up, stiffness_n_per_mm=stiffness, anchor_mm=(-RADIUS_MM, 0.0, 0.0),
        ),
    )
    sphere_contact = PenaltySphereContact(
        pairs=((0, 1, 2 * RADIUS_MM, stiffness), (0, 2, 2 * RADIUS_MM, stiffness))
    )
    ground_contact = PenaltyNormalContact(
        planes=(
            (1, (0.0, 0.0, 0.0), up, stiffness, RADIUS_MM),
            (2, (0.0, 0.0, 0.0), up, stiffness, RADIUS_MM),
        )
    )
    stick = MaterialPointStickSpring(springs=springs)
    coupling = RotationStickCoupling(springs=springs)
    registry = EnergyRegistry(
        terms=(UniformGravity(), sphere_contact, ground_contact, stick, coupling)
    )
    context = EnergyContext(
        context_id="context/three_sphere_pyramid_rotational",
        node_masses_kg=(MASS_KG,) * 3,
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    fixed = {3 * node + 1 for node in range(3)}
    for body in range(3):
        base = layout.rotation_base(body)
        fixed |= {base, base + 2}
    if pin_top_spin:
        fixed.add(layout.rotation_base(0) + 1)

    result = solve_equilibrium(
        registry, context, layout.layout, initial,
        fixed_indices=frozenset(fixed),
        residual_tol_n=_residual_tolerance(stiffness),
        max_iterations=max_iterations,
    )
    return {
        "layout": layout, "registry": registry, "context": context,
        "result": result, "fixed": frozenset(fixed),
        "sphere_contact": sphere_contact, "ground_contact": ground_contact,
        "coupling": coupling,
    }


def _magnitude(vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _forces(solved: dict) -> dict[str, float]:
    state = solved["result"].state
    tangential = solved["coupling"].tangential_force_n(state)
    return {
        "sphere_normal": solved["sphere_contact"].contact_force_n(state)[0],
        "sphere_tangential": _magnitude(tangential[0]),
        "ground_normal": solved["ground_contact"].normal_force_n(state)[0],
        "ground_tangential": _magnitude(tangential[2]),
    }


class TestExactRationalGoldStandard:
    """金标本身：`Q(√3)`精确算术。**它在生成期就断言过，这里判它没有漂。**"""

    def test_the_generator_reproduces_the_independently_recomputed_solution(self) -> None:
        """精确解的四个量必须与research/15第3.2节逐位相同。

        research/15是**另一个证人**（另一份实现、另一次装配），
        本仓的生成器是第一个。两者逐位相同才叫复核。
        """

        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "pyramid_rot_oracle", CASE / "generate_oracle.py"
        )
        module = importlib.util.module_from_spec(spec)
        #: `dataclasses`在解析注解时要从`sys.modules`里取模块字典，
        #: 所以动态加载的模块**必须先注册再exec**。
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        solution = module.exact_solution()
        assert solution["F1"].to_float() == 0.5
        assert solution["N1"].to_float() == 1.5
        #: research/15第3.2节表里的十五位数字。
        assert f"{solution['f1'].to_float():.15f}" == "0.133974596215561"
        assert f"{solution['T1'].to_float():.15f}" == "-0.133974596215561"
        ratio = solution["f1"] / solution["F1"]
        assert (ratio - (module.rational(2) - module.SQRT3)).is_zero(), (
            "|f|/F 与 2−√3 不是**精确**相等——research/15说的是逐位，不是'差很小'"
        )

    def test_the_two_branches_differ_by_the_documented_factor(self, oracles) -> None:
        """``(2−√3)/(1/(3√3)) = 1.392305``——0063第二节那句"偏约四成"的算术形式。"""

        entry = oracles["oracle:pyramid_rot/rotation_free_variant_differs"]
        expected = entry["expected"]["critical_friction_ratio"]
        measured = (2.0 - SQRT3) / (1.0 / (3.0 * SQRT3))
        assert measured == pytest.approx(expected, rel=1e-12)
        assert f"{measured:.6f}" == "1.392305"


class TestPyramidWithRotation:
    """引擎实测对精确金标。**逐档对上**，不是"数量级差不多"。"""

    def test_the_static_decomposition_matches_the_exact_solution(self, oracles) -> None:
        entry = oracles["oracle:pyramid_rot/force_decomposition"]
        stiffness = entry["inputs"]["stiffness_n_per_mm"]
        solved = solve_pyramid(stiffness)
        assert solved["result"].converged, solved["result"].reason
        assert solved["result"].backtracks == 0, (
            "出现了线搜索回溯——回溯次数高本身就是'这个问题不好'的信号，"
            "不许被'最终收敛了'盖过去（spec/12第4.3节）"
        )
        forces = _forces(solved)
        expected = entry["expected"]
        tolerance = entry["tolerances"]["ground_normal_n"]["rel"]
        assert forces["sphere_normal"] == pytest.approx(
            expected["sphere_contact_force_n"], rel=tolerance
        )
        assert forces["sphere_tangential"] == pytest.approx(
            expected["sphere_tangential_n"], rel=tolerance
        )
        assert forces["ground_normal"] == pytest.approx(
            expected["ground_normal_n"], rel=tolerance
        )
        assert forces["ground_tangential"] == pytest.approx(
            expected["ground_tangential_n"], rel=tolerance
        )

    def test_it_is_not_the_frictionless_variant(self) -> None:
        """**必红专防**：带力矩平衡的解**不许**落回``1/(3√3)``那一支。

        没有这一条，一个把力矩项接错成恒零的实现会安静地给出旧答案，
        而旧答案也有一条金标，于是那条门是绿的。**这就是0063说的
        "它只是安静地给出一个偏了四成的数"。**
        """

        solved = solve_pyramid(2.0e6)
        forces = _forces(solved)
        critical = forces["sphere_tangential"] / forces["sphere_normal"]
        assert abs(critical - 1.0 / (3.0 * SQRT3)) > 0.05, (
            "临界摩擦落回了无摩擦变体的0.19245——力矩平衡没有真的参与"
        )
        assert forces["sphere_normal"] != pytest.approx(
            WEIGHT_N / SQRT3, rel=1e-3
        ), "球-球法向落回了W/√3——那是没有力矩平衡的那一支"

    def test_the_symmetry_is_solved_not_assumed(self) -> None:
        """两个底球的``θ_y``必须**等大反号**——对称性是解出来的，不是装配进去的。"""

        solved = solve_pyramid(2.0e6)
        state = solved["result"].state
        layout = solved["layout"]
        right = state.vector[layout.rotation_base(1) + 1]
        left = state.vector[layout.rotation_base(2) + 1]
        assert right != 0.0, "底球根本没转——转动自由度没有被激活"
        assert right == pytest.approx(-left, rel=1e-9)

    def test_the_equilibrium_is_a_minimum_not_a_saddle(self) -> None:
        """`solve_equilibrium`收敛只说明``∇U = 0``，**不说明那是不是极小**。"""

        solved = solve_pyramid(2.0e6)
        assert tangent_stiffness_is_positive_definite(
            solved["registry"], solved["context"], solved["result"].state,
            fixed_indices=solved["fixed"],
        )

    def test_the_whole_assembly_rolling_sideways_is_a_zero_mode(self) -> None:
        """**这是"为什么钉住顶球自旋"的全部证据。**

        三个球都能自转时，切线刚度有一个**一维零空间**：整座金字塔沿地面
        **整体侧滚**——三个球心同步平移``R·dθ``，两底球同向自转、顶球反向自转，
        四个接触点一个都不滑，全部法向间隙不变，**而重力一点功都不做**
        （纯水平平移，质心不升不降）。

        它的三条后果各判一次：

        1. 能量沿这个方向是**平的**（二阶量为零）；
        2. 梯度在这个方向上的投影为零 ⟹ **平衡解存在，只是不唯一**
           （是一条平衡线，不是一个点）；
        3. 因此把顶球``θ_y``钉住是一次**规范固定**而不是改问题——
           钉住处的约束反力矩恰为``0.0``（下一条门）。

        **注意它不是"金字塔会散架"**：起草时我按那个猜想写过一条门，
        实测零模是侧滚而不是外滚，重力沿它不做功。**猜想与实测不符时改的是猜想。**
        """

        solved = solve_pyramid(2.0e6, pin_top_spin=False, max_iterations=40)
        assert solved["result"].converged, (
            "整体侧滚零模上重力不做功，所以平衡解**存在**——不收敛说明别的地方坏了"
        )
        layout = solved["layout"]
        state = solved["result"].state
        registry, context = solved["registry"], solved["context"]

        #: 解析构造那个零模：``dx = R·dθ``，两底球同号、顶球反号。
        spin = 1.0e-6
        direction = [0.0] * len(state.vector)
        for node in range(3):
            direction[3 * node] = RADIUS_MM * spin
        direction[layout.rotation_base(0) + 1] = -spin
        direction[layout.rotation_base(1) + 1] = spin
        direction[layout.rotation_base(2) + 1] = spin

        energy, gradient, _ = registry.total(state, context, need_gradient=True)
        moved = state.with_vector(
            tuple(state.vector[i] + direction[i] for i in range(len(direction)))
        )
        moved_energy, _, _ = registry.total(moved, context)

        #: (1) 能量是平的。参照量取"同样大小的一步走在受约束方向上"要付的能量，
        #: 这里用罚刚度 × 位移² 估：``k·(R·spin)² = 2e6 × 1e-10 = 2e-4 N·mm``。
        scale = 2.0e6 * (RADIUS_MM * spin) ** 2
        assert abs(moved_energy - energy) < 1.0e-6 * scale, (
            f"沿整体侧滚方向能量变了{moved_energy - energy!r}——那不是零模"
        )

        #: (2) 梯度在零模上的投影为零 ⟹ 方程组相容。
        projection = sum(gradient[i] * direction[i] for i in range(len(direction)))
        gradient_norm = math.sqrt(sum(value * value for value in gradient))
        step_norm = math.sqrt(sum(value * value for value in direction))
        assert abs(projection) < 1.0e-6 * gradient_norm * step_norm, (
            "梯度在零模上有分量 ⟹ 方程组不相容 ⟹ 平衡解根本不存在"
        )

        #: (3) 必红那一半：随便换一个方向，能量**必须**真的变。
        #: 没有这一条，上面两条在"能量项全接错成常数"时也是绿的。
        askew = list(direction)
        askew[layout.rotation_base(1) + 1] = -spin  # 右底球反着转 ⟹ 接触点开始滑
        skew_energy, _, _ = registry.total(
            state.with_vector(
                tuple(state.vector[i] + askew[i] for i in range(len(askew)))
            ),
            context,
        )
        assert abs(skew_energy - energy) > 0.01 * scale, (
            "把零模改坏之后能量还是不变——说明这些能量项根本没连上转动块"
        )

    def test_the_penalty_compliance_is_first_order(self, oracles) -> None:
        """刚度涨10倍偏差降约10倍——**这条比"偏差很小"强得多**：

        它证明那个偏差**是模型的柔度而不是实现的错误**。
        """

        entry = oracles["oracle:pyramid_rot/compliance_is_first_order"]
        deviations = []
        for stiffness in entry["inputs"]["stiffnesses_n_per_mm"]:
            solved = solve_pyramid(stiffness)
            assert solved["result"].converged
            forces = _forces(solved)
            critical = max(
                forces["sphere_tangential"] / forces["sphere_normal"],
                forces["ground_tangential"] / forces["ground_normal"],
            )
            deviations.append(abs(critical - (2.0 - SQRT3)) / (2.0 - SQRT3))
        assert entry["expected"]["deviations_shrink"]
        for first, second in zip(deviations, deviations[1:], strict=False):
            assert second < first, f"偏差没有逐档减小：{deviations}"
        low = entry["expected"]["deviation_ratio_low"]
        high = entry["expected"]["deviation_ratio_high"]
        for first, second in zip(deviations, deviations[1:], strict=False):
            assert low <= first / second <= high, (
                f"收敛阶不在[{low}, {high}]：{[a/b for a, b in zip(deviations, deviations[1:], strict=False)]}"
            )


class TestSphereOnAnIncline:
    """斜面那一半：**准静态路径能给的是约束反力矩**，不是滚动加速度。

    ``a = (5/7)g·sinα``是0063第五节**第3步**（路径B、时间积分）的判据，
    准静态路径给不了它——本组不假装能给。见case.md第六节。
    """

    @staticmethod
    def _solve(angle_deg: float, stiffness: float, *, release_spin: bool = False):
        alpha = math.radians(angle_deg)
        mass = 2.0
        weight = mass * GRAVITY_MM_S2 / 1000.0
        normal = (math.sin(alpha), 0.0, math.cos(alpha))
        layout = build_rigid_body_layout(
            layout_id="layout/incline_rotational", node_count=1, rotating_bodies=(0,)
        )
        centre = tuple(
            (RADIUS_MM - weight * math.cos(alpha) / stiffness) * value for value in normal
        )
        spring = StickSpring(
            first=MaterialPoint(
                0, tuple(-RADIUS_MM * v for v in normal), layout.rotation_base(0)
            ),
            normal=normal, stiffness_n_per_mm=stiffness, anchor_mm=(0.0, 0.0, 0.0),
        )
        coupling = RotationStickCoupling(springs=(spring,))
        registry = EnergyRegistry(
            terms=(
                UniformGravity(),
                PenaltyNormalContact(
                    planes=((0, (0.0, 0.0, 0.0), normal, stiffness, RADIUS_MM),)
                ),
                MaterialPointStickSpring(springs=(spring,)),
                coupling,
            )
        )
        context = EnergyContext(
            context_id="context/incline_rotational", node_masses_kg=(mass,),
            gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
        )
        base = layout.rotation_base(0)
        fixed = {1, base, base + 2}
        if not release_spin:
            fixed.add(base + 1)
        result = solve_equilibrium(
            registry, context, layout.layout, layout.initial_vector(centre),
            fixed_indices=frozenset(fixed),
            residual_tol_n=_residual_tolerance(stiffness), max_iterations=60,
        )
        return layout, registry, context, result, weight, coupling

    @pytest.mark.parametrize("angle_deg", (5.0, 10.0, 20.0, 30.0, 40.0))
    def test_the_reaction_moment_matches_the_closed_form(self, angle_deg) -> None:
        """``M_reaction = W·R·sinα``——闭式，五个角度逐个对。

        **这是"接触力对质心取矩"最直接的一条门**：法向过球心不给力矩，
        全部力矩来自切向接触力乘半径。今天的模型（无转动块）算不出这个数，
        它连"需要一个力矩"这件事都表达不出来。
        """

        stiffness = 2.0e6
        layout, registry, context, result, weight, _ = self._solve(angle_deg, stiffness)
        assert result.converged, result.reason
        _, gradient, _ = registry.total(result.state, context, need_gradient=True)
        reaction = -gradient[layout.rotation_base(0) + 1]
        closed_form = weight * RADIUS_MM * math.sin(math.radians(angle_deg))
        assert reaction == pytest.approx(closed_form, rel=1.0e-9)

    @pytest.mark.parametrize("angle_deg", (10.0, 30.0))
    def test_the_friction_demand_is_still_the_tangent(self, angle_deg) -> None:
        """``T/N = tanα``**不因转动而变**——回归判据：新能力不许改旧答案。"""

        layout, registry, context, result, _, coupling = self._solve(angle_deg, 2.0e6)
        tangential = _magnitude(coupling.tangential_force_n(result.state)[0])
        normal = registry.terms[1].normal_force_n(result.state)[0]
        assert tangential / normal == pytest.approx(
            math.tan(math.radians(angle_deg)), rel=1e-9
        )

    def test_a_free_sphere_on_an_incline_has_no_static_equilibrium(self) -> None:
        """**放开自旋就没有静平衡**：力矩平衡要``f = 0``，力平衡要``f = W·sinα``。

        这是转动自由度带进来的**新物理**，而不是求解失败。今天的模型
        （质点+半径）在同一构型上安静地给出一个平衡解——**它连摇晃这个
        自由度都没有**（0063第一节）。
        """

        *_, result, _, _ = self._solve(20.0, 2.0e6, release_spin=True)
        assert not result.converged, (
            "球在斜面上解出了静平衡——那意味着力矩平衡没有被真的施加"
        )



def test_the_critical_friction_is_two_minus_root_three(oracles) -> None:
    """**本案例的头条判据。**"""

    entry = oracles["oracle:pyramid_rot/critical_friction"]
    stiffness = entry["inputs"]["stiffness_n_per_mm"]
    solved = solve_pyramid(stiffness)
    assert solved["result"].converged
    forces = _forces(solved)
    sphere_demand = forces["sphere_tangential"] / forces["sphere_normal"]
    ground_demand = forces["ground_tangential"] / forces["ground_normal"]
    critical = max(sphere_demand, ground_demand)
    assert critical == pytest.approx(
        entry["expected"]["critical_friction"],
        rel=entry["tolerances"]["critical_friction"]["rel"],
    )
    #: **零容差**：卡住的必须是球-球，不是地面。
    assert sphere_demand > ground_demand
    assert ground_demand == pytest.approx((2.0 - SQRT3) / 3.0, rel=1e-6)



def test_the_top_sphere_moment_is_exactly_redundant(oracles) -> None:
    """**逐位零容差。** 钉住顶球自旋处的约束反力矩必须恰为``0.0``。

    它把research/15第3.2节"第9条方程冗余"从一句论断变成一条可判的门，
    同时证明"钉住顶球自旋"这个边界条件**没有改问题**。
    """

    entry = oracles["oracle:pyramid_rot/top_sphere_moment_is_redundant"]
    solved = solve_pyramid(entry["inputs"]["stiffness_n_per_mm"])
    _, gradient, _ = solved["registry"].total(
        solved["result"].state, solved["context"], need_gradient=True
    )
    reaction = gradient[solved["layout"].rotation_base(0) + 1]
    assert reaction.hex() == (0.0).hex(), (
        f"顶球约束反力矩不是逐位零而是{reaction!r}——"
        "钉住那一步改了问题，或者对称性没有被解出来"
    )


# ---------------------------------------------------------------------------
# 两侧：撑住与塌（决策0080，C档第7条的兑现）
# ---------------------------------------------------------------------------
#
# 前面那一半全部跑在**纯粘着**支上：四根弹簧从不滑，锚点从不动。
# 这一半把`contact.stepper`接进来，于是锚点会动——**而锚点是历史**（0033）。
#
# ## “塌”在准静态框架里到底是什么
#
# 不是“求解器炸了”。准静态问题问的是**静解存不存在**，而“不存在”是可观测的：
#
# * ``μ ≥ μc``：初始位形就是不动点。走多少步，**整条状态向量逐位不变**、
#   四个滑距逐位是``0x0.0p+0``。这不是“变化很小”，是**什么都没发生**；
# * ``μ < μc``：**每一步都滑掉同样多**，而且滑距有闭式``(μc − μ)·F/k_t``。
#   累计滑移线性发散、两底球逐步分开、顶球逐步下降——
#   **没有不动点，所以没有静解**。
#
# 这两侧之间那条线由二分独立走出来：只问“走一步之后锚点动没动”（一个布尔量），
# 二分到``1e-12``。它落在``2 − √3``上，相对偏差1.766e-07——
# **与全粘着解那条路给的是同一个数**，而那两条路不共用任何一行判据代码。

from physics_engine.contact.errors import ContactError  # noqa: E402
from physics_engine.contact.layout import (  # noqa: E402
    REGIME_SLIP,
    REGIME_STICK,
    SLOT_WIDTH,
    ContactDeclaration,
    build_contact_layout,
)
from physics_engine.contact.stepper import (  # noqa: E402
    ContactPoint,
    advance_contact_quasistatic,
    advance_contacts_quasistatic,
)

NORMAL_RIGHT = (-0.5, 0.0, SQRT3 / 2.0)
NORMAL_LEFT = (0.5, 0.0, SQRT3 / 2.0)
UP = (0.0, 0.0, 1.0)
#: 槽的声明次序**即形制**：它定了每个锚点落在向量的哪一格。
CONTACT_PAIRS = ("sphere_right", "sphere_left", "ground_right", "ground_left")
SPHERE_SLOTS = (0, 1)
GROUND_SLOTS = (2, 3)
ZERO_HEX = (0.0).hex()

#: —— 既有产物的金标（`test_the_legacy_stepper_products_are_frozen`）——
#: 在``d082b65``的基线树上取。**改动这几个数就是改动既有产物**，
#: 而那要一份决策记录，不是一次"顺手更新预期值"。
LEGACY_FROZEN_VECTOR = [
    "0x1.f75104d551d69p-15",
    "0x0.0p+0",
    "-0x1.34988e00df231p-13",
    "0x1.0000000000000p+0",
    "0x1.1d20f81d3295fp-17",
    "0x0.0p+0",
    "0x0.0p+0",
    "0x1.0000000000000p+1",
]
LEGACY_FROZEN_NORMAL = "0x1.d6e147ae147aep+3"
LEGACY_FROZEN_TANGENTIAL = ["0x1.499db22d0e560p+2", "0x0.0p+0", "0x0.0p+0"]
LEGACY_FROZEN_SLIP = "0x1.1d20f81d3295fp-17"



def _pyramid_stepper_setup(stiffness: float):
    """金字塔的**接触布局**版：节点块 + 四个锚点槽 + 三个转动块。

    与`solve_pyramid`的差别只有一条，但它是这一整节存在的理由：
    那边的粘着弹簧是**调用方装配的能量项**（锚点是常量），
    这边的锚点**住在状态里**，由`advance_contacts_quasistatic`改写。

    杠杆臂取参考构型下的``∓R·n``——球面上接触点就在连心线上，
    所以这个杠杆臂同时是**物质点**与**几何接触点**。
    两者在有限转角下会分开（球转过去之后贴地的是另一个物质点），
    因此``θ``的量级本身是判据的一部分（`test_the_march_stays_in_the_local_chart`）。
    """

    layout = build_contact_layout(
        layout_id="layout/three_sphere_pyramid_rotational_two_sided",
        node_count=3,
        declarations=tuple(ContactDeclaration(name) for name in CONTACT_PAIRS),
        rotating_bodies=(0, 1, 2),
    )
    ground_penetration = 1.5 * WEIGHT_N / stiffness
    sphere_penetration = 0.5 * WEIGHT_N / stiffness
    base_z = RADIUS_MM - ground_penetration
    positions = (
        0.0, 0.0, base_z + SQRT3 * RADIUS_MM - 2.0 * sphere_penetration / SQRT3,
        RADIUS_MM, 0.0, base_z,
        -RADIUS_MM, 0.0, base_z,
    )
    vector = list(layout.initial_vector(positions))
    #: 地面那两根弹簧的零应力锚点在接触点正下方（``x = ±R``），不是原点。
    #: 球-球那两根的锚点是**相对**位移，参考构型下恰为零矢量，故留全零。
    for name, anchor_x in (("ground_right", RADIUS_MM), ("ground_left", -RADIUS_MM)):
        vector[layout.slot_of(name).anchor_base] = anchor_x

    sphere_contact = PenaltySphereContact(
        pairs=((0, 1, 2 * RADIUS_MM, stiffness), (0, 2, 2 * RADIUS_MM, stiffness))
    )
    ground_contact = PenaltyNormalContact(
        planes=(
            (1, (0.0, 0.0, 0.0), UP, stiffness, RADIUS_MM),
            (2, (0.0, 0.0, 0.0), UP, stiffness, RADIUS_MM),
        )
    )
    #: **不含粘着项**——粘着弹簧由步进器每趟按当前锚点自己造（那是它的契约）。
    registry = EnergyRegistry(terms=(UniformGravity(), sphere_contact, ground_contact))
    context = EnergyContext(
        context_id="context/three_sphere_pyramid_rotational_two_sided",
        node_masses_kg=(MASS_KG,) * 3,
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    rotation = tuple(layout.rotation_base(body) for body in range(3))

    fixed = {3 * node + 1 for node in range(3)}
    #: 锚点槽**不是自由度**：它们是历史，交给`fixed_indices`。
    #: 不钉住的话牛顿会把历史当未知数解——那正是`_check_material_point`
    #: 挡的同一类错，只不过这里是调用方这一侧的义务。
    slot_start = layout.layout.node_dof_count
    fixed |= set(range(slot_start, slot_start + 5 * len(layout.slots)))
    for base in rotation:
        fixed |= {base, base + 2}
    #: 规范固定：顶球自旋。理由与代价见`solve_pyramid`的docstring。
    fixed.add(rotation[0] + 1)
    return {
        "layout": layout,
        "registry": registry,
        "context": context,
        "sphere_contact": sphere_contact,
        "ground_contact": ground_contact,
        "vector": tuple(vector),
        "rotation": rotation,
        "fixed": frozenset(fixed),
        "stiffness": stiffness,
    }


def _pyramid_contacts(setup, friction: float):
    """四个接触点。两个球-球**两端都是可转的物质点**，两个球-地对世界锚点。

    这是`ContactPoint.node`吃`MaterialPoint`那条扩展的第一个真实调用方：
    球-球接触的两端都在动、都在转，`contact.friction.TangentialStickSpring`
    做不了它（那个项只接“节点对固定锚点”）。
    """

    layout = setup["layout"]
    sphere_contact = setup["sphere_contact"]
    ground_contact = setup["ground_contact"]
    rotation = setup["rotation"]
    stiffness = setup["stiffness"]
    return (
        ContactPoint(
            slot=layout.slot_of("sphere_right"),
            node=MaterialPoint(
                0, tuple(-RADIUS_MM * v for v in NORMAL_RIGHT), rotation[0]
            ),
            counterpart=MaterialPoint(
                1, tuple(RADIUS_MM * v for v in NORMAL_RIGHT), rotation[1]
            ),
            normal=NORMAL_RIGHT,
            normal_force_of=lambda state: sphere_contact.contact_force_n(state)[0],
            tangential_stiffness_n_per_mm=stiffness,
            friction_coefficient=friction,
        ),
        ContactPoint(
            slot=layout.slot_of("sphere_left"),
            node=MaterialPoint(
                0, tuple(-RADIUS_MM * v for v in NORMAL_LEFT), rotation[0]
            ),
            counterpart=MaterialPoint(
                2, tuple(RADIUS_MM * v for v in NORMAL_LEFT), rotation[2]
            ),
            normal=NORMAL_LEFT,
            normal_force_of=lambda state: sphere_contact.contact_force_n(state)[1],
            tangential_stiffness_n_per_mm=stiffness,
            friction_coefficient=friction,
        ),
        ContactPoint(
            slot=layout.slot_of("ground_right"),
            node=MaterialPoint(1, (0.0, 0.0, -RADIUS_MM), rotation[1]),
            normal=UP,
            normal_force_of=lambda state: ground_contact.normal_force_n(state)[0],
            tangential_stiffness_n_per_mm=stiffness,
            friction_coefficient=friction,
        ),
        ContactPoint(
            slot=layout.slot_of("ground_left"),
            node=MaterialPoint(2, (0.0, 0.0, -RADIUS_MM), rotation[2]),
            normal=UP,
            normal_force_of=lambda state: ground_contact.normal_force_n(state)[1],
            tangential_stiffness_n_per_mm=stiffness,
            friction_coefficient=friction,
        ),
    )


def march_pyramid(friction: float, *, steps: int, stiffness: float = 2.0e6):
    """走``steps``步准静态接触。**每一步都必须收敛**，不收敛直接抛。

    抛出去而不是记一个标志，是因为本节的判据里**“不收敛”永远不是一个答案**：
    它既不算撑住也不算塌（见`test_the_collapse_never_leans_on_non_convergence`）。
    """

    setup = _pyramid_stepper_setup(stiffness)
    contacts = _pyramid_contacts(setup, friction)
    current = setup["vector"]
    history = []
    for _ in range(steps):
        step = advance_contacts_quasistatic(
            registry_without_stick=setup["registry"],
            context=setup["context"],
            contact_layout=setup["layout"],
            contacts=contacts,
            vector=current,
            fixed_indices=setup["fixed"],
            residual_tol_n=_residual_tolerance(stiffness),
            max_iterations=60,
        )
        current = step.state.vector
        history.append(
            {
                "vector": current,
                "slips": step.slip_increment_mm,
                "regimes": step.regime,
                "normal": step.normal_force_n,
                "tangential": step.tangential_force_n,
                "base_gap": current[3] - current[6],
                "top_z": current[2],
                "spins": tuple(current[base + 1] for base in setup["rotation"]),
            }
        )
    return setup, history


def _demand_ratio(record) -> float:
    return max(
        _magnitude(record["tangential"][index]) / record["normal"][index]
        for index in range(4)
    )


# --------------------------------------------------------------------------
# ``μ > μc``：**初始位形就是不动点**。
# --------------------------------------------------------------------------


def test_the_pyramid_holds_and_nothing_happens(oracles) -> None:
    """撑住的含义是**这一步什么都没发生**——四条各判一次，全部逐位。

    只判“滑距为零”不够：滑距为零而位形在漂说明有别的东西在写状态。
    只判“位形不变”也不够：一个把滑距算错但把状态写回原值的实现照样绿。
    """

    entry = oracles["oracle:pyramid_rot/two_sided_hold_and_collapse"]
    friction = entry["inputs"]["hold_friction"]
    steps = entry["inputs"]["hold_steps"]
    setup, history = march_pyramid(
        friction, steps=steps, stiffness=entry["inputs"]["stiffness_n_per_mm"]
    )

    #: (1) 滑距逐位零。**是`float.hex()`不是`== 0.0`**——后者收``-0.0``。
    slips = {float(value).hex() for record in history for value in record["slips"]}
    assert slips == {ZERO_HEX}, f"撑住那一边出现了非零滑距：{sorted(slips)}"

    #: (2) 判别全是粘。它与滑距为零是两段代码写出来的两件事。
    regimes = {value for record in history for value in record["regimes"]}
    assert regimes == {REGIME_STICK}, f"撑住那一边出现了非粘判别：{regimes}"

    #: (3) **整条状态向量逐位不变**——含锚点槽、含转动块。
    vectors = {tuple(v.hex() for v in record["vector"]) for record in history}
    assert len(vectors) == 1, (
        f"{steps}步走出了{len(vectors)}个不同的状态——撑住那一边必须是不动点"
    )

    #: (4) 需求比对上``μc``且不超过``μ``。**判力而不只判“没滑”**：
    #: 没滑可能是因为力算错了小了，而需求比对上``2−√3``说明力与锥都是对的。
    ratio = _demand_ratio(history[0])
    assert ratio == pytest.approx(
        entry["expected"]["hold_demand_ratio"],
        rel=entry["tolerances"]["hold_demand_ratio"]["rel"],
    )
    assert ratio <= friction, (
        f"需求比{ratio!r}超过了μ={friction!r}却还报粘——摩擦锥没判"
    )


def test_a_friction_below_the_threshold_does_not_hold(oracles) -> None:
    """**必红专防**：同一段代码在``μ < μc``时必须**不是**不动点。

    没有这一条，一个“锚点永远不动”的实现把上面四条全判绿——
    而那正是2026-08-18之前的状态（滑移根本没接转动块）。
    """

    entry = oracles["oracle:pyramid_rot/two_sided_hold_and_collapse"]
    _, history = march_pyramid(entry["inputs"]["collapse_friction"], steps=3)
    slips = {float(value).hex() for record in history for value in record["slips"]}
    assert slips != {ZERO_HEX}, (
        "μ低于μc时锚点仍然一个都没动——**滑移没有接上**，"
        "这条案例的两侧判据于是只剩一侧"
    )
    vectors = {tuple(v.hex() for v in record["vector"]) for record in history}
    assert len(vectors) == 3, "μ低于μc时状态却是不动点——静解不该存在"


# --------------------------------------------------------------------------
# ``μ < μc``：**没有不动点，所以没有静解**。
# --------------------------------------------------------------------------


def test_the_pyramid_collapses_with_a_closed_form_rate(oracles) -> None:
    """塌那一边的六条判据。**每一条都是可观测量，没有一条来自异常。**"""

    entry = oracles["oracle:pyramid_rot/two_sided_hold_and_collapse"]
    friction = entry["inputs"]["collapse_friction"]
    steps = entry["inputs"]["collapse_steps"]
    expected = entry["expected"]
    tolerances = entry["tolerances"]
    _, history = march_pyramid(
        friction, steps=steps, stiffness=entry["inputs"]["stiffness_n_per_mm"]
    )

    #: (1) 球-球在滑、(2) 地面在粘。**“塌”不是“到处都在滑”**：
    #: 地面的需求是``(2−√3)/3``，宽松3倍，``μ = 0.20``按不动它。
    for record in history:
        for index in SPHERE_SLOTS:
            assert record["regimes"][index] == expected["collapse_sphere_regime"]
        for index in GROUND_SLOTS:
            assert record["regimes"][index] == expected["collapse_ground_regime"]

    #: (3) 地面滑距逐位零——判别与滑距由两段不同的代码写出来，都要对。
    ground = {
        float(record["slips"][index]).hex()
        for record in history
        for index in GROUND_SLOTS
    }
    assert ground == {ZERO_HEX}, f"地面滑了：{sorted(ground)}"

    #: (4) 方向：两底球**逐步严格分开**、顶球**逐步严格下降**。
    #: 这两条把“塌”钉成两个独立方向上的单调。
    #: **朝反方向单调也满足“锚点动了”**——那正是把锚点修正符号取反的实现
    #: 会给出的样子（见`_assemble_stick`那段符号注释）。
    gaps = [record["base_gap"] for record in history]
    tops = [record["top_z"] for record in history]
    assert all(gaps[i] < gaps[i + 1] for i in range(len(gaps) - 1)), (
        "两底球没有逐步分开——滑移的方向不是散架那一侧"
    )
    assert all(tops[i] > tops[i + 1] for i in range(len(tops) - 1)), (
        "顶球没有逐步下降——重力没有在这条不可逆路径上做正功"
    )

    #: (5) **滑距不衰减**：这一条才是“塌”与“松一下就稳住”的分界。
    first = sum(history[0]["slips"])
    last = sum(history[-1]["slips"])
    assert last >= first, (
        f"滑距在衰减（{first!r} → {last!r}）——那是趋于一个静解，不是塌"
    )

    #: (6) 速率律：单接触滑距 = ``(μc − μ)·F/k_t``。
    measured = history[0]["slips"][SPHERE_SLOTS[0]]
    assert measured == pytest.approx(
        expected["collapse_slip_per_contact_mm"],
        rel=tolerances["collapse_slip_per_contact_mm"]["rel"],
    )


def test_the_collapse_never_leans_on_non_convergence(oracles) -> None:
    """**本节最要害的一条：不许拿“求解器不收敛”冒充“塌了”。**

    `march_pyramid`在任何一步不收敛时直接抛`ContactError`，
    所以上面那条判据能跑完本身就说明每一步都收敛了。这里把它**写成断言**，
    因为“它没抛”是一件太容易在重构里悄悄消失的事。

    近阈值确有一条不收敛带（case.md第四节第7条），本组的``μ``刻意远离它——
    而“远离”这件事也必须是被判的，不是被相信的。
    """

    entry = oracles["oracle:pyramid_rot/two_sided_hold_and_collapse"]
    friction = entry["inputs"]["collapse_friction"]
    steps = entry["inputs"]["collapse_steps"]
    #: 不收敛会抛，抛了就是红。**不写`pytest.raises`的反面**——
    #: 那会把“没抛”变成一句没人验的旁白。
    _, history = march_pyramid(friction, steps=steps)
    assert len(history) == steps, "行进没走满，说明中途被截断了"
    assert entry["expected"]["collapse_every_step_converges"] is True

    #: 必红那一半：证明这条门**分得开**——近阈值那一带确实会抛。
    #: 没有它，"每一步都收敛"在一个永远不抛的实现上也是绿的。
    with pytest.raises(ContactError, match="did not converge"):
        march_pyramid(0.26794, steps=20)


def test_the_march_stays_in_the_local_chart(oracles) -> None:
    """``θ``必须一直很小——**否则杠杆臂就该重输运，而本组没有重输运**。

    `rotation.retransport_levers`存在正是为了大转角（0079第六节）。
    本组的杠杆臂是参考构型下的``∓R·n``且全程不动，
    **这个选择只有在``θ``小到杠杆臂几乎没转过时才成立**。
    所以这不是一条锦上添花的观测，是本组建模前提的判据。

    实测60步末``max|θ| = 5.79e-07 rad``——比指数映射的级数/闭式切换阈值
    ``0.2``小五个数量级，比``π``小七个。**因此本路径给不了`retransport_levers`
    第一个真实调用方**，这一条如实登记在case.md第四节第8条。
    """

    entry = oracles["oracle:pyramid_rot/two_sided_hold_and_collapse"]
    _, history = march_pyramid(
        entry["inputs"]["collapse_friction"], steps=entry["inputs"]["collapse_steps"]
    )
    largest = max(abs(spin) for record in history for spin in record["spins"])
    assert largest < 1.0e-5, (
        f"末步转角{largest!r}已经不小了——杠杆臂必须重输运，"
        "本组的“杠杆臂全程不动”前提不再成立"
    )
    #: 必红那一半：转动**确实动了**。一个把转动块钉死的实现会给出逐位零，
    #: 而那会让上面那条上界断言变得毫无内容。
    assert largest > 0.0, "转动块全程为零——转动根本没有参与这条不可逆路径"


# --------------------------------------------------------------------------
# 两侧之间那条线：**只问锚点动没动**，二分出来。
# --------------------------------------------------------------------------

def _slips_on_the_first_step(friction: float, stiffness: float) -> bool:
    _, history = march_pyramid(friction, steps=1, stiffness=stiffness)
    return any(history[0]["slips"][index] > 0.0 for index in SPHERE_SLOTS)


def test_the_slip_onset_brackets_two_minus_root_three(oracles) -> None:
    """对``μ``二分“走一步之后锚点动没动”，夹到``1e-12``。

    **这条路不读任何力**：判据是一个布尔量（锚点是历史，动没动没有中间态），
    所以它与`test_the_critical_friction_is_two_minus_root_three`那条
    **不共用任何一行判据代码**，却必须落在同一个数上。
    """

    entry = oracles["oracle:pyramid_rot/slip_onset_threshold"]
    stiffness = entry["inputs"]["stiffness_n_per_mm"]
    low = entry["inputs"]["bracket_low"]
    high = entry["inputs"]["bracket_high"]
    width = entry["inputs"]["bracket_width"]

    #: 前置断言：**两个端点必须分处两侧**，否则二分夹的是空气。
    assert _slips_on_the_first_step(low, stiffness), "下端点没滑，区间取错了"
    assert not _slips_on_the_first_step(high, stiffness), "上端点滑了"

    while high - low > width:
        middle = 0.5 * (low + high)
        if _slips_on_the_first_step(middle, stiffness):
            low = middle
        else:
            high = middle
    threshold = 0.5 * (low + high)

    assert high - low <= width, "夹取精度没达到声明值"
    assert threshold == pytest.approx(
        entry["expected"]["critical_friction"],
        rel=entry["tolerances"]["critical_friction"]["rel"],
    )

    #: **两条路必须落在同一个夹取区间里**，而不只是各自对上``2−√3``：
    #: 5e-7的容差留得下一个1e-7量级的**共同**偏差，
    #: 那样两条路可以一起错而两条门一起绿。
    _, hold = march_pyramid(0.30, steps=1, stiffness=stiffness)
    stick_demand = _demand_ratio(hold[0])
    assert abs(stick_demand - threshold) <= 0.5 * width, (
        f"滑移起始阈值{threshold!r}与全粘着解的需求比{stick_demand!r}对不上——"
        "两条路走到了两个不同的数"
    )


def test_the_collapse_rate_extrapolates_to_the_threshold(oracles) -> None:
    """速率律三档。**沿``μc − μ``线性对上才说明速率律本身是对的。**

    单档对上只说明那一个数凑巧；三档跨一个数量级都对上，
    说明``滑距/步 = (μc − μ)·F/k_t``这条闭式成立，
    于是“撑住/塌”不是一个一刀切的开关，而是一条连续的、外推到``μc``归零的曲线。
    """

    entry = oracles["oracle:pyramid_rot/collapse_rate_law"]
    stiffness = entry["inputs"]["stiffness_n_per_mm"]
    for friction in entry["inputs"]["frictions"]:
        key = f"slip_per_contact_mm_at_mu_{friction:.2f}".replace(".", "p")
        _, history = march_pyramid(friction, steps=1, stiffness=stiffness)
        measured = history[0]["slips"][SPHERE_SLOTS[0]]
        assert measured == pytest.approx(
            entry["expected"][key], rel=entry["tolerances"][key]["rel"]
        ), f"μ={friction}那一档的滑移速率对不上闭式"


class TestTheMaterialPointGeneralisationIsFaithful:
    """接口推广的忠实性：**零杠杆臂、不转的物质点必须与节点号逐位相同**。"""

    def test_a_material_point_with_no_lever_matches_the_node_index(self) -> None:
        """``MaterialPoint(n, (0,0,0), None)``与``n``描述的是同一个东西。

        这条门守的是`_assemble_stick`那张表：``int``走
        `contact.friction.TangentialStickSpring`，物质点走
        `rotation.MaterialPointStickSpring`，**两条是两串代码**。
        它们在这个退化点上必须逐位一致——否则“推广”就是“换了一个物理”。

        **``-0.0``是这条门唯一的边角**：``x + 0.0``对普通浮点是恒等，
        对``-0.0``不是（它变成``+0.0``）。杠杆臂带零分量是常态
        （地面弹簧的杠杆臂就是``(0,0,−R)``），所以这一条写在这里而不是被略过——
        0079注错验证抓到的空门一号就是同一个形状。
        """

        stiffness = 1.0e5
        weight = MASS_KG * GRAVITY_MM_S2 / 1000.0
        layout = build_contact_layout(
            layout_id="layout/material-point-degeneracy",
            node_count=1,
            declarations=(ContactDeclaration("ground"),),
        )
        context = EnergyContext(
            context_id="context/material-point-degeneracy",
            node_masses_kg=(MASS_KG,),
            gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
        )
        ground = PenaltyNormalContact(
            planes=((0, (0.0, 0.0, 0.0), UP, stiffness, 0.0),)
        )
        registry = EnergyRegistry(
            terms=(UniformGravity(), ground, PointLoad(loads=((0, (6.0, 0.0, 0.0)),)))
        )
        slot = layout.slot_of("ground")
        vector = layout.initial_vector((0.0, 0.0, -weight / stiffness))
        fixed = frozenset({1} | set(range(3, layout.layout.dof_count)))

        def advance(end):
            return advance_contact_quasistatic(
                registry_without_stick=registry,
                context=context,
                contact_layout=layout,
                slot=slot,
                vector=vector,
                node=end,
                normal=UP,
                normal_force_of=lambda state: ground.normal_force_n(state)[0],
                tangential_stiffness_n_per_mm=stiffness,
                friction_coefficient=0.35,
                fixed_indices=fixed,
                residual_tol_n=1.0e-9,
                max_iterations=60,
            )

        by_index = advance(0)
        by_point = advance(MaterialPoint(0, (0.0, 0.0, 0.0), None))
        assert [v.hex() for v in by_point.state.vector] == [
            v.hex() for v in by_index.state.vector
        ], "零杠杆臂的物质点与节点号给出了不同的状态——推广不忠实"
        assert by_point.slip_increment_mm.hex() == by_index.slip_increment_mm.hex()
        assert by_point.normal_force_n.hex() == by_index.normal_force_n.hex()
        assert [v.hex() for v in by_point.tangential_force_n] == [
            v.hex() for v in by_index.tangential_force_n
        ], (
            "切向力逐位不同——最可能的来路是试探力的符号："
            "两个类的`tangential_force_n`差一个负号（见`_assemble_stick`）"
        )
        assert by_point.regime == by_index.regime

    def test_a_rotation_block_overlapping_an_anchor_slot_fails_closed(self) -> None:
        """**必红**：转动块压在锚点槽上必须当场拒收。

        它挡的是本次扩展带进来的**新**危险：转动块是牛顿要解的自由度，
        锚点是历史。两者重叠时牛顿会把历史当未知数解，
        **而这不会抛任何异常**——它只会安静地给出一个“收敛了”的错答案。
        """

        layout = build_contact_layout(
            layout_id="layout/rotation-overlaps-slot",
            node_count=1,
            declarations=(ContactDeclaration("ground"),),
            rotating_bodies=(0,),
        )
        slot = layout.slot_of("ground")
        vector = layout.initial_vector((0.0, 0.0, 0.0))
        context = EnergyContext(
            context_id="context/rotation-overlaps-slot",
            node_masses_kg=(MASS_KG,),
            gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
        )
        ground = PenaltyNormalContact(
            planes=((0, (0.0, 0.0, 0.0), UP, 1.0e5, 0.0),)
        )
        registry = EnergyRegistry(terms=(UniformGravity(), ground))

        def advance(rotation_base):
            return advance_contact_quasistatic(
                registry_without_stick=registry,
                context=context,
                contact_layout=layout,
                slot=slot,
                vector=vector,
                node=MaterialPoint(0, (0.0, 0.0, -1.0), rotation_base),
                normal=UP,
                normal_force_of=lambda state: ground.normal_force_n(state)[0],
                tangential_stiffness_n_per_mm=1.0e5,
                friction_coefficient=0.35,
                fixed_indices=frozenset({1}),
            )

        #: 锚点槽在``[3, 8)``；把转动块指到``4``就压在锚点上。
        with pytest.raises(ContactError, match="重叠"):
            advance(slot.anchor_base)
        #: 指进节点块同样拒收。
        with pytest.raises(ContactError, match="落在节点块之内"):
            advance(0)
        #: 越过向量末尾也拒收。
        with pytest.raises(ContactError, match="越过了状态向量末尾"):
            advance(len(vector) - 1)

        #: **物质点的节点号同样要在节点块内**——注错验证第二轮的空门。
        #: `int`那一支有这条检查而物质点那一支的没人验过，
        #: 而写出节点块就是写进锚点槽：**改的是别人的历史**。
        with pytest.raises(ContactError, match="落在节点块之外"):
            advance_contact_quasistatic(
                registry_without_stick=registry,
                context=context,
                contact_layout=layout,
                slot=slot,
                vector=vector,
                node=MaterialPoint(7, (0.0, 0.0, -1.0), layout.rotation_base(0)),
                normal=UP,
                normal_force_of=lambda state: ground.normal_force_n(state)[0],
                tangential_stiffness_n_per_mm=1.0e5,
                friction_coefficient=0.35,
                fixed_indices=frozenset({1}),
            )
        #: **正的那一支必须能过**：正确的转动基址由布局给出。
        #: 没有这一条，一个"什么都拒收"的实现把上面三条全判绿。
        assert layout.rotation_base(0) == len(vector) - 3

    def test_a_counterpart_without_a_material_point_fails_closed(self) -> None:
        """**必红**：给了对边却把这一端写成节点号——那是两种形制混着用。"""

        layout = build_contact_layout(
            layout_id="layout/counterpart-misuse",
            node_count=2,
            declarations=(ContactDeclaration("pair"),),
        )
        context = EnergyContext(
            context_id="context/counterpart-misuse",
            node_masses_kg=(MASS_KG,) * 2,
            gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
        )
        registry = EnergyRegistry(terms=(UniformGravity(),))
        with pytest.raises(ContactError, match="counterpart"):
            advance_contact_quasistatic(
                registry_without_stick=registry,
                context=context,
                contact_layout=layout,
                slot=layout.slot_of("pair"),
                vector=layout.initial_vector((0.0,) * 6),
                node=0,
                counterpart=MaterialPoint(1, (0.0, 0.0, 0.0), None),
                normal=UP,
                normal_force_of=lambda state: 1.0,
                tangential_stiffness_n_per_mm=1.0e5,
                friction_coefficient=0.35,
                fixed_indices=frozenset(),
            )


class TestTheRotationBlockInTheContactLayout:
    """转动块进接触布局的四条校验，外加"空元组时逐字节不变"那一条。

    这一整类是注错验证第一轮抓到的**空门二**：三条校验从
    `rotation.build_rigid_body_layout`抄过来时**一条用例都没跟过来**，
    于是删掉它们全部照样绿。
    """

    def test_an_empty_declaration_leaves_the_layout_byte_identical(self) -> None:
        """``rotating_bodies=()``时字段清单、向量长度、**指纹**全都不变。

        这是三前提第三条在布局上的形式：**能加转动块**与
        **加了转动块的默认路径**是两件事，而既有产物走的是后者。
        指纹判到位是因为它是`run_package`那一层的承重量（0019）。
        """

        declarations = (ContactDeclaration("a"), ContactDeclaration("b"))
        plain = build_contact_layout(
            layout_id="layout/rotation-default",
            node_count=3,
            declarations=declarations,
        )
        explicit = build_contact_layout(
            layout_id="layout/rotation-default",
            node_count=3,
            declarations=declarations,
            rotating_bodies=(),
        )
        assert plain.layout.fingerprint() == explicit.layout.fingerprint()
        assert plain.rotating_bodies == ()
        assert plain.rotation_dof_count == 0
        assert [field.name for field in plain.layout.fields] == [
            field.name for field in explicit.layout.fields
        ]
        positions = tuple(float(index) for index in range(9))
        assert [v.hex() for v in plain.initial_vector(positions)] == [
            v.hex() for v in explicit.initial_vector(positions)
        ]

    def test_the_rotation_block_sits_after_every_anchor_slot(self) -> None:
        """转动块排在**所有**锚点槽之后——槽的``base``一个都不许被顶掉。

        插在节点块与槽之间会让每个既有槽后移，**既有产物的锚点会读到别人的历史**。
        """

        layout = build_contact_layout(
            layout_id="layout/rotation-after-slots",
            node_count=3,
            declarations=(ContactDeclaration("a"), ContactDeclaration("b")),
            rotating_bodies=(2, 0),
        )
        last_slot_end = max(slot.base + SLOT_WIDTH for slot in layout.slots)
        assert layout.rotation_base(2) == last_slot_end
        #: 次序即声明次序：``(2, 0)``意味着体2排前面。
        assert layout.rotation_base(0) == last_slot_end + 3
        assert layout.rotation_dof_count == 6
        assert layout.rotation_indices() == frozenset(
            range(last_slot_end, last_slot_end + 6)
        )
        assert layout.layout.dof_count == last_slot_end + 6
        #: 加了转动块之后，**槽的落位一个字都没变**。
        plain = build_contact_layout(
            layout_id="layout/rotation-after-slots",
            node_count=3,
            declarations=(ContactDeclaration("a"), ContactDeclaration("b")),
        )
        assert [slot.base for slot in layout.slots] == [
            slot.base for slot in plain.slots
        ]

    def test_the_rotation_fields_are_named_and_ordered(self) -> None:
        """转动块的字段名与次序**都要判**——注错验证第二轮的空门。

        把三个轴写成``("y", "x", "z")``只改名字不改向量长度，
        于是所有数值门照样绿，**而按名字读状态的人拿到的是另一个分量**。
        单位后缀``rad``同时判到：它是`StateField`那道量纲门认的后缀
        （0079第二节：转动矢量是有量纲的量，不走`is_dimensionless`那个口子）。
        """

        layout = build_contact_layout(
            layout_id="layout/rotation-field-names",
            node_count=2,
            declarations=(ContactDeclaration("a"),),
            rotating_bodies=(1, 0),
        )
        names = [field.name for field in layout.layout.fields]
        assert names[-6:] == [
            "body1_theta_x_rad",
            "body1_theta_y_rad",
            "body1_theta_z_rad",
            "body0_theta_x_rad",
            "body0_theta_y_rad",
            "body0_theta_z_rad",
        ]
        #: 转动块**不是历史**：牛顿要解它们，标成历史会让守恒断言把它们排除掉。
        for field in layout.layout.fields[-6:]:
            assert field.width == 1
            assert not field.is_history

    def test_a_body_declared_twice_fails_closed(self) -> None:
        """重复声明意味着同一个体有两个转动块，而哪一个说了算只能靠读实现。"""

        with pytest.raises(ContactError, match="declared twice"):
            build_contact_layout(
                layout_id="layout/rotation-duplicate",
                node_count=3,
                declarations=(ContactDeclaration("a"),),
                rotating_bodies=(1, 1),
            )

    def test_a_body_outside_the_node_block_fails_closed(self) -> None:
        """转动块必须挂在某个**存在的**节点上。"""

        with pytest.raises(ContactError, match="outside the node block"):
            build_contact_layout(
                layout_id="layout/rotation-out-of-range",
                node_count=2,
                declarations=(ContactDeclaration("a"),),
                rotating_bodies=(2,),
            )
        with pytest.raises(ContactError, match="outside the node block"):
            build_contact_layout(
                layout_id="layout/rotation-negative",
                node_count=2,
                declarations=(ContactDeclaration("a"),),
                rotating_bodies=(-1,),
            )

    @pytest.mark.parametrize("body", [True, 1.0, "0"])
    def test_a_body_that_is_not_an_int_fails_closed(self, body) -> None:
        """``True``单列一条：``isinstance(True, int)``是真，而它显然不是节点号。"""

        with pytest.raises(ContactError, match="int node index"):
            build_contact_layout(
                layout_id="layout/rotation-not-an-int",
                node_count=2,
                declarations=(ContactDeclaration("a"),),
                rotating_bodies=(body,),
            )

    def test_asking_for_a_body_without_a_rotation_block_fails_closed(self) -> None:
        """没声明转动块的体问``rotation_base``必须报错，**不许返回一个下标**。

        返回一个下标就等于凭空造一个转动块，而那个下标会指到别人的地盘上。
        """

        layout = build_contact_layout(
            layout_id="layout/rotation-partial",
            node_count=3,
            declarations=(ContactDeclaration("a"),),
            rotating_bodies=(1,),
        )
        assert layout.rotation_base(1) >= layout.layout.node_dof_count
        with pytest.raises(ContactError, match="carries no rotation block"):
            layout.rotation_base(0)



def test_the_legacy_stepper_products_are_frozen() -> None:
    """**既有产物逐位不变**（0001三前提第三条）——金标是2026-08-18的基线树跑出来的。

    这条门与`test_a_material_point_with_no_lever_matches_the_node_index`
    解的不是同一道题：那一条判"两条路径互相一致"，
    **两条一起改坏它照样绿**；这一条判"这条路径与**加转动之前**一致"，
    金标是常量，改不动。

    金标的来路写在这里，因为一条读者复算不出来的金标不是金标：
    在``d082b65``的树上跑同一段装置，取`ContactStep`的全部浮点产物的`float.hex()`。
    装置是`_ground_drag`那一族（横载6 N、``μ = 0.35``、``k = 1e5``），
    滑移那一支——**粘着那一支是平凡的，冻它冻不住任何东西**。
    """

    stiffness = 1.0e5
    weight = MASS_KG * GRAVITY_MM_S2 / 1000.0
    layout = build_contact_layout(
        layout_id="layout/legacy-frozen",
        node_count=1,
        declarations=(ContactDeclaration("ground"),),
    )
    context = EnergyContext(
        context_id="context/legacy-frozen",
        node_masses_kg=(MASS_KG,),
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    ground = PenaltyNormalContact(planes=((0, (0.0, 0.0, 0.0), UP, stiffness, 0.0),))
    registry = EnergyRegistry(
        terms=(UniformGravity(), ground, PointLoad(loads=((0, (6.0, 0.0, 0.0)),)))
    )
    slot = layout.slot_of("ground")
    step = advance_contact_quasistatic(
        registry_without_stick=registry,
        context=context,
        contact_layout=layout,
        slot=slot,
        vector=layout.initial_vector((0.0, 0.0, -weight / stiffness)),
        node=0,
        normal=UP,
        normal_force_of=lambda state: ground.normal_force_n(state)[0],
        tangential_stiffness_n_per_mm=stiffness,
        friction_coefficient=0.35,
        fixed_indices=frozenset({1} | set(range(3, layout.layout.dof_count))),
        residual_tol_n=1.0e-9,
        max_iterations=100,
    )
    assert [value.hex() for value in step.state.vector] == LEGACY_FROZEN_VECTOR
    assert step.normal_force_n.hex() == LEGACY_FROZEN_NORMAL
    assert [v.hex() for v in step.tangential_force_n] == LEGACY_FROZEN_TANGENTIAL
    assert step.slip_increment_mm.hex() == LEGACY_FROZEN_SLIP
    assert step.regime == REGIME_SLIP
