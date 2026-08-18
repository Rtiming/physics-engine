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
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
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
