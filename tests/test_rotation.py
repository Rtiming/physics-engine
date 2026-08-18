"""转动自由度进准静态接触（决策0079）。

**每条判据配一条必红用例**（本仓纪律）：注错验证表见0079第七节。

本文件的四组门与简报的四条判据一一对应：

1. `TestExponentialMap`——指数映射与它的一/二阶导（有限差分收敛阶）；
2. `TestBitExactDegeneracy`——**转动全钉住时逐位退化**，判`float.hex()`；
3. `TestFiniteDifference`——两个能量项的梯度对能量、Hessian对梯度；
4. 金字塔与斜面的物理判据在`tests/cases/test_three_sphere_pyramid_rotational.py`。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact.penalty import PenaltyNormalContact
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
from physics_engine.rotation import (
    AppliedMoment,
    MaterialPoint,
    MaterialPointStickSpring,
    RotationError,
    RotationStickCoupling,
    StickSpring,
    build_rigid_body_layout,
    compose,
    log_rotation,
    retransport_levers,
    rotate,
    rotate_hessian,
    rotate_jacobian,
    rotation_matrix,
)
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State, StateError

SAMPLE_THETAS = (
    (0.0, 0.0, 0.0),
    (1.0e-9, 0.0, 0.0),
    (1.0e-3, 2.0e-3, -5.0e-4),
    #: 恰好骑在级数/闭式切换阈值上（`_SERIES_THRESHOLD = 0.2`）——**两侧都要走到**。
    (0.19999, 0.0, 0.0),
    (0.20001, 0.0, 0.0),
    (0.05, -0.03, 0.09),
    (0.4, -1.1, 0.7),
    (2.0, 1.0, -1.5),
)
SAMPLE_VECTOR = (1.3, -0.7, 2.1)


def _central(function, theta, index, step):
    high = list(theta)
    low = list(theta)
    high[index] += step
    low[index] -= step
    return function(tuple(high)), function(tuple(low))


class TestExponentialMap:
    """SO(3)指数映射本身。**这是全部转动物理的地基，所以它单独成组。**"""

    @pytest.mark.parametrize("theta", SAMPLE_THETAS)
    def test_the_map_lands_in_so3(self, theta) -> None:
        """``R``必须正交且行列式为``+1``——出了SO(3)一切力矩都没有意义。"""

        matrix = rotation_matrix(theta)
        for row in range(3):
            for column in range(3):
                product = sum(matrix[m][row] * matrix[m][column] for m in range(3))
                assert product == pytest.approx(1.0 if row == column else 0.0, abs=1e-14)
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        assert determinant == pytest.approx(1.0, abs=1e-14)

    def test_zero_rotation_returns_the_input_bit_for_bit(self) -> None:
        """``θ = 0``必须**逐位**返回输入——退化门的结构性零就建在这一条上。

        这不是"数值上很接近"：`RotationStickCoupling`的节点块梯度是两个表达式的差，
        只有这一条成立时那个差才恰为``0.0``。

        ## 杠杆臂里带``−0.0``那一档是本条的**要害**（2026-08-18注错验证补）

        起草时这条门只用了一个三个分量都非零的向量，于是把恒等分支拆掉
        （强制走通用Rodrigues）**这条门照样绿**——因为``v + 1.0*0.0 + 0.5*0.0``
        对普通浮点恰好就等于``v``。**那一轮注错把它抓成了空门。**

        真正分得开两条路的是``−0.0``：通用路径把它加成``+0.0``，符号位没了。
        杠杆臂带零分量是常态（金字塔的地面弹簧杠杆臂就是``(0.0, 0.0, −R)``），
        所以这不是一个人造的边角料。
        """

        rotated = rotate((0.0, 0.0, 0.0), SAMPLE_VECTOR)
        assert [value.hex() for value in rotated] == [
            value.hex() for value in SAMPLE_VECTOR
        ]
        signed_zero = (-0.0, 1.5, -0.0)
        assert [value.hex() for value in rotate((0.0, 0.0, 0.0), signed_zero)] == [
            value.hex() for value in signed_zero
        ], "``−0.0``的符号位丢了——说明走的不是恒等分支而是通用Rodrigues"
        #: ``θ``自己带``−0.0``也必须走同一条路（`-0.0 == 0.0`为真）。
        assert [value.hex() for value in rotate((-0.0, 0.0, -0.0), signed_zero)] == [
            value.hex() for value in signed_zero
        ]

    @pytest.mark.parametrize("phi", (0.2, 0.25, 0.3))
    def test_the_series_and_the_closed_form_agree_across_the_threshold(self, phi) -> None:
        """五个标量系数在``_SERIES_THRESHOLD``两侧必须接得上。

        **这条门是注错验证补的**（2026-08-18）：把`_versine_curvature`的级数
        砍掉一项之后，有限差分那几条门**全部照样绿**——截断误差落在它们的容差以下。
        于是"级数截到第几项够用"这件事**没有任何东西看着**。

        做法是直接比同一个``φ``上的两条分支（临时把阈值挪开），
        **这比有限差分敏感好几个数量级**，因为它不经过任何差商。

        **采样点只取交叉点附近**（0.2—0.3）：再往左是闭式相消在烂、
        再往右是级数截断在烂，那两处的偏差是**已知的、被`_SERIES_THRESHOLD`
        的注释量化过的**，不是实现错误。这条门要判的是"级数项数够不够"，
        所以它必须站在两条分支都可信的那一段上。

        这条门顺带把阈值本身从0.1改到了0.2（谷底），见`_SERIES_THRESHOLD`的表。
        """

        from physics_engine import rotation as module

        helpers = (
            module._sinc,
            module._versine,
            module._sinc_slope,
            module._sinc_curvature,
            module._versine_slope,
            module._versine_curvature,
        )
        original = module._SERIES_THRESHOLD
        try:
            module._SERIES_THRESHOLD = 1.0e9  # 强制走级数
            series = [helper(phi) for helper in helpers]
            module._SERIES_THRESHOLD = 0.0  # 强制走闭式
            closed = [helper(phi) for helper in helpers]
        finally:
            module._SERIES_THRESHOLD = original

        for index, (from_series, from_closed) in enumerate(
            zip(series, closed, strict=True)
        ):
            #: 容差1e-9：交叉点附近两条分支的实测最坏偏差是1.9e-11（φ=0.3），
            #: 留约50倍余量。而级数少截一项在这一段上的偏差是**1.9e-06**——
            #: 相差三个数量级，所以这个容差既不会假红也抓得住真错。
            assert from_series == pytest.approx(from_closed, rel=1.0e-9), (
                f"第{index}个系数在φ={phi}上两条分支对不上："
                f"级数{from_series!r} vs 闭式{from_closed!r}——"
                "要么级数项数不够，要么闭式抄错了"
            )

    @pytest.mark.parametrize("theta", SAMPLE_THETAS)
    def test_the_jacobian_matches_finite_differences(self, theta) -> None:
        """``∂(Rv)/∂θ``对中心差分。"""

        analytic = rotate_jacobian(theta, SAMPLE_VECTOR)
        step = 1.0e-6
        for k in range(3):
            high, low = _central(lambda t: rotate(t, SAMPLE_VECTOR), theta, k, step)
            for axis in range(3):
                numeric = (high[axis] - low[axis]) / (2.0 * step)
                assert numeric == pytest.approx(analytic[k][axis], abs=1e-8)

    @pytest.mark.parametrize("theta", SAMPLE_THETAS)
    def test_the_second_derivative_matches_finite_differences(self, theta) -> None:
        """``∂²(Rv)/∂θ²``对中心差分，并且**必须对称**。"""

        analytic = rotate_hessian(theta, SAMPLE_VECTOR)
        step = 1.0e-6
        for j in range(3):
            high, low = _central(
                lambda t: rotate_jacobian(t, SAMPLE_VECTOR), theta, j, step
            )
            for k in range(3):
                for axis in range(3):
                    numeric = (high[k][axis] - low[k][axis]) / (2.0 * step)
                    assert numeric == pytest.approx(analytic[j][k][axis], abs=1e-7)
        for j in range(3):
            for k in range(3):
                for axis in range(3):
                    assert analytic[j][k][axis] == pytest.approx(
                        analytic[k][j][axis], abs=1e-15
                    )

    def test_the_jacobian_at_the_chart_origin_is_the_cross_product(self) -> None:
        """``θ = 0``处``∂(Rv)/∂θ_k = e_k × v``——**唯一一档能手算的**。"""

        analytic = rotate_jacobian((0.0, 0.0, 0.0), SAMPLE_VECTOR)
        expected = (
            (0.0, -SAMPLE_VECTOR[2], SAMPLE_VECTOR[1]),
            (SAMPLE_VECTOR[2], 0.0, -SAMPLE_VECTOR[0]),
            (-SAMPLE_VECTOR[1], SAMPLE_VECTOR[0], 0.0),
        )
        for k in range(3):
            for axis in range(3):
                assert analytic[k][axis] == pytest.approx(expected[k][axis], abs=1e-15)

    def test_the_derivatives_converge_at_second_order(self) -> None:
        """中心差分是二阶的：步长减10倍，误差应降约100倍。

        **这条比"误差很小"强**：它证明解析导数与被差分的那个函数是同一个函数，
        而不是"两个都错但错得差不多"。
        """

        theta = (0.4, -1.1, 0.7)
        analytic = rotate_jacobian(theta, SAMPLE_VECTOR)
        errors = []
        for step in (1.0e-2, 1.0e-3, 1.0e-4):
            worst = 0.0
            for k in range(3):
                high, low = _central(lambda t: rotate(t, SAMPLE_VECTOR), theta, k, step)
                for axis in range(3):
                    numeric = (high[axis] - low[axis]) / (2.0 * step)
                    worst = max(worst, abs(numeric - analytic[k][axis]))
            errors.append(worst)
        for first, second in zip(errors, errors[1:], strict=False):
            assert 50.0 < first / second < 200.0, f"收敛阶不是二阶：{errors}"

    @pytest.mark.parametrize("theta", SAMPLE_THETAS)
    def test_log_inverts_exp(self, theta) -> None:
        """``log(exp(θ)) = θ``（主值域内）。"""

        recovered = log_rotation(rotation_matrix(theta))
        for axis in range(3):
            assert recovered[axis] == pytest.approx(theta[axis], abs=1e-12)

    def test_log_handles_the_half_turn_branch(self) -> None:
        """``φ → π``是`log_rotation`唯一的非平凡分支，**必须被走到**。"""

        axis = (1.0 / math.sqrt(3.0),) * 3
        theta = tuple(math.pi * value for value in axis)
        recovered = log_rotation(rotation_matrix(theta))
        #: ``±π·n``是同一个旋转，故只判"转回去等于原矩阵"。
        back = rotation_matrix(recovered)
        original = rotation_matrix(theta)
        for row in range(3):
            for column in range(3):
                assert back[row][column] == pytest.approx(original[row][column], abs=1e-7)

    def test_retransport_folds_the_chart_origin_forward(self) -> None:
        """重输运：把``θ``折进杠杆臂后，新图原点上的物质点与老图上的**同一个点**。"""

        theta = (0.31, -0.22, 0.44)
        levers = ((3.0, 1.0, -2.0), (0.0, 5.0, 0.0))
        moved = retransport_levers(theta, levers)
        for lever, new_lever in zip(levers, moved, strict=True):
            for axis in range(3):
                assert new_lever[axis] == pytest.approx(rotate(theta, lever)[axis], abs=0.0)
        #: 折两次等于复合一次——局部图是同一个群的坐标，不是两套东西。
        second = (0.1, 0.05, -0.07)
        twice = retransport_levers(second, moved)
        once = retransport_levers(compose(second, theta), levers)
        for a, b in zip(twice, once, strict=True):
            for axis in range(3):
                assert a[axis] == pytest.approx(b[axis], abs=1e-12)


class TestLayout:
    """布局：转动块挂在节点块之后，``node_dof_count``不受影响。"""

    def test_the_node_block_stays_a_multiple_of_three(self) -> None:
        """`StateLayout`强制``node_dof_count % 3 == 0``——转动块不许破坏它。"""

        layout = build_rigid_body_layout(
            layout_id="layout/rot_two", node_count=2, rotating_bodies=(0, 1)
        )
        assert layout.layout.node_dof_count == 6
        assert layout.layout.dof_count == 12
        assert layout.rotation_base(0) == 6
        assert layout.rotation_base(1) == 9
        assert layout.rotation_indices() == frozenset(range(6, 12))

    def test_a_layout_without_rotation_is_byte_identical_to_the_plain_node_block(self) -> None:
        """**不声明转动体时，打包契约的指纹与纯节点布局逐位相同。**

        这是"加这条特性不动既有产物"的第一层：不用它的人拿到的是同一份字节。
        """

        from physics_engine.contact.layout import build_contact_layout

        plain = build_contact_layout(
            layout_id="layout/x", node_count=3, declarations=()
        )
        rotational = build_rigid_body_layout(
            layout_id="layout/x", node_count=3, rotating_bodies=()
        )
        assert rotational.layout.fingerprint() == plain.layout.fingerprint()

    def test_only_declared_bodies_carry_a_rotation_block(self) -> None:
        """质点与刚体可以混在同一条向量里——**不是每个节点都必须能转**。"""

        layout = build_rigid_body_layout(
            layout_id="layout/mixed", node_count=3, rotating_bodies=(1,)
        )
        assert layout.rotation_base(1) == 9
        with pytest.raises(RotationError, match="carries no rotation block"):
            layout.rotation_base(0)

    def test_a_duplicate_rotating_body_fails_closed(self) -> None:
        """**必红用例**：同一个体两个转动块，哪一个说了算只能靠读实现。"""

        with pytest.raises(RotationError, match="declared twice"):
            build_rigid_body_layout(
                layout_id="layout/dup", node_count=2, rotating_bodies=(0, 0)
            )

    def test_a_rotating_body_outside_the_node_block_fails_closed(self) -> None:
        """**必红用例**：转动块指向不存在的节点。"""

        with pytest.raises(RotationError, match="outside the node block"):
            build_rigid_body_layout(
                layout_id="layout/oob", node_count=2, rotating_bodies=(2,)
            )

    def test_the_rotation_field_names_carry_a_unit_suffix(self) -> None:
        """轴2规则3：物理量字段必须带单位后缀。``rad``在`BASE_UNIT_SUFFIXES`里。"""

        layout = build_rigid_body_layout(
            layout_id="layout/units", node_count=1, rotating_bodies=(0,)
        )
        names = [field.name for field in layout.layout.fields]
        assert names[3:] == ["body0_theta_x_rad", "body0_theta_y_rad", "body0_theta_z_rad"]
        #: 这一条由`StateField.__post_init__`守着；换成不带后缀的名字要当场红。
        with pytest.raises(StateError, match="unit suffix"):
            from physics_engine.state import StateField

            StateField("body0_theta_x", 1)


def _degeneracy_problem(*, with_coupling: bool):
    """一个真接触问题：斜面上的球，罚法向 + 物质点粘着弹簧，转动块在但被钉住。"""

    alpha = math.radians(25.0)
    radius = 10.0
    mass = 2.0
    gravity = 9810.0
    stiffness = 2.0e6
    weight = mass * gravity / 1000.0
    normal = (math.sin(alpha), 0.0, math.cos(alpha))
    layout = build_rigid_body_layout(
        layout_id="layout/degenerate", node_count=1, rotating_bodies=(0,)
    )
    centre = tuple(
        (radius - weight * math.cos(alpha) / stiffness) * value for value in normal
    )
    initial = layout.initial_vector(centre)
    spring = StickSpring(
        first=MaterialPoint(0, tuple(-radius * v for v in normal), layout.rotation_base(0)),
        normal=normal,
        stiffness_n_per_mm=stiffness,
        anchor_mm=(0.0, 0.0, 0.0),
    )
    terms = [
        UniformGravity(),
        PenaltyNormalContact(
            planes=((0, (0.0, 0.0, 0.0), normal, stiffness, radius),)
        ),
        MaterialPointStickSpring(springs=(spring,)),
    ]
    if with_coupling:
        terms.append(RotationStickCoupling(springs=(spring,)))
    registry = EnergyRegistry(terms=tuple(terms))
    context = EnergyContext(
        context_id="context/degenerate",
        node_masses_kg=(mass,),
        gravity_mm_s2=(0.0, 0.0, -gravity),
    )
    fixed = frozenset({1}) | layout.rotation_indices()
    return layout, registry, context, initial, fixed


class TestBitExactDegeneracy:
    """**三前提第三条**：转动自由度全部钉住时，既有产物逐位不变。判`float.hex()`。

    这里的"逐位"是**结构性的**不是数值上的巧合：``θ = 0``时
    `RotationStickCoupling`的能量与节点块梯度是两个**同一表达式**的差，
    而它的节点-节点Hessian块**一个条目都不出**。
    """

    def test_the_coupling_term_contributes_exactly_zero_at_the_chart_origin(self) -> None:
        """能量恰为``0.0``、节点块梯度恰为``0.0``、节点-节点Hessian条目一个都没有。

        **在已经受力的构型上判**，不在初始构型上判：初始构型的粘着弹簧还没被拉开，
        那里连力矩都是零，于是"全都是零"这句话对错两种实现都成立——
        那样的门看着绿，其实什么也没验。
        """

        layout, registry, context, initial, fixed = _degeneracy_problem(with_coupling=True)
        solved = solve_equilibrium(
            registry, context, layout.layout, initial,
            fixed_indices=fixed, residual_tol_n=1.0e-8, max_iterations=60,
        )
        assert solved.converged
        state = solved.state
        coupling = registry.terms[-1]
        assert isinstance(coupling, RotationStickCoupling)

        assert coupling.energy(state, context).hex() == (0.0).hex()
        gradient = coupling.gradient(state, context)
        for index in range(3):
            assert gradient[index] == 0.0, "节点块梯度必须恰为零"
        node_entries = [
            entry
            for entry in coupling.hessian_entries(state, context)
            if entry[0] < 3 and entry[1] < 3
        ]
        assert node_entries == [], (
            "节点-节点Hessian块必须一个条目都不出——"
            "``∂²ΔU/∂x∂x = kP − kP``在任何θ下都是零"
        )
        #: **而转动块上的梯度必须是那个力矩**，否则这个项什么也没接上。
        #: 闭式：接触力``W·sinα``挂在半径``R``上 ⟹ ``|M| = W·R·sinα``。
        weight = 2.0 * 9810.0 / 1000.0
        expected = weight * 10.0 * math.sin(math.radians(25.0))
        assert abs(gradient[layout.rotation_base(0) + 1]) == pytest.approx(
            expected, rel=1e-6
        )

    def test_the_degeneracy_really_can_break(self) -> None:
        """**必红用例**：转动块**不**钉住时两条路必须走向不同的结局。

        没有这一条，上一条门在"转动根本没接上"时也是绿的。
        """

        from physics_engine.solve import SolveError

        #: 不带耦合项时转动自由度**不受任何能量项约束** → Hessian那一行整行为零，
        #: `solve_equilibrium`必须失败关闭而不是返回垃圾解。
        layout, registry, context, initial, fixed = _degeneracy_problem(with_coupling=False)
        free = frozenset(index for index in fixed if index != layout.rotation_base(0) + 1)
        with pytest.raises(SolveError, match="singular system"):
            solve_equilibrium(
                registry, context, layout.layout, initial,
                fixed_indices=free, residual_tol_n=1.0e-8, max_iterations=8,
            )

        #: 带耦合项时那一行不再是零，于是求解器走得动——它走到哪里是另一回事
        #: （斜面上的球**没有**静平衡，见`tests/cases/`那条），
        #: 这里只判"两条路的结局不同"。
        layout, registry, context, initial, fixed = _degeneracy_problem(with_coupling=True)
        free = frozenset(index for index in fixed if index != layout.rotation_base(0) + 1)
        result = solve_equilibrium(
            registry, context, layout.layout, initial,
            fixed_indices=free, residual_tol_n=1.0e-8, max_iterations=8,
        )
        assert not result.converged, (
            "球在斜面上竟然解出了静平衡——力矩平衡要求摩擦力为零，"
            "而力平衡要求它等于W·sinα，两者不相容"
        )


class TestFiniteDifference:
    """梯度对能量、Hessian对梯度。**转动那三个分量单独也要判。**"""

    @staticmethod
    def _fixture():
        layout = build_rigid_body_layout(
            layout_id="layout/fd", node_count=2, rotating_bodies=(0, 1)
        )
        springs = (
            StickSpring(
                first=MaterialPoint(0, (1.0, -2.0, 0.5), layout.rotation_base(0)),
                second=MaterialPoint(1, (-0.3, 0.7, 1.1), layout.rotation_base(1)),
                normal=(0.0, 0.6, 0.8),
                stiffness_n_per_mm=3.0,
                anchor_mm=(0.2, -0.1, 0.4),
            ),
            StickSpring(
                first=MaterialPoint(1, (0.4, 0.9, -1.2), layout.rotation_base(1)),
                normal=(1.0, 0.0, 0.0),
                stiffness_n_per_mm=5.0,
                anchor_mm=(1.0, 2.0, 3.0),
            ),
        )
        context = EnergyContext(context_id="context/fd", node_masses_kg=(1.0, 1.0))
        vector = (
            0.3, -0.2, 1.7, 2.4, 0.9, -1.1,
            0.21, -0.13, 0.34, -0.4, 0.15, 0.27,
        )
        return layout, springs, context, vector

    @pytest.mark.parametrize("kind", ("material_stick", "rotation_coupling"))
    def test_the_gradient_matches_the_energy(self, kind) -> None:
        layout, springs, context, vector = self._fixture()
        term = (
            MaterialPointStickSpring(springs=springs)
            if kind == "material_stick"
            else RotationStickCoupling(springs=springs)
        )
        gradient = term.gradient(State(layout=layout.layout, vector=vector), context)
        step = 1.0e-6
        for index in range(len(vector)):
            high = list(vector)
            low = list(vector)
            high[index] += step
            low[index] -= step
            numeric = (
                term.energy(State(layout=layout.layout, vector=tuple(high)), context)
                - term.energy(State(layout=layout.layout, vector=tuple(low)), context)
            ) / (2.0 * step)
            assert numeric == pytest.approx(gradient[index], abs=1.0e-6)

    @pytest.mark.parametrize("kind", ("material_stick", "rotation_coupling"))
    def test_the_hessian_matches_the_gradient(self, kind) -> None:
        layout, springs, context, vector = self._fixture()
        term = (
            MaterialPointStickSpring(springs=springs)
            if kind == "material_stick"
            else RotationStickCoupling(springs=springs)
        )
        state = State(layout=layout.layout, vector=vector)
        hessian = term.hessian(state, context)
        step = 1.0e-6
        for index in range(len(vector)):
            high = list(vector)
            low = list(vector)
            high[index] += step
            low[index] -= step
            gh = term.gradient(State(layout=layout.layout, vector=tuple(high)), context)
            gl = term.gradient(State(layout=layout.layout, vector=tuple(low)), context)
            for row in range(len(vector)):
                numeric = (gh[row] - gl[row]) / (2.0 * step)
                assert numeric == pytest.approx(hessian[row][index], abs=1.0e-5)

    def test_the_rotation_block_alone_converges_at_second_order(self) -> None:
        """**转动那三个分量单独判**：中心差分的收敛阶必须是二阶。

        整条向量一起判会被平动分量的大数盖过去——那正是"跑得通但全错"的形状。
        """

        layout, springs, context, vector = self._fixture()
        term = RotationStickCoupling(springs=springs)
        gradient = term.gradient(State(layout=layout.layout, vector=vector), context)
        rotation_indices = sorted(layout.rotation_indices())
        errors = []
        for step in (1.0e-2, 1.0e-3, 1.0e-4):
            worst = 0.0
            for index in rotation_indices:
                high = list(vector)
                low = list(vector)
                high[index] += step
                low[index] -= step
                numeric = (
                    term.energy(State(layout=layout.layout, vector=tuple(high)), context)
                    - term.energy(State(layout=layout.layout, vector=tuple(low)), context)
                ) / (2.0 * step)
                worst = max(worst, abs(numeric - gradient[index]))
            errors.append(worst)
        for first, second in zip(errors, errors[1:], strict=False):
            assert 50.0 < first / second < 200.0, f"转动分量的收敛阶不是二阶：{errors}"

    def test_the_geometric_stiffness_is_really_there(self) -> None:
        """几何刚度（``offset·P(∂²(Rℓ)/∂θ²)``）**漏掉了梯度照样对**，所以单独判它。

        做法：把弹簧拉出一个非零的``offset``，然后比较转动块对角线上的Hessian
        与"只有弹簧刚度那一块"的值——两者必须不同。
        """

        layout, springs, context, vector = self._fixture()
        term = RotationStickCoupling(springs=springs)
        state = State(layout=layout.layout, vector=vector)
        hessian = term.hessian(state, context)
        base = layout.rotation_base(1)
        directions = rotate_jacobian(
            (vector[base], vector[base + 1], vector[base + 2]), (0.4, 0.9, -1.2)
        )
        normal = (1.0, 0.0, 0.0)

        def project(value):
            along = sum(value[i] * normal[i] for i in range(3))
            return tuple(value[i] - along * normal[i] for i in range(3))

        spring_only = 5.0 * sum(
            project(directions[1])[i] * project(directions[1])[i] for i in range(3)
        )
        #: 第二根弹簧只挂在体1上，所以它对``(base+1, base+1)``的贡献可以单独算出来。
        #: 但第一根弹簧也挂在体1上，故这里只断言"总量与纯弹簧刚度不同"。
        assert hessian[base + 1][base + 1] != pytest.approx(spring_only, rel=1e-12)


class TestFailClosed:
    """声明层的必红用例。**豁免必须有名字，错误必须有理由。**"""

    def test_a_non_unit_normal_fails_closed(self) -> None:
        with pytest.raises(RotationError, match="unit vector"):
            StickSpring(
                first=MaterialPoint(0, (1.0, 0.0, 0.0)),
                normal=(0.0, 0.0, 2.0),
                stiffness_n_per_mm=1.0,
            )

    def test_a_nan_normal_fails_closed(self) -> None:
        """``abs(nan − 1.0) > tol``是``False``——单位矢量那道门挡不住nan。

        `contact.friction`在2026-08-06的对抗审核上正是栽在这里，本模块从出生就堵。
        """

        with pytest.raises(RotationError, match="finite 3-vector"):
            StickSpring(
                first=MaterialPoint(0, (1.0, 0.0, 0.0)),
                normal=(float("nan"), 0.0, 0.0),
                stiffness_n_per_mm=1.0,
            )

    def test_a_nonpositive_stiffness_fails_closed(self) -> None:
        with pytest.raises(RotationError, match="stiffness must be positive"):
            StickSpring(
                first=MaterialPoint(0, (1.0, 0.0, 0.0)),
                normal=(0.0, 0.0, 1.0),
                stiffness_n_per_mm=0.0,
            )

    def test_a_negative_node_fails_closed(self) -> None:
        """``node = -1``读的是向量尾巴——`PenaltySphereContact`吃过这个亏。"""

        with pytest.raises(RotationError, match="nonnegative int"):
            MaterialPoint(-1, (1.0, 0.0, 0.0))

    def test_a_coupling_without_any_rotating_end_fails_closed(self) -> None:
        """**必红用例**：一个恒为零的耦合项会让读者以为转动接上了。"""

        with pytest.raises(RotationError, match="no rotating end"):
            RotationStickCoupling(
                springs=(
                    StickSpring(
                        first=MaterialPoint(0, (1.0, 0.0, 0.0)),
                        normal=(0.0, 0.0, 1.0),
                        stiffness_n_per_mm=1.0,
                    ),
                )
            )

    def test_two_applied_moments_on_one_block_fail_closed(self) -> None:
        with pytest.raises(RotationError, match="two applied moments"):
            AppliedMoment(moments=((3, (1.0, 0.0, 0.0)), (3, (0.0, 1.0, 0.0))))

    def test_an_empty_spring_list_fails_closed(self) -> None:
        with pytest.raises(RotationError, match="at least one spring"):
            MaterialPointStickSpring(springs=())


class TestAppliedMoment:
    """`PointLoad`的转动对应物。**符号是这个项的全部内容。**"""

    def test_the_sign_makes_the_moment_do_positive_work(self) -> None:
        layout = build_rigid_body_layout(
            layout_id="layout/moment", node_count=1, rotating_bodies=(0,)
        )
        base = layout.rotation_base(0)
        term = AppliedMoment(moments=((base, (0.0, 7.0, 0.0)),))
        context = EnergyContext(context_id="context/moment", node_masses_kg=(1.0,))
        turned = list(layout.initial_vector((0.0, 0.0, 0.0)))
        turned[base + 1] = 0.5
        rest = State(layout=layout.layout, vector=layout.initial_vector((0.0, 0.0, 0.0)))
        moved = State(layout=layout.layout, vector=tuple(turned))
        assert term.energy(moved, context) < term.energy(rest, context)
        assert term.gradient(rest, context)[base + 1] == -7.0

    def test_the_hessian_is_exactly_zero(self) -> None:
        """能量对``θ``线性 ⟹ 切线刚度没有任何贡献。**这是`PointLoad`那条陷阱。**"""

        layout = build_rigid_body_layout(
            layout_id="layout/moment", node_count=1, rotating_bodies=(0,)
        )
        term = AppliedMoment(moments=((layout.rotation_base(0), (1.0, 2.0, 3.0)),))
        context = EnergyContext(context_id="context/moment", node_masses_kg=(1.0,))
        state = State(layout=layout.layout, vector=layout.initial_vector((0.0, 0.0, 0.0)))
        assert term.hessian_entries(state, context) == ()
        assert all(value == 0.0 for row in term.hessian(state, context) for value in row)


def test_pinning_the_rotation_reproduces_the_rotation_free_solve_bit_for_bit() -> None:
    """**本文件最硬的一条。** 同一个接触问题两条路，逐位判`float.hex()`。"""

    results = []
    for with_coupling in (False, True):
        layout, registry, context, initial, fixed = _degeneracy_problem(
            with_coupling=with_coupling
        )
        result = solve_equilibrium(
            registry,
            context,
            layout.layout,
            initial,
            fixed_indices=fixed,
            residual_tol_n=1.0e-8,
            max_iterations=60,
        )
        assert result.converged
        energy, gradient, _ = registry.total(
            result.state, context, need_gradient=True
        )
        results.append((result, energy, gradient))

    without, with_it = results
    assert [v.hex() for v in without[0].state.vector] == [
        v.hex() for v in with_it[0].state.vector
    ], "解向量逐位不同——退化不成立"
    assert without[1].hex() == with_it[1].hex(), "总能量逐位不同"
    #: **节点块**梯度逐位相同。转动块上两者当然不同——那正是新增的那个力矩，
    #: 由下一条断言单独判。把它一起塞进"逐位相同"里会把新增能力也判成回归。
    assert [v.hex() for v in without[2][:3]] == [
        v.hex() for v in with_it[2][:3]
    ], "节点块梯度逐位不同"
    assert without[0].iterations == with_it[0].iterations
    assert without[0].backtracks == with_it[0].backtracks
    assert without[0].residual_n.hex() == with_it[0].residual_n.hex()

    #: 而两条路**必须**在转动块上不同，否则上面那些"相同"只是因为什么都没接上。
    rotation_index = with_it[0].state.layout.dof_count - 2
    assert without[2][rotation_index] == 0.0
    assert abs(with_it[2][rotation_index]) > 1.0, (
        "带耦合项时转动块梯度仍是零——这个项什么也没做，"
        "上面那几条逐位相同就成了空门"
    )
