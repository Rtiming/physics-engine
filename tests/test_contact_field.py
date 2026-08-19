"""`physics_engine.contact.field`的判据——甲2（决策0085第三节、0074第二节）。

**每道门配一个必红**（AGENTS.md）。本文件里判的东西分四组，不要混：

1. **插值本身**：单位分解、导数和为零、二阶矩恒为1/3——这三条决定了那个``h²/6``；
2. **逼近阶**：拿仓里已有的三条解析SDF采样成场，量``g``／``∇g``／``∇²g``随``h``的阶。
   半空间是**仿射**所以逐位精确（连截断项都没有），球与圆柱实测**二阶**；
3. **接触项**：能量的FD梯度二阶收敛比恒4.0000（本仓样板），
   Hessian的FD同样恒4.0000，协议与`PenaltyNormalContact`逐条对齐；
4. **窄带外**：失败关闭（0085裁的），必红是"查一个没烘的点必须抛"。

**C²那一条单独一节**，因为它是0074第二节第4条那句"三线性不够"的实测出口：
同一份场、同一个胞界，三次B样条的``∇g``跳幅随``ε``线性趋零，
三线性的``∂g/∂x``跳幅**恒为5.082e-02、与``ε``无关**——那是一个真的间断。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact.field import (
    PenaltySignedDistanceField,
    SignedDistanceField,
    SignedDistanceFieldError,
    _weights,
    _weights_d1,
    _weights_d2,
    cylinder_distance_mm,
    half_space_distance_mm,
    sample_narrow_band,
    sphere_distance_mm,
)
from physics_engine.contact.penalty import PenaltyNormalContact
from physics_engine.energies import EnergyContext
from physics_engine.state import State, StateField, StateLayout

LAYOUT = StateLayout(
    layout_id="layout/sdf-probe", fields=(StateField(name="position_mm", width=3),)
)
CONTEXT = EnergyContext(context_id="context/sdf-probe", node_masses_kg=(1.0,))
STIFFNESS_N_PER_MM = 1000.0

SPHERE_CENTRE = (0.0, 0.0, 0.0)
SPHERE_RADIUS_MM = 10.0
PLANE_POINT = (0.0, 0.0, 0.0)
PLANE_NORMAL = (0.0, 0.0, 1.0)
AXIS_POINT = (0.0, 0.0, 0.0)
AXIS_DIRECTION = (0.0, 0.0, 1.0)
CYLINDER_RADIUS_MM = 7.0


def _state(point: tuple[float, float, float]) -> State:
    return State(layout=LAYOUT, vector=tuple(point))


def _sphere_field(spacing_mm: float, *, band_mm: float = 3.5) -> SignedDistanceField:
    extent = 13.0
    count = int(2.0 * extent / spacing_mm) + 1
    return sample_narrow_band(
        lambda point: sphere_distance_mm(point, SPHERE_CENTRE, SPHERE_RADIUS_MM),
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=spacing_mm,
        node_counts=(count, count, count),
        band_mm=band_mm,
    )


def _plane_field(spacing_mm: float) -> SignedDistanceField:
    extent = 10.0
    count = int(2.0 * extent / spacing_mm) + 1
    return sample_narrow_band(
        lambda point: half_space_distance_mm(point, PLANE_POINT, PLANE_NORMAL),
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=spacing_mm,
        node_counts=(count, count, count),
        band_mm=8.0,
    )


def _tight_field() -> SignedDistanceField:
    """带薄（1.8 mm）、块细（block_log2=2，即4³＝64个节点/块）的同一个球。

    **专为"窄带外"那三条门烘**：默认的8³块在这个尺度上太粗，
    粗到球心那一块也被留下来——那时"带外"演示不出来。
    这不是把演示凑出来，是块粒度**真的**决定了带外从哪里开始，
    而这条性质本身就是`test_the_block_granularity_eats_the_narrow_band_saving`判的东西。
    """

    extent = 13.0
    return sample_narrow_band(
        lambda point: sphere_distance_mm(point, SPHERE_CENTRE, SPHERE_RADIUS_MM),
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=0.5,
        node_counts=(53, 53, 53),
        band_mm=1.8,
        block_log2=2,
    )


def _cylinder_field(spacing_mm: float) -> SignedDistanceField:
    extent = 11.0
    count = int(2.0 * extent / spacing_mm) + 1
    return sample_narrow_band(
        lambda point: cylinder_distance_mm(
            point, AXIS_POINT, AXIS_DIRECTION, CYLINDER_RADIUS_MM
        ),
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=spacing_mm,
        node_counts=(count, count, count),
        band_mm=3.5,
    )


def _shell_points() -> tuple[tuple[float, float, float], ...]:
    """球面附近一圈查询点。**不取节点上的点**——节点上插值恰好落回样本，
    量出来的是0而不是逼近误差。"""

    points = []
    for a in (0.3, 0.9, 1.7, 2.5):
        for b in (0.2, 1.1, 2.0):
            radius = 9.3 + 0.37 * a
            points.append(
                (
                    radius * math.sin(a) * math.cos(b),
                    radius * math.sin(a) * math.sin(b),
                    radius * math.cos(a),
                )
            )
    return tuple(points)


def _ratios(errors: list[float]) -> list[float]:
    return [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]


# --------------------------- 一、插值本身 ---


@pytest.mark.parametrize("t", [0.0, 0.125, 0.5, 0.7734375, 0.999])
def test_the_cubic_bspline_weights_are_a_partition_of_unity(t: float) -> None:
    """和为1、一阶导和为0、二阶导和为0。**常函数进去必须常函数出来。**"""

    assert sum(_weights(t)) == pytest.approx(1.0, abs=1.0e-15)
    assert sum(_weights_d1(t)) == pytest.approx(0.0, abs=1.0e-15)
    assert sum(_weights_d2(t)) == pytest.approx(0.0, abs=1.0e-15)


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75])
def test_the_second_moment_of_the_cubic_bspline_is_one_third_at_every_t(t: float) -> None:
    """``Σ_k (k − t)²·B(k − t) = 1/3``，**与``t``无关**。

    这个常数就是模块docstring那条``S(x) = f(x) + (h²/6)·f''(x)``里的``1/6``
    （``½ · ⅓``）。它与``t``无关，正是"误差的主项不随查询点在胞内的位置摆动"
    ——于是在**固定点**上加密``h``也能量出干净的二阶，
    不必对一堆点取最大值来抹平胞内相位。
    """

    weights = _weights(t)
    offsets = (-1.0 - t, -t, 1.0 - t, 2.0 - t)
    moment = sum(w * d * d for w, d in zip(weights, offsets, strict=True))
    assert moment == pytest.approx(1.0 / 3.0, rel=1.0e-14)
    #: 一阶矩为零——仿射函数被精确重构的直接原因。
    first = sum(w * d for w, d in zip(weights, offsets, strict=True))
    assert first == pytest.approx(0.0, abs=1.0e-15)


def test_the_stored_samples_round_trip_through_the_block_hash() -> None:
    """块坐标哈希 + 块内行主序偏移的**往返**。

    这道门守的是一类静默到极点的缺陷：索引算错时场照样能查、照样光滑、
    照样收敛——**只是收敛到了一个平移过的形状**。
    所以这里逐节点对拍解析值，不留任何一个节点。
    """

    field = _sphere_field(1.0)
    width = field.block_width
    checked = 0
    for bi, bj, bk in field.blocks:
        for di in range(width):
            for dj in range(width):
                for dk in range(width):
                    i, j, k = bi * width + di, bj * width + dj, bk * width + dk
                    position = field.node_position_mm(i, j, k)
                    expected = sphere_distance_mm(
                        position, SPHERE_CENTRE, SPHERE_RADIUS_MM
                    )
                    assert field.sample_at(i, j, k).hex() == expected.hex()  # type: ignore[union-attr]
                    checked += 1
    assert checked == field.stored_node_count
    assert checked > 20000, checked


def test_the_block_granularity_eats_the_narrow_band_saving() -> None:
    """窄带省多少，**块粒度说了算**——这是本片量出来的一条实测账，写在门上。

    同一个球（R = 10 mm）、同一个包围盒（±13 mm）、同一个``h = 0.5``：

    | 带宽 | 块 | 存下的节点 | 稠密 | 比 |
    |---|---|---|---|---|
    | 3.5 mm | 8³ | 134144 | 148877 | **0.9010** |
    | 1.8 mm | 4³ | 60480 | 148877 | **0.4062** |

    第一行是要紧的那一行：**8³块把带撑到了几乎整个包围盒**。
    0074第5.1节那张MB表按"±4胞窄带"估，它**没有把块粒度算进去**——
    在包围盒本来就贴着物体的件上，用8³块存一条3.5 mm的带省不下什么。
    这不是缺陷（块粒度是OpenVDB形制的一部分），是一条**必须写在明处的代价**：
    要省内存就得让带宽与块宽同量级，而带宽的下限被支撑宽度``2h√3``钉着。

    这条门同时挡住"块剔除写错成一个都不剔"（那时两个数会相等）。
    """

    coarse = _sphere_field(0.5)
    tight = _tight_field()
    assert coarse.stored_node_count == 134144
    assert tight.stored_node_count == 60480
    assert coarse.dense_node_count == tight.dense_node_count == 148877
    assert 0.89 < coarse.stored_node_count / coarse.dense_node_count < 0.91
    assert 0.40 < tight.stored_node_count / tight.dense_node_count < 0.41
    #: 不许剔过头——带里的点必须查得了。
    assert coarse.contains_stencil((0.0, 0.0, SPHERE_RADIUS_MM - 0.1))
    assert tight.contains_stencil((0.0, 0.0, SPHERE_RADIUS_MM - 0.1))


# --------------------------- 二、逼近阶 ---


def test_the_half_space_is_reproduced_to_roundoff_at_every_resolution() -> None:
    """半空间是**仿射**的，三次B样条精确重构它——误差里连截断项都没有。

    实测：``h``取2.0/1.0/0.5三档，值的绝对误差全在1e-15以内、
    ``∇g``就是``(0, 0, 1)``到末位、``∇²g``全零到末位。
    **这条不是"精度很好"，是"阶量不出来因为误差已经是舍入"**——
    所以本条门判的是绝对量，不判阶。想量阶要去球与圆柱那两条。
    """

    query = (0.37, -1.23, 0.813)
    exact = half_space_distance_mm(query, PLANE_POINT, PLANE_NORMAL)
    for spacing in (2.0, 1.0, 0.5):
        field = _plane_field(spacing)
        value, slope, curvature = field.evaluate(
            query, need_gradient=True, need_hessian=True
        )
        assert abs(value - exact) < 1.0e-14
        assert slope[0] == pytest.approx(0.0, abs=1.0e-15)  # type: ignore[index]
        assert slope[1] == pytest.approx(0.0, abs=1.0e-15)  # type: ignore[index]
        assert slope[2] == pytest.approx(1.0, abs=1.0e-15)  # type: ignore[index]
        assert (
            max(abs(curvature[i][j]) for i in range(3) for j in range(3))  # type: ignore[index]
            < 1.0e-14
        )


def test_the_sphere_field_converges_at_second_order_in_all_three_quantities() -> None:
    """``g``／``∇g``／``∇²g``全部二阶。实测比：

    * ``g``：**4.0012 / 4.0002**
    * ``∇g``：**4.0008 / 3.9998**
    * ``∇²g``：**4.0784 / 3.9263**（二阶导那一支带胞内相位抖动，见下）

    ``∇²g``的窗口放到``[3.8, 4.2]``而不是``[3.9, 4.1]``，理由是实测而不是让步：
    二阶导的误差里除了``(h²/6)f''''``还有一支**随胞内相位摆动**的项，
    在取最大值的口径下它不会被抹平。**窗口宽度是量出来的，不是挑出来的**——
    `test_the_order_window_is_tight_enough_to_reject_first_order`证明它仍有分辨力。
    """

    points = _shell_points()

    def _exact_gradient(point):
        norm = math.sqrt(sum(c * c for c in point))
        return tuple(c / norm for c in point)

    def _exact_hessian(point):
        norm = math.sqrt(sum(c * c for c in point))
        direction = tuple(c / norm for c in point)
        return tuple(
            tuple(
                ((1.0 if i == j else 0.0) - direction[i] * direction[j]) / norm
                for j in range(3)
            )
            for i in range(3)
        )

    value_errors: list[float] = []
    gradient_errors: list[float] = []
    hessian_errors: list[float] = []
    for spacing in (1.0, 0.5, 0.25):
        field = _sphere_field(spacing)
        worst_value = worst_gradient = worst_hessian = 0.0
        for query in points:
            value, slope, curvature = field.evaluate(
                query, need_gradient=True, need_hessian=True
            )
            exact_value = sphere_distance_mm(query, SPHERE_CENTRE, SPHERE_RADIUS_MM)
            exact_slope = _exact_gradient(query)
            exact_curvature = _exact_hessian(query)
            worst_value = max(worst_value, abs(value - exact_value))
            worst_gradient = max(
                worst_gradient,
                max(abs(slope[i] - exact_slope[i]) for i in range(3)),  # type: ignore[index]
            )
            worst_hessian = max(
                worst_hessian,
                max(
                    abs(curvature[i][j] - exact_curvature[i][j])  # type: ignore[index]
                    for i in range(3)
                    for j in range(3)
                ),
            )
        value_errors.append(worst_value)
        gradient_errors.append(worst_gradient)
        hessian_errors.append(worst_hessian)

    assert all(3.9 < ratio < 4.1 for ratio in _ratios(value_errors)), value_errors
    assert all(3.9 < ratio < 4.1 for ratio in _ratios(gradient_errors)), gradient_errors
    assert all(3.8 < ratio < 4.2 for ratio in _ratios(hessian_errors)), hessian_errors
    #: 误差确实降到了小数，否则上面那串比值可以由三个大数凑出来。
    assert value_errors[-1] < 3.0e-3
    assert gradient_errors[-1] < 3.0e-4
    assert hessian_errors[-1] < 1.0e-4


def test_the_cylinder_field_converges_at_second_order() -> None:
    """圆柱侧面：``ρ − R``。轴向是**仿射方向**（场沿轴不变），
    所以全部误差来自径向那一维——阶与球同为二阶，来源不同。"""

    points = tuple(
        (
            (CYLINDER_RADIUS_MM - 0.4 + 0.3 * k) * math.cos(0.4 * k),
            (CYLINDER_RADIUS_MM - 0.4 + 0.3 * k) * math.sin(0.4 * k),
            -3.0 + 1.7 * k,
        )
        for k in range(5)
    )
    errors: list[float] = []
    for spacing in (1.0, 0.5, 0.25):
        field = _cylinder_field(spacing)
        errors.append(
            max(
                abs(
                    field.value_mm(query)
                    - cylinder_distance_mm(
                        query, AXIS_POINT, AXIS_DIRECTION, CYLINDER_RADIUS_MM
                    )
                )
                for query in points
            )
        )
    assert all(3.9 < ratio < 4.1 for ratio in _ratios(errors)), errors
    assert errors[-1] < 3.0e-3


def test_the_order_window_is_tight_enough_to_reject_first_order() -> None:
    """必须红：一阶的误差序列（比值2）必须落在窗口外。

    这条判的不是场，是**判据本身有没有分辨力**——一个宽到什么都能过的窗口
    与没有窗口是同一件事。这里直接拿一条``h¹``的合成序列去撞它。
    """

    first_order = [1.0e-2, 5.0e-3, 2.5e-3]
    assert not all(3.9 < ratio < 4.1 for ratio in _ratios(first_order))
    assert not all(3.8 < ratio < 4.2 for ratio in _ratios(first_order))
    assert _ratios(first_order) == [pytest.approx(2.0), pytest.approx(2.0)]


# --------------------------- 三、C²：0074第二节第4条的实测出口 ---


def _trilinear_x_slope(field: SignedDistanceField, point) -> float:
    """三线性插值的``∂g/∂x``。**只在测试里存在**——它是被否掉的那条路的证人。"""

    h = field.spacing_mm
    u = [(point[a] - field.origin_mm[a]) / h for a in range(3)]
    base = [math.floor(v) for v in u]
    t = [u[a] - base[a] for a in range(3)]
    slope = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                sample = field.sample_at(base[0] + a, base[1] + b, base[2] + c)
                assert sample is not None
                weight_y = t[1] if b else 1.0 - t[1]
                weight_z = t[2] if c else 1.0 - t[2]
                slope += sample * (1.0 if a else -1.0) * weight_y * weight_z / h
    return slope


def test_the_bspline_is_c2_across_a_cell_boundary_and_trilinear_is_not() -> None:
    """**同一份场、同一个胞界**，两种插值的行为差一个量级差到无穷。

    实测（``h = 0.5``、胞界恰在``x = 0``）：

    | ``ε`` | B样条``\\|Δ∇g\\|`` | B样条``\\|Δ∇²g\\|`` | 三线性``\\|Δ∂g/∂x\\|`` |
    |---|---|---|---|
    | 1e-3 | 2.033e-04 | 2.065e-05 | **5.082e-02** |
    | 1e-4 | 2.033e-05 | 2.065e-06 | **5.082e-02** |
    | 1e-5 | 2.033e-06 | 2.065e-07 | **5.082e-02** |
    | 1e-6 | 2.033e-07 | 2.065e-08 | **5.082e-02** |

    B样条那两列随``ε``线性趋零（=连续）；三线性那一列**一位都不动**——
    它是一个真的间断。`solve.py`第29行申报的适用域"``U``二次连续可微"
    在三线性下**当场不成立**，牛顿会在胞界上抖。
    这就是0074第二节第4条那句"不是精度偏好，是适用域的硬要求"的全部内容。
    """

    field = _sphere_field(0.5)
    #: ``origin = -13.0``、``h = 0.5``，于是``x = 0``恰是第26个节点=一个胞界。
    assert (0.0 - field.origin_mm[0]) / field.spacing_mm == 26.0

    trilinear_jumps = []
    for epsilon in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        left = (-epsilon, 0.31, 9.83)
        right = (epsilon, 0.31, 9.83)
        _, slope_l, curvature_l = field.evaluate(
            left, need_gradient=True, need_hessian=True
        )
        _, slope_r, curvature_r = field.evaluate(
            right, need_gradient=True, need_hessian=True
        )
        gradient_jump = max(abs(slope_l[i] - slope_r[i]) for i in range(3))  # type: ignore[index]
        hessian_jump = max(
            abs(curvature_l[i][j] - curvature_r[i][j])  # type: ignore[index]
            for i in range(3)
            for j in range(3)
        )
        assert gradient_jump < 300.0 * epsilon, (epsilon, gradient_jump)
        assert hessian_jump < 300.0 * epsilon, (epsilon, hessian_jump)
        trilinear_jumps.append(
            abs(_trilinear_x_slope(field, left) - _trilinear_x_slope(field, right))
        )

    #: 三线性的跳幅**与``ε``无关**——四档全部落在同一个数上，且不是零。
    assert min(trilinear_jumps) > 1.0e-2, trilinear_jumps
    assert max(trilinear_jumps) / min(trilinear_jumps) < 1.0 + 1.0e-6, trilinear_jumps


# --------------------------- 四、接触项 ---


def _sphere_term(field: SignedDistanceField, radius_mm: float = 0.6):
    return PenaltySignedDistanceField(
        field=field, contacts=((0, STIFFNESS_N_PER_MM, radius_mm),)
    )


ACTIVE_POINT = (0.13, 0.29, 10.17)


def test_the_gradient_is_the_derivative_of_the_energy_at_second_order() -> None:
    """**中心差分二阶收敛，实测比恒4.0000**——本仓样板（0078那条）。

    实测误差：5.7845e-04 / 1.4461e-04 / 3.6153e-05 / 9.0383e-06。
    这条门是接触项存在的技术前提：梯度不是某个势的导数时，
    线搜索与收敛判据全部失去依据。
    """

    field = _sphere_field(0.5)
    term = _sphere_term(field)
    assert term.gap_mm(_state(ACTIVE_POINT))[0] < 0.0
    analytic = term.gradient(_state(ACTIVE_POINT), CONTEXT)

    errors = []
    for step in (0.02, 0.01, 0.005, 0.0025):
        measured = []
        for axis in range(3):
            ahead = tuple(
                ACTIVE_POINT[i] + (step if i == axis else 0.0) for i in range(3)
            )
            behind = tuple(
                ACTIVE_POINT[i] - (step if i == axis else 0.0) for i in range(3)
            )
            measured.append(
                (
                    term.energy(_state(ahead), CONTEXT)
                    - term.energy(_state(behind), CONTEXT)
                )
                / (2.0 * step)
            )
        errors.append(max(abs(a - b) for a, b in zip(measured, analytic, strict=True)))

    ratios = _ratios(errors)
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    assert errors[-1] < 1.0e-5


def test_the_hessian_is_the_derivative_of_the_gradient_at_second_order() -> None:
    """``H = k(∇g⊗∇g) + k·g·∇²g``——**第二块只有场能给**，所以它必须被单独验。

    实测比恒**4.0000**（误差1.9253e-03 / 4.8132e-04 / 1.2033e-04 / 3.0083e-05）。
    半空间那一族第二块恒为零，于是这条门是本项相对`PenaltyNormalContact`
    **多出来的那一半**的唯一证人。
    """

    field = _sphere_field(0.5)
    term = _sphere_term(field)
    analytic = term.hessian(_state(ACTIVE_POINT), CONTEXT)

    errors = []
    for step in (0.02, 0.01, 0.005, 0.0025):
        worst = 0.0
        for axis in range(3):
            ahead = tuple(
                ACTIVE_POINT[i] + (step if i == axis else 0.0) for i in range(3)
            )
            behind = tuple(
                ACTIVE_POINT[i] - (step if i == axis else 0.0) for i in range(3)
            )
            forward = term.gradient(_state(ahead), CONTEXT)
            backward = term.gradient(_state(behind), CONTEXT)
            for row in range(3):
                column = (forward[row] - backward[row]) / (2.0 * step)
                worst = max(worst, abs(column - analytic[row][axis]))
        errors.append(worst)

    ratios = _ratios(errors)
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    assert errors[-1] < 1.0e-4


def test_dropping_the_curvature_block_would_be_caught_by_the_hessian_gate() -> None:
    """必须红：把``k·g·∇²g``那一块丢掉（=照抄半空间那一族），上一条门必须红。

    这是本仓最容易犯的一种错——**梯度照样对、平衡点照样对**，
    只有收敛速度与稳定性判据会变（`PenaltySphereContact`docstring原话）。
    """

    field = _sphere_field(0.5)
    term = _sphere_term(field)
    state = _state(ACTIVE_POINT)
    full = term.hessian(state, CONTEXT)

    value, slope, curvature = field.evaluate(
        ACTIVE_POINT, need_gradient=True, need_hessian=True
    )
    gap = value - 0.6
    outer_only = tuple(
        tuple(STIFFNESS_N_PER_MM * slope[a] * slope[b] for b in range(3))  # type: ignore[index]
        for a in range(3)
    )
    deviation = max(
        abs(full[a][b] - outer_only[a][b]) for a in range(3) for b in range(3)
    )
    #: 丢掉的那一块的量级是``k·|g|·|∇²g| ≈ 1000 · 0.417 · 0.1``——**不是末位噪声**。
    assert deviation > 1.0
    assert deviation == pytest.approx(
        STIFFNESS_N_PER_MM
        * abs(gap)
        * max(abs(curvature[a][b]) for a in range(3) for b in range(3)),  # type: ignore[index]
        rel=1.0e-12,
    )


def test_the_field_term_matches_the_analytic_half_space_term_to_roundoff() -> None:
    """**与`PenaltyNormalContact`并排**：同一构型、同一刚度、同一半径。

    半空间是仿射的，所以两条路的``g``只差舍入——于是能量、梯度、法向力
    三样在**任何**分辨率下都对得上到1e-12相对。
    这条门判的正是"SDF只是把``g``换个来路，形制一个字没改"。

    **阶在这条门上量不出来**（误差已经是舍入），要量阶去球那一条。
    """

    point = (1.7, -2.3, -0.42)
    analytic = PenaltyNormalContact(
        planes=((0, PLANE_POINT, PLANE_NORMAL, STIFFNESS_N_PER_MM, 0.25),)
    )
    for spacing in (2.0, 1.0, 0.5):
        term = PenaltySignedDistanceField(
            field=_plane_field(spacing), contacts=((0, STIFFNESS_N_PER_MM, 0.25),)
        )
        state = _state(point)
        assert term.energy(state, CONTEXT) == pytest.approx(
            analytic.energy(state, CONTEXT), rel=1.0e-12
        )
        measured = term.gradient(state, CONTEXT)
        expected = analytic.gradient(state, CONTEXT)
        assert max(abs(a - b) for a, b in zip(measured, expected, strict=True)) < 1.0e-9
        assert term.normal_force_n(state)[0] == pytest.approx(
            analytic.normal_force_n(state)[0], rel=1.0e-12
        )


def test_the_side_by_side_gate_would_go_red_on_a_shifted_plane() -> None:
    """必须红：把场的平面挪开0.1 mm，上一条并排门必须红。

    没有这条，"两边对得上"可能只是因为两边都是零（点分离时能量恒为0）。
    """

    point = (1.7, -2.3, -0.42)
    analytic = PenaltyNormalContact(
        planes=((0, PLANE_POINT, PLANE_NORMAL, STIFFNESS_N_PER_MM, 0.25),)
    )
    extent = 10.0
    field = sample_narrow_band(
        lambda p: half_space_distance_mm(p, (0.0, 0.0, 0.1), PLANE_NORMAL),
        origin_mm=(-extent, -extent, -extent),
        spacing_mm=1.0,
        node_counts=(21, 21, 21),
        band_mm=6.0,
    )
    term = PenaltySignedDistanceField(
        field=field, contacts=((0, STIFFNESS_N_PER_MM, 0.25),)
    )
    state = _state(point)
    assert analytic.energy(state, CONTEXT) > 0.0
    assert term.energy(state, CONTEXT) != pytest.approx(
        analytic.energy(state, CONTEXT), rel=1.0e-6
    )


def test_the_fused_path_reproduces_the_energy_bit_for_bit() -> None:
    """spec/12第3.1节：融合路径的能量值与单独调`energy`**逐字节**相同。"""

    field = _sphere_field(1.0)
    term = _sphere_term(field)
    state = _state(ACTIVE_POINT)
    alone = term.energy(state, CONTEXT)
    for need_gradient in (False, True):
        fused, gradient, _ = term.quantities(
            state, CONTEXT, need_gradient=need_gradient, need_hessian=False
        )
        assert fused.hex() == alone.hex(), need_gradient
        if need_gradient:
            direct = term.gradient(state, CONTEXT)
            for a, b in zip(gradient, direct, strict=True):
                assert a.hex() == b.hex()


def test_the_dense_hessian_is_exactly_the_assembled_entries() -> None:
    """`hessian`与`hessian_entries`是同一个矩阵的两种形状，逐字节。"""

    field = _sphere_field(1.0)
    term = _sphere_term(field)
    state = _state(ACTIVE_POINT)
    dense = term.hessian(state, CONTEXT)
    assembled = [[0.0] * 3 for _ in range(3)]
    for row, column, value in term.hessian_entries(state, CONTEXT):
        assembled[row][column] += value
    for i in range(3):
        for j in range(3):
            assert dense[i][j].hex() == assembled[i][j].hex()


def test_the_registry_sparse_and_dense_paths_agree_bitwise() -> None:
    """把本项装进`EnergyRegistry`，**稀疏读法与稠密路径逐位相同**。

    spec/13第一节义务2那条（`tests/test_energies.py`的
    `test_sparse_hessian_matches_the_dense_one_bitwise`是样板）。
    这条对新接触项是承重的：`solve_equilibrium`走的正是稀疏那一条，
    **一个新项如果只把`hessian`写对而`hessian_entries`写岔，
    单元门全绿而求解器用的是错的那一份**。
    """

    from physics_engine.energies import EnergyRegistry, PointLoad

    field = _sphere_field(1.0)
    registry = EnergyRegistry(
        terms=(_sphere_term(field), PointLoad(loads=((0, (0.0, 0.0, -25.0)),)))
    )
    state = _state(ACTIVE_POINT)
    _, _, dense = registry.total(state, CONTEXT, need_gradient=True, need_hessian=True)
    sparse = registry.hessian_entries(state, CONTEXT)
    assert dense is not None
    for (row, column), value in sparse.items():
        assert dense[row][column].hex() == value.hex(), (row, column)
    #: 稀疏里没有的位置必须在稠密里恰好是0.0——**"结构非零"这句话要能被验**。
    for row in range(len(state.vector)):
        for column in range(len(state.vector)):
            if (row, column) not in sparse:
                assert dense[row][column] == 0.0, (row, column)


def test_a_separated_contact_emits_nothing_at_all() -> None:
    """分离时能量、梯度、Hessian项、法向力全部为零。**一个非零项都不出。**"""

    field = _sphere_field(1.0)
    term = _sphere_term(field)
    outside = _state((0.0, 0.0, 11.4))
    assert term.gap_mm(outside)[0] > 0.0
    assert term.energy(outside, CONTEXT) == 0.0
    assert term.gradient(outside, CONTEXT) == (0.0, 0.0, 0.0)
    assert term.hessian_entries(outside, CONTEXT) == ()
    assert term.normal_force_n(outside) == (0.0,)


def test_the_normal_force_is_stiffness_times_penetration() -> None:
    """``N = k·|g|``——本项唯一精确的输出（照`PenaltyNormalContact`那条）。"""

    field = _sphere_field(0.5)
    term = _sphere_term(field)
    state = _state(ACTIVE_POINT)
    gap = term.gap_mm(state)[0]
    assert term.normal_force_n(state)[0].hex() == (STIFFNESS_N_PER_MM * -gap).hex()
    assert term.node_index_bound() == 1


# --------------------------- 五、窄带外：失败关闭 ---


def test_a_point_outside_the_narrow_band_fails_closed() -> None:
    """必须红：没烘的地方查询必须抛，**不外推、不返回0**。

    0085裁的正是这一条。理由写在模块docstring里：
    "远在体外"与"深在体内"在稀疏块表里长得一模一样。
    """

    field = _tight_field()
    far_outside = (0.0, 0.0, 12.0)
    deep_inside = (0.0, 0.0, 0.0)
    for query in (far_outside, deep_inside):
        assert not field.contains_stencil(query)
        with pytest.raises(SignedDistanceFieldError, match="outside the narrow band"):
            field.value_mm(query)


def test_the_two_kinds_of_outside_are_genuinely_indistinguishable() -> None:
    """**这条门就是那个裁决的理由本身。**

    球心（深在体内、``g = −10 mm``、接触力极大）与远在体外的点，
    在块表里给出的是**同一个答案**：块不存在。
    外推要在这两者之间猜一个，而猜错的那一半是静默的。
    """

    field = _tight_field()
    far_outside = (0.0, 0.0, 12.0)
    deep_inside = (0.0, 0.0, 0.0)
    #: 两者的真值差了六个量级的**符号**：一个不接触，一个接触力极大。
    assert sphere_distance_mm(far_outside, SPHERE_CENTRE, SPHERE_RADIUS_MM) == (
        pytest.approx(2.0)
    )
    assert sphere_distance_mm(deep_inside, SPHERE_CENTRE, SPHERE_RADIUS_MM) == (
        pytest.approx(-10.0)
    )

    signals = []
    for query in (far_outside, deep_inside):
        assert not field.contains_stencil(query)
        base = [
            math.floor((query[axis] - field.origin_mm[axis]) / field.spacing_mm) - 1
            for axis in range(3)
        ]
        holes = [
            (a, b, c)
            for a in range(4)
            for b in range(4)
            for c in range(4)
            if field.sample_at(base[0] + a, base[1] + b, base[2] + c) is None
        ]
        assert holes, query
        signals.append(
            {field.sample_at(base[0] + a, base[1] + b, base[2] + c) for a, b, c in holes}
        )

    #: **场给出的信号一模一样**：``{None}``。没有任何一个存下来的字节带符号，
    #: 所以"往哪一边外推"这个问题在这份存储里没有答案。
    assert signals == [{None}, {None}]


def test_the_band_gate_is_not_vacuous() -> None:
    """门不许恒红：带内的点必须查得了，且值与解析吻合。"""

    field = _sphere_field(1.0)
    inside = (0.0, 0.0, 9.73)
    assert field.contains_stencil(inside)
    assert field.value_mm(inside) == pytest.approx(
        sphere_distance_mm(inside, SPHERE_CENTRE, SPHERE_RADIUS_MM), abs=5.0e-2
    )


def test_a_band_thinner_than_the_stencil_is_refused_at_bake_time() -> None:
    """必须红：``band < 2h√3``会烘出"带里也查不了"的场，当场拒。"""

    with pytest.raises(SignedDistanceFieldError, match="thinner than"):
        sample_narrow_band(
            lambda p: sphere_distance_mm(p, SPHERE_CENTRE, SPHERE_RADIUS_MM),
            origin_mm=(-13.0, -13.0, -13.0),
            spacing_mm=1.0,
            node_counts=(27, 27, 27),
            band_mm=3.0,
        )


# --------------------------- 六、构造期校验 ---


def test_the_field_refuses_malformed_declarations() -> None:
    """必须红：五类畸形声明各一条。"""

    good = dict(
        origin_mm=(0.0, 0.0, 0.0),
        spacing_mm=1.0,
        node_counts=(8, 8, 8),
        band_mm=4.0,
    )
    with pytest.raises(SignedDistanceFieldError, match="spacing_mm"):
        SignedDistanceField(**{**good, "spacing_mm": 0.0})
    with pytest.raises(SignedDistanceFieldError, match="node_counts"):
        SignedDistanceField(**{**good, "node_counts": (8, 8, 3)})
    with pytest.raises(SignedDistanceFieldError, match="band_mm"):
        SignedDistanceField(**{**good, "band_mm": -1.0})
    with pytest.raises(SignedDistanceFieldError, match="block_log2"):
        SignedDistanceField(**{**good, "block_log2": 0})
    with pytest.raises(SignedDistanceFieldError, match="expected 512"):
        SignedDistanceField(**{**good, "blocks": {(0, 0, 0): (0.0,)}})


def test_the_contact_term_refuses_malformed_declarations() -> None:
    """必须红：节点下标、刚度、半径三条，逐条照`PenaltyNormalContact`。

    **负节点下标那一条是有前科的**：`PenaltySphereContact`当年漏了它，
    ``node = -1``读到的正是接触锚点槽（penalty.py那段注释记着316681 N·mm）。
    """

    field = _sphere_field(1.0)
    with pytest.raises(SignedDistanceFieldError, match="at least one"):
        PenaltySignedDistanceField(field=field, contacts=())
    with pytest.raises(SignedDistanceFieldError, match="nonnegative int"):
        PenaltySignedDistanceField(field=field, contacts=((-1, 1.0, 0.0),))
    with pytest.raises(SignedDistanceFieldError, match="stiffness"):
        PenaltySignedDistanceField(field=field, contacts=((0, 0.0, 0.0),))
    with pytest.raises(SignedDistanceFieldError, match="radius"):
        PenaltySignedDistanceField(field=field, contacts=((0, 1.0, -0.1),))
