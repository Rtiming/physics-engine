"""电磁域第一块的协议门（案例判据在`tests/cases/`，本文件不重复它们）。

分工照0024与0031的先例：**案例验物理，本文件验协议**——
公开面、失败关闭、单位边界的对称性、以及两个AGM坑的回归门。

本文件里有三条**必须红**的守门测试（`test_the_*_gate_is_red_on_*`）：
它们把"教科书分组会崩""AGM不终止会静默出错""约定混用差10%"三件事
各钉成一条会红的断言。**边界不是免责声明，是要有门守着的**——
写在案例页的已知失效清单里而没有门，半年后就没人知道它还成不成立。
"""

from __future__ import annotations

import math

import pytest

import physics_engine
from physics_engine import electromagnetics
from physics_engine.electromagnetics import (
    AGM_ITERATION_BOUND,
    COMPLETE_E_RELATIVE_ACCURACY,
    COMPLETE_K_RELATIVE_ACCURACY,
    EM_LENGTH_UNIT,
    GEOMETRY_LENGTH_UNIT,
    LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M,
    MAXWELL_BRACKET_RELATIVE_ACCURACY,
    MILLIMETRES_PER_METRE,
    MODULUS_MAX,
    VACUUM_PERMEABILITY_EVIDENCE_GRADE,
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMEABILITY_RELATIVE_UNCERTAINTY,
    CircularLoop,
    ElectromagneticsError,
    agm_kernel,
    coaxial_mutual_inductance_h,
    complete_elliptic_e,
    complete_elliptic_e_of_parameter,
    complete_elliptic_k,
    complete_elliptic_k_of_parameter,
    dipole_mutual_inductance_h,
    flux_linkage_wb,
    maxwell_mutual_bracket,
    metres_from_millimetres,
    millimetres_from_metres,
    mutual_inductance_h,
    require_em_length_unit,
    vacuum_permeability_relative_deviation_from_legacy,
)

# ---------------------------------------------------------------------------
# 公开面
# ---------------------------------------------------------------------------


def test_every_exported_name_exists():
    for name in electromagnetics.__all__:
        assert hasattr(electromagnetics, name), f"__all__列了不存在的名字{name!r}"
    assert electromagnetics.__all__ == sorted(electromagnetics.__all__)


def test_electromagnetics_names_stay_out_of_the_package_facade():
    """域的公开名不进包门面——基座不依赖物理域（spec/01第一节）。

    域隔离门③从import那一侧守着同一件事；这一条从**名字**那一侧守。
    两条都要，因为一个模块可以不import却把名字复制过去。
    """

    leaked = sorted(set(electromagnetics.__all__) & set(physics_engine.__all__))
    assert not leaked, f"电磁域的名字漏进了包门面：{leaked}"


# ---------------------------------------------------------------------------
# 椭圆积分：约定、精度、两个AGM坑
# ---------------------------------------------------------------------------


def test_the_two_elliptic_conventions_are_named_and_convert_correctly():
    """模k与参数m = k²两种约定：换算对，而且两者**确实不同**。"""

    for modulus in (0.1, 0.5, 0.9):
        assert complete_elliptic_k_of_parameter(modulus * modulus) == (
            complete_elliptic_k(modulus)
        )
        assert complete_elliptic_e_of_parameter(modulus * modulus) == (
            complete_elliptic_e(modulus)
        )


def test_the_convention_gate_is_red_when_the_two_are_confused():
    """**必须红**：把参数当模用差多少——实测约10%，不会离谱到一眼看出。"""

    confused = complete_elliptic_k_of_parameter(0.5)
    correct = complete_elliptic_k(0.5)
    relative = abs(confused - correct) / correct
    assert relative > 0.09, (
        f"K(k=0.5)与K(m=0.5)的差只有{relative!r}——若它小到看不出，"
        "本仓两种约定并存这件事就不再是风险，本条门可以撤"
    )
    assert relative < 0.11, f"预期约10%，实测{relative!r}——数变了要重新看约定"


def test_complete_elliptic_integrals_hit_the_reference_values():
    """A&S表17.1的两个点。**引用值当判据，不重推公式**。"""

    assert abs(complete_elliptic_k(0.5) - 1.6857503548125960) < 1e-15
    assert abs(complete_elliptic_e(0.5) - 1.4674622093394272) < 1e-15
    assert complete_elliptic_k(0.0) == math.pi / 2.0
    assert complete_elliptic_e(0.0) == math.pi / 2.0


def test_the_bracket_is_zero_at_zero_modulus_without_dividing():
    """`k = 0`时方括号恰为0（极限精确值），不走除法也不给nan。"""

    assert maxwell_mutual_bracket(0.0) == 0.0


def test_the_agm_tail_gate_is_red_on_a_non_terminating_loop():
    """**必须红**：AGM循环不终止会静默出错，这条把那个错的量级钉住。

    `cₙ`踩到浮点地板后停在一个ulp不动，而`2ⁿ`还在翻倍——继续加下去是把地板噪声
    放大成真误差。注入的是`tail`上的一个**绝对地板**约`2^N·ε²`，
    而`tail ≈ k⁴/8`，所以**危害与k成反比**：k=0.1处恰为0，k=1e-4处已达5.4e-4。

    本条在k=1e-4上复现那个写法（固定迭代40次、不判停滞），
    确认相对偏差**大于1e-5**。选1e-4这个模不是为了好看：
    它正是"远场退化"那一层要走的区间的延长线上。
    """

    def non_terminating_kernel(modulus: float) -> tuple[float, float]:
        complement = math.sqrt((1.0 - modulus) * (1.0 + modulus))
        a, b = 1.0, complement
        c = modulus * modulus / (2.0 * (1.0 + complement))
        tail = 0.0
        power = 2.0
        for _ in range(AGM_ITERATION_BOUND):
            if c <= 0.0:
                break
            tail += power * c * c
            a, b = 0.5 * (a + b), math.sqrt(a * b)
            c = 0.5 * (a - b)          # ← 少了`c >= previous`那一条停机判定
            power *= 2.0
        return math.pi / (2.0 * a), tail

    modulus = 1.0e-4
    correct_k, correct_tail = agm_kernel(modulus)
    broken_k, broken_tail = non_terminating_kernel(modulus)
    assert correct_k == broken_k, "K不受这个坑影响——受影响的只有tail"
    relative = abs(broken_tail - correct_tail) / correct_tail
    assert relative > 1.0e-5, (
        f"不终止的AGM只差{relative!r}——若它真的无害，本条门与elliptic.py"
        "第四节坑一的整段说明都该撤掉"
    )
    # 反面：k大时同一个坑**恰好无害**（`cₙ`落到0而不是停滞）。
    # 只断"它有害"会让人以为任何k都危险，那是把边界说宽了。
    assert non_terminating_kernel(0.5)[1] == agm_kernel(0.5)[1]


def test_the_small_modulus_cancellation_gate_is_red_on_the_naive_first_step():
    """**必须红**：`c₁ = (1 − √(1−k²))/2`在小k下相消，误差进`tail`再原样进M。"""

    modulus = 1.0e-4
    complement = math.sqrt((1.0 - modulus) * (1.0 + modulus))
    naive_c1 = 0.5 * (1.0 - complement)
    exact_c1 = modulus * modulus / (2.0 * (1.0 + complement))
    relative = abs(naive_c1 - exact_c1) / exact_c1
    assert relative > 1.0e-9, (
        f"两种写法只差{relative!r}——小k下的相消若不存在，elliptic.py第四节坑二"
        "那段说明就是错的"
    )


def test_the_textbook_grouping_gate_is_red_in_the_far_field():
    """**必须红**：`(2/k − k)K − (2/k)E`直接算在小k下相消放大约`1/k⁴`。

    这条同时说明**为什么远场退化那一层判据必须换算法**：照教科书分组写，
    误差在d还没拉开两个倍频程时就淹没掉待测的偏差本身。
    """

    for modulus, floor in ((1.0e-2, 1.0e-9), (1.0e-3, 1.0e-5), (1.0e-4, 1.0)):
        stable = maxwell_mutual_bracket(modulus)
        textbook = (2.0 / modulus - modulus) * complete_elliptic_k(modulus) - (
            2.0 / modulus
        ) * complete_elliptic_e(modulus)
        relative = abs(textbook - stable) / stable
        assert relative > floor, (
            f"k={modulus!r}处教科书分组只差{relative!r}（预期>{floor!r}）——"
            "相消若不存在，elliptic.py第三节整张表都要重测"
        )


def test_the_declared_accuracies_are_ordered_and_positive():
    """申报的三条精度必须是正数且量级合理（判据本身也要被验）。"""

    for accuracy in (
        COMPLETE_K_RELATIVE_ACCURACY,
        COMPLETE_E_RELATIVE_ACCURACY,
        MAXWELL_BRACKET_RELATIVE_ACCURACY,
    ):
        assert 1e-16 < accuracy < 1e-14


@pytest.mark.parametrize(
    "call",
    [
        lambda: complete_elliptic_k(-0.1),
        lambda: complete_elliptic_k(1.0),
        lambda: complete_elliptic_e(float("nan")),
        lambda: complete_elliptic_k(True),
        lambda: complete_elliptic_k_of_parameter(1.0),
        lambda: complete_elliptic_e_of_parameter(-1e-9),
        lambda: maxwell_mutual_bracket(float("inf")),
    ],
)
def test_elliptic_domain_violations_fail_closed(call):
    with pytest.raises(ElectromagneticsError):
        call()


def test_the_modulus_ceiling_refuses_rather_than_clamping():
    """`k > MODULUS_MAX`拒跑，**不夹边界也不返回nan**。"""

    with pytest.raises(ElectromagneticsError, match="对数发散"):
        complete_elliptic_k(1.0 - 1.0e-15)
    assert math.isfinite(complete_elliptic_k(MODULUS_MAX))


# ---------------------------------------------------------------------------
# 回路与互感
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: CircularLoop(radius_m=0.0, axial_position_m=0.0),
        lambda: CircularLoop(radius_m=-1.0, axial_position_m=0.0),
        lambda: CircularLoop(radius_m=float("inf"), axial_position_m=0.0),
        lambda: CircularLoop(radius_m=0.1, axial_position_m=float("nan")),
        lambda: CircularLoop(radius_m=0.1, axial_position_m=0.0, turns=0),
        lambda: CircularLoop(radius_m=0.1, axial_position_m=0.0, turns=1.5),
        lambda: CircularLoop(radius_m=0.1, axial_position_m=0.0, turns=True),
        lambda: CircularLoop(radius_m=0.1, axial_position_m=0.0, current_a=float("inf")),
    ],
)
def test_loop_declaration_violations_fail_closed(call):
    with pytest.raises(ElectromagneticsError):
        call()


def test_coincident_filamentary_loops_fail_closed():
    """两条丝状回路重合时互感对数发散——拒跑，不返回inf也不夹边界。"""

    with pytest.raises(ElectromagneticsError, match="对数发散"):
        coaxial_mutual_inductance_h(
            radius_a_m=0.05, radius_b_m=0.05, axial_separation_m=0.0
        )


def test_coplanar_concentric_loops_are_legal():
    """`d = 0`本身**不是**退化：不同半径的共面同心回路互感有限。"""

    value = coaxial_mutual_inductance_h(
        radius_a_m=0.100, radius_b_m=0.010, axial_separation_m=0.0
    )
    assert math.isfinite(value) and value > 0.0


def test_mutual_inductance_falls_off_monotonically_with_separation():
    """结构判据：互感随轴向间距单调下降。清单抓不到这一类形状错。"""

    previous = float("inf")
    for step in range(1, 40):
        value = coaxial_mutual_inductance_h(
            radius_a_m=0.05, radius_b_m=0.05, axial_separation_m=0.002 * step
        )
        assert value < previous, f"互感在d={0.002 * step!r}处不再下降"
        previous = value


def test_the_separation_is_signed_free():
    """互感与"谁在上谁在下"无关：轴向间距取绝对值。"""

    above = CircularLoop(radius_m=0.05, axial_position_m=0.02)
    below = CircularLoop(radius_m=0.05, axial_position_m=-0.02)
    origin = CircularLoop(radius_m=0.04, axial_position_m=0.0)
    assert mutual_inductance_h(origin, above) == mutual_inductance_h(origin, below)


def test_flux_linkage_is_linear_in_the_source_current():
    """`λ = M·I`：I=0恰为0，且不再乘一次匝数。"""

    target = CircularLoop(radius_m=0.05, axial_position_m=0.02, turns=3)
    quiet = CircularLoop(radius_m=0.05, axial_position_m=0.0, turns=2, current_a=0.0)
    assert flux_linkage_wb(source=quiet, target=target) == 0.0
    live = CircularLoop(radius_m=0.05, axial_position_m=0.0, turns=2, current_a=2.0)
    assert flux_linkage_wb(source=live, target=target) == (
        2.0 * mutual_inductance_h(live, target)
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: coaxial_mutual_inductance_h(
            radius_a_m=0.0, radius_b_m=0.05, axial_separation_m=0.01
        ),
        lambda: coaxial_mutual_inductance_h(
            radius_a_m=0.05, radius_b_m=0.05, axial_separation_m=-0.01
        ),
        lambda: dipole_mutual_inductance_h(
            radius_a_m=0.05, radius_b_m=0.05, axial_separation_m=0.0
        ),
        lambda: mutual_inductance_h(CircularLoop(radius_m=0.05, axial_position_m=0.0), 3),
    ],
)
def test_inductance_domain_violations_fail_closed(call):
    with pytest.raises(ElectromagneticsError):
        call()


# ---------------------------------------------------------------------------
# 单位边界与μ₀的证据分级
# ---------------------------------------------------------------------------


def test_the_domain_is_metric_and_says_so():
    assert EM_LENGTH_UNIT == "m"
    assert GEOMETRY_LENGTH_UNIT == "mm"
    assert require_em_length_unit("m") == "m"
    with pytest.raises(ElectromagneticsError, match="metres_from_millimetres"):
        require_em_length_unit("mm")


def test_the_length_conversion_is_reversible_and_uses_one_factor():
    assert MILLIMETRES_PER_METRE == 1.0e3
    for value in (50.0, 20.0, 0.1, 1234.5678):
        assert millimetres_from_metres(metres_from_millimetres(value)) == value


def test_passing_millimetres_as_metres_inflates_the_inductance_by_a_thousand():
    """传错单位**不会报任何错**，只会大一千倍——所以往返判据是唯一的门。"""

    metric = coaxial_mutual_inductance_h(
        radius_a_m=0.05, radius_b_m=0.05, axial_separation_m=0.02
    )
    mistaken = coaxial_mutual_inductance_h(
        radius_a_m=50.0, radius_b_m=50.0, axial_separation_m=20.0
    )
    assert abs(mistaken / metric - 1.0e3) < 1.0e-9


def test_mu0_is_declared_measured_not_a_defined_constant():
    """2019 SI重定义之后μ₀不再按定义精确——分级不得是`benchmark_constant`。

    research/08第4.4节的结论在这里成为一条门。
    """

    assert VACUUM_PERMEABILITY_EVIDENCE_GRADE == "measured"
    assert VACUUM_PERMEABILITY_EVIDENCE_GRADE != "benchmark_constant"
    deviation = vacuum_permeability_relative_deviation_from_legacy()
    assert 0.0 < deviation < VACUUM_PERMEABILITY_RELATIVE_UNCERTAINTY, (
        f"新旧μ₀的相对偏差{deviation!r}应当落在CODATA不确定度"
        f"{VACUUM_PERMEABILITY_RELATIVE_UNCERTAINTY!r}之内——"
        "落在外面说明抄错了一个数字"
    )
    assert VACUUM_PERMEABILITY_H_PER_M != LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M


@pytest.mark.parametrize(
    "call",
    [
        lambda: metres_from_millimetres(float("nan")),
        lambda: millimetres_from_metres(float("inf")),
        lambda: metres_from_millimetres("50"),
        lambda: metres_from_millimetres(True),
    ],
)
def test_unit_conversion_violations_fail_closed(call):
    with pytest.raises(ElectromagneticsError):
        call()
