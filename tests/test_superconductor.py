"""`electromagnetics/superconductor.py`的模块级门：申报精度、单位边界、缺口登记。

案例`cases/norris_thin_strip`守的是**判据**（对50位参考、对电流守恒恒等式）；
本文件守的是三件案例守不到的事：

1. **模块自己申报的精度是不是真的**（`SHEET_CURRENT_RELATIVE_ACCURACY`、
   `NORRIS_LOSS_RELATIVE_ACCURACY`）——申报值只在案例的判据表里被引用，
   没人验它就成了自说自话；
2. **单位边界**：公开参数名带不带单位后缀、`per_`那个方向反了的坑踩没踩上、
   两个长度制之间的换算因子与`materials`的换算表对不对得上；
3. **两条缺口的绊线**：材料记录今天装不下`Ic`（决策0047第四节），
   以及"幂律指数典型20—30"这个**未核实**的数不许脱离标记被转述
   （research/09价格条那次事故的形制）。

第3条的两条门都是"**缺口还在**"的断言：缺口被补上的那天它们变红，
红了就回去更新决策0047。这是本仓对付"规范里写了但没人实现"的办法——
把缺口变成会响的绊线，而不是一句文档里的话。
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import pytest

from physics_engine.electromagnetics import superconductor
from physics_engine.electromagnetics.superconductor import (
    LOSS_SERIES_LIMIT,
    NORRIS_LOSS_RELATIVE_ACCURACY,
    SHEET_CURRENT_RELATIVE_ACCURACY,
    SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES,
    SUPERCONDUCTOR_LENGTH_UNIT,
    VACUUM_PERMEABILITY_H_PER_M,
    flux_free_half_width_m,
    norris_ellipse_normalised_loss,
    norris_strip_loss_j_per_m_per_cycle,
    norris_strip_normalised_loss,
    sheet_critical_current_a_per_m,
    sheet_current_density_a_per_m,
    strip_critical_current_a,
)
from physics_engine.identity import (
    BASE_UNIT_SUFFIXES,
    IdentityError,
    assert_quantity_fields_have_units,
)
from physics_engine.materials import (
    EvidenceRef,
    MaterialError,
    MaterialProperty,
    MaterialRecord,
    unit_suffix_of,
)

ROOT = Path(__file__).resolve().parents[1]

WIDTH_M = 4.0e-3
SHEET_A_PER_M = 3.0e4
CRITICAL_A = WIDTH_M * SHEET_A_PER_M


# ---------------------------------------------------------------------------
# 一、申报精度是不是真的
# ---------------------------------------------------------------------------
def equivalent_sheet_current(
    *, position_m: float, width_m: float, transport_current_a: float,
    critical_sheet_current_a_per_m: float,
) -> float:
    """`K(x)`的**代数等价第三形**：``(2Kc/π)·arcsin(sqrt((a²−b²)/(a²−x²)))``。

    由`arctan(sqrt(A/B)) = arcsin(sqrt(A/(A+B)))`得到，`A+B = a²−x²`。
    它与被验实现既不共用超越函数（arcsin对arctan）也不共用分母，
    因此两者一致到eps量级这件事是**申报精度的证据**而不是同义反复。
    """

    half_width = 0.5 * width_m
    critical = width_m * critical_sheet_current_a_per_m
    ratio = transport_current_a / critical
    front = half_width * math.sqrt((1.0 - ratio) * (1.0 + ratio))
    distance = abs(position_m)
    if distance >= front:
        return critical_sheet_current_a_per_m
    numerator = (half_width - front) * (half_width + front)
    denominator = (half_width - distance) * (half_width + distance)
    return (2.0 * critical_sheet_current_a_per_m / math.pi) * math.asin(
        math.sqrt(numerator / denominator)
    )


def test_the_declared_sheet_current_accuracy_holds_on_the_declared_domain():
    """申报的`SHEET_CURRENT_RELATIVE_ACCURACY`在`|x| ≤ 0.999·b`上成立。

    域是申报的一部分：`|x|/b → 1`时闭式本身病态（见下一条），
    所以精度只在这个域里申报，域外**写作未测**而不是外推。

    **这条门是两条float64路径的互校，不是权威精度测量**，如实说明：
    `arcsin`在自变量趋近1时自己也病态，所以实测的9900点最坏3.06e-13里
    有相当一部分是**参考路径**的误差。对50位算术的权威测量在案例那边
    （`oracle:norris/sheet_current_profile`，49点最坏1.45e-14；
    本轮定标时在同一申报域上全域测到的最坏是4.48e-14）。
    两个数都在申报的1e-12以内，而本门的价值在于它**每次运行都在跑**，
    不必依赖生成器。
    """

    worst = 0.0
    for index in range(1, 100):
        ratio = index / 100.0
        current = ratio * CRITICAL_A
        front = flux_free_half_width_m(
            width_m=WIDTH_M, transport_current_a=current, critical_current_a=CRITICAL_A
        )
        for step in range(0, 100):
            position = front * (0.999 * step / 99.0)
            got = sheet_current_density_a_per_m(
                position_m=position, width_m=WIDTH_M, transport_current_a=current,
                critical_sheet_current_a_per_m=SHEET_A_PER_M,
            )
            want = equivalent_sheet_current(
                position_m=position, width_m=WIDTH_M, transport_current_a=current,
                critical_sheet_current_a_per_m=SHEET_A_PER_M,
            )
            worst = max(worst, abs(got - want) / abs(want))
    assert worst <= SHEET_CURRENT_RELATIVE_ACCURACY, (
        f"两条代数等价路径的最坏相对差{worst!r}超过申报精度"
        f"{SHEET_CURRENT_RELATIVE_ACCURACY!r}——申报是假的"
    )
    assert worst > SHEET_CURRENT_RELATIVE_ACCURACY / 1.0e4, (
        f"最坏相对差只有{worst!r}，比申报精度小四个数量级以上——"
        "申报值松到没有判别力了，收紧它（本仓不接受'反正过得去'的申报）"
    )


def test_the_closed_form_is_ill_conditioned_at_the_flux_front():
    """**已知失效登记**：`|x|/b → 1`时相对精度掉到1e-9量级，且**换写法也救不回来**。

    两条代数等价路径在`|x|/b = 1 − 1e-9`处一起掉到同一个量级——
    这正是"病态不是实现缺陷"的证据：算法换了、误差没变。
    若哪天有人把它压下去，这条门会红，那时把案例页第四节那一行删掉。
    """

    current = 0.001 * CRITICAL_A
    front = flux_free_half_width_m(
        width_m=WIDTH_M, transport_current_a=current, critical_current_a=CRITICAL_A
    )
    position = front * (1.0 - 1.0e-9)
    got = sheet_current_density_a_per_m(
        position_m=position, width_m=WIDTH_M, transport_current_a=current,
        critical_sheet_current_a_per_m=SHEET_A_PER_M,
    )
    other = equivalent_sheet_current(
        position_m=position, width_m=WIDTH_M, transport_current_a=current,
        critical_sheet_current_a_per_m=SHEET_A_PER_M,
    )
    deviation = abs(got - other) / abs(other)
    assert deviation > SHEET_CURRENT_RELATIVE_ACCURACY, (
        f"磁通前沿附近两条路径只差{deviation!r}——比申报精度还好，"
        "案例页第四节登记的病态不成立了，去删掉那一行"
    )


def logarithmic_slope(function, point: float) -> float:
    """``dlnF/dlni``在`point`处的中心差分估计，**两个取值点都在同一支上**。

    容差自己算出来，不拍一个"看起来够大"的常数：跨切换点的跳变
    只被允许等于函数本身在这段上的斜率乘以跨度。
    """

    span = 0.01 * point
    low, high = point - span, point + span
    return (
        math.log(function(high) / function(low)) / math.log(high / low)
    )


def test_the_two_loss_branches_agree_across_the_series_switch():
    """级数支与`log1p`支在切换点两侧必须接得上，接缝**只许跳斜率×跨度那么多**。

    分段实现最典型的错是接缝处跳一下：两边各自都"对"，合起来不连续。
    实测两条式在切换点的对数斜率分别是4.60与3.76，
    跨度`2·offset/i`乘上它就是允许的跳变，多一点都不行（另留2%余量）。
    """

    for function in (norris_strip_normalised_loss, norris_ellipse_normalised_loss):
        # 斜率取在**级数支内部**（0.99·切换点），不跨过切换点自己去估自己。
        slope = logarithmic_slope(function, 0.99 * LOSS_SERIES_LIMIT)
        for offset in (1.0e-12, 1.0e-10, 1.0e-8):
            below = LOSS_SERIES_LIMIT - offset
            above = LOSS_SERIES_LIMIT + offset
            low, high = function(below), function(above)
            jump = abs(high - low) / abs(low)
            allowed = (
                1.02 * abs(slope) * (above - below) / LOSS_SERIES_LIMIT
                + NORRIS_LOSS_RELATIVE_ACCURACY
            )
            assert jump <= allowed, (
                f"{function.__name__}在切换点两侧跳了{jump!r}，"
                f"而按对数斜率{slope!r}只该跳{allowed!r}——分段接缝没接上"
            )


def test_the_loss_endpoints_are_the_textbook_closed_values():
    """`i = 1`两端点：薄带`2(2ln2 − 1)`、椭圆恰为1；`i = 0`两式都恰为0。"""

    assert norris_strip_normalised_loss(1.0) == 2.0 * (2.0 * math.log(2.0) - 1.0)
    assert norris_ellipse_normalised_loss(1.0) == 1.0
    assert norris_strip_normalised_loss(0.0) == 0.0
    assert norris_ellipse_normalised_loss(0.0) == 0.0


# ---------------------------------------------------------------------------
# 二、单位边界
# ---------------------------------------------------------------------------
#: 公开函数的**无量纲**参数（轴2规则5：无量纲必须显式列出，留空装有是禁止的形状）。
DIMENSIONLESS_PARAMETERS: frozenset[str] = frozenset({"current_ratio_value"})


def public_functions() -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, getattr(superconductor, name))
        for name in superconductor.__all__
        if inspect.isfunction(getattr(superconductor, name))
    )


def public_parameter_names() -> tuple[str, ...]:
    """本模块全部公开函数的参数名（去重、排序）。"""

    names: set[str] = set()
    for _, member in public_functions():
        names.update(inspect.signature(member).parameters)
    return tuple(sorted(names))


def test_every_public_parameter_carries_a_unit_suffix_or_is_declared_dimensionless():
    """量纲门：公开签名逐个参数过轴2规则3。**新函数带裸参数名即红**。

    用`inspect`读真实签名而不是抄一张名单——名单会随新函数进仓慢慢空掉，
    形制同域隔离门第四条（完备性）。
    """

    assert_quantity_fields_have_units(
        public_parameter_names(),
        dimensionless=DIMENSIONLESS_PARAMETERS,
        extra_units=SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES,
    )


def test_every_public_function_name_declares_the_unit_of_what_it_returns():
    """量纲门的另一半：**返回值的单位在函数名里**，无量纲的必须显式登记。

    Python没有返回值单位的声明位，本仓的办法是把它写进名字
    （`flux_free_half_width_m`、`norris_strip_loss_j_per_m_per_cycle`）。
    这条门把那条约定变成会红的东西——**光靠约定半年后就会有一个裸名字混进来**。
    """

    assert_quantity_fields_have_units(
        tuple(name for name, _ in public_functions()),
        dimensionless=superconductor.SUPERCONDUCTOR_DIMENSIONLESS_RESULTS,
        extra_units=SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES,
    )
    stray = superconductor.SUPERCONDUCTOR_DIMENSIONLESS_RESULTS - {
        name for name, _ in public_functions()
    }
    assert not stray, f"无量纲登记表里有本模块没有的函数：{sorted(stray)}"


@pytest.mark.parametrize(
    ("names", "dimensionless"),
    [
        ((("peak_field",)), DIMENSIONLESS_PARAMETERS),
        ((("norris_strip_loss",)), superconductor.SUPERCONDUCTOR_DIMENSIONLESS_RESULTS),
    ],
)
def test_must_be_red_a_naked_name_fails_the_dimension_gate(names, dimensionless):
    """必须红：没有单位后缀又没声明无量纲的名字当场炸——参数与函数名两侧都验。"""

    with pytest.raises(IdentityError):
        assert_quantity_fields_have_units(
            names, dimensionless=dimensionless,
            extra_units=SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES,
        )


def test_the_ampere_is_not_in_the_frozen_base_suffix_set():
    """本域为什么必须自带补充集：**轴2的基础集里一个电学单位都没有**。

    这不是抱怨，是把"补充集不是随手加的"钉住：基础集真的缺，
    而改基础集是spec/14的面（要走决策记录，且是并行波次的共享冲突面），
    所以本域按`identity.has_unit_suffix`明写的口子传补充集。

    `k`（开尔文）与`w`（瓦特）在基础集里，`a`（安培）不在——
    **这不是遗漏的随机分布，是热域先于电磁域被想到**。
    """

    for unit in ("a", "v", "ohm", "wb", "per_m", "per_m2", "per_cycle"):
        assert unit not in BASE_UNIT_SUFFIXES, (
            f"基础集里已经有{unit!r}了——本域的补充集该相应缩小"
        )
    assert SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES.isdisjoint(BASE_UNIT_SUFFIXES), (
        "补充集与基础集有重叠——重叠的那几个该从补充集里删掉"
    )


def test_the_per_trap_does_not_bite_this_domains_quantities():
    """`per_`那个方向反了的坑（research/07审计发现的1e12）在本域的量上踩没踩上。

    坑的形状：单纯的最长匹配会把`..._a_per_m2`的后缀判成**面积**单位`m2`，
    于是换算走面积的`×1e6`，而`A/m² → A/mm²`是`×1e-6`——方向反了、差1e12、不报错。
    修法是"最长匹配不许跨过`per_`"。本域的量正落在这条边界上，逐条实测。
    """

    assert unit_suffix_of("critical_sheet_current_a_per_m") == "per_m"
    assert unit_suffix_of("critical_current_density_a_per_m2") == "per_m2"
    assert unit_suffix_of("sheet_current_density_a_per_mm") == "per_mm"
    # 反例：不带`per_`的同族名字才该判成裸长度单位。
    assert unit_suffix_of("superconducting_layer_thickness_m") == "m"
    assert unit_suffix_of("tape_cross_section_mm2") == "mm2"


def test_the_metre_to_millimetre_factor_agrees_with_the_shared_conversion_table():
    """本域声明的`A/m → A/mm`因子必须**等于`materials`换算表给的那个**。

    不自己写1e-3，而是让共享换算表把它算出来：两处各写一个常数，
    半年后一定有一处漂。这条门把它们钉在一起。
    """

    evidence = EvidenceRef(
        grade="estimated", evidence_id="evidence/gate_only", method="量纲门的占位证据，不是物性"
    )
    record = MaterialRecord(
        material_id="material/dimension_gate_probe",
        applicable_domains=("em",),
        length_unit=SUPERCONDUCTOR_LENGTH_UNIT,
        properties=(
            MaterialProperty(
                name="critical_sheet_current_a_per_m", value=SHEET_A_PER_M,
                domains=("em",), evidence=evidence,
            ),
        ),
    )
    converted = record.converted_to("mm")
    (field,) = converted.properties
    assert field.name == "critical_sheet_current_a_per_mm"
    assert field.value == SHEET_A_PER_M * 1.0e-3, (
        "共享换算表给的A/m→A/mm因子不是1e-3——本域的量纲账要重算"
    )


def test_length_scaling_is_bit_exact_under_a_power_of_two():
    """量纲齐次性的**零容差**门：长度按2的幂缩放时`b`与`K`逐位地跟着缩放。

    取2的幂而不是1000，是为了让"逐位"真的成立：乘以2的幂在浮点里精确，
    于是`a²−b²`、`(b−x)(b+x)`、它们的商、arctan全都逐位不变，
    `K`只随`Kc`按`2^{-k}`精确缩放。**任何一处漏乘或多乘一次比例都会当场红**，
    而这类错在1e-15容差下是看不见的。
    """

    scale = 1024.0
    current = 0.37 * CRITICAL_A
    for step in range(0, 33):
        position = 0.5 * WIDTH_M * step / 32.0
        base = sheet_current_density_a_per_m(
            position_m=position, width_m=WIDTH_M, transport_current_a=current,
            critical_sheet_current_a_per_m=SHEET_A_PER_M,
        )
        scaled = sheet_current_density_a_per_m(
            position_m=position * scale, width_m=WIDTH_M * scale,
            transport_current_a=current,
            critical_sheet_current_a_per_m=SHEET_A_PER_M / scale,
        )
        assert scaled == base / scale, (
            f"x={position!r} m处K没有逐位随长度制缩放：{base / scale!r} 对 {scaled!r}"
        )
    base_front = flux_free_half_width_m(
        width_m=WIDTH_M, transport_current_a=current, critical_current_a=CRITICAL_A
    )
    scaled_front = flux_free_half_width_m(
        width_m=WIDTH_M * scale, transport_current_a=current, critical_current_a=CRITICAL_A
    )
    assert scaled_front == base_front * scale
    assert strip_critical_current_a(
        width_m=WIDTH_M * scale, critical_sheet_current_a_per_m=SHEET_A_PER_M / scale
    ) == CRITICAL_A


def test_the_millimetre_reading_is_the_metre_reading_over_a_thousand():
    """mm制读数与m制读数的比必须是1e-3，容差只留浮点取整（1000不是2的幂）。

    与上一条分工：上一条零容差验**齐次性**，这一条验的是**真的那个1000**。
    """

    current = 0.62 * CRITICAL_A
    worst = 0.0
    for step in range(1, 33):
        position = 0.5 * WIDTH_M * step / 32.0
        metre = sheet_current_density_a_per_m(
            position_m=position, width_m=WIDTH_M, transport_current_a=current,
            critical_sheet_current_a_per_m=SHEET_A_PER_M,
        )
        millimetre = sheet_current_density_a_per_m(
            position_m=position * 1.0e3, width_m=WIDTH_M * 1.0e3,
            transport_current_a=current,
            critical_sheet_current_a_per_m=SHEET_A_PER_M * 1.0e-3,
        )
        worst = max(worst, abs(millimetre - metre * 1.0e-3) / abs(millimetre))
    assert worst < 1.0e-15, f"mm制与m制读数之比不是1e-3，最坏相对差{worst!r}"


def test_the_loss_scales_as_the_square_of_the_critical_current():
    """量纲门之三：同一个`i`下损耗按`Ic²`缩放，**2的幂上逐位精确**。

    它抓的是量纲包装里`Ic`的**幂次**写错——写成`Ic`或`Ic³`时
    归一化量一个字不变、逐点金标照样绿，只有这条会红。
    """

    ratio = 0.4
    base = norris_strip_loss_j_per_m_per_cycle(
        critical_current_a=CRITICAL_A, current_amplitude_a=ratio * CRITICAL_A
    )
    doubled = norris_strip_loss_j_per_m_per_cycle(
        critical_current_a=2.0 * CRITICAL_A, current_amplitude_a=ratio * 2.0 * CRITICAL_A
    )
    assert doubled == 4.0 * base


def test_the_permeability_is_the_pre_2019_defined_value():
    """`μ0`取2019年SI重新定义**之前**的定义值`4π×10⁻⁷`，与CODATA实测值差约1.3e-10。

    这条门不是"哪个值更对"，是**不许它被悄悄换掉**：
    案例的损耗容差是1e-14，换成CODATA值案例会红——那时该走决策记录，
    而不是顺手把容差放宽到1e-9。
    """

    assert VACUUM_PERMEABILITY_H_PER_M == 4.0e-7 * math.pi
    codata_2022 = 1.25663706127e-6
    deviation = abs(VACUUM_PERMEABILITY_H_PER_M - codata_2022) / codata_2022
    assert 1.0e-11 < deviation < 1.0e-9, (
        f"定义值与CODATA实测值的差已经不是1e-10量级了（{deviation!r}）——"
        "这条门的前提变了，回去看案例页第二节"
    )


def test_the_thin_strip_approximation_is_where_the_thickness_goes():
    """`Kc = Jc·d`：厚度在这一步被吸收进片电流密度，此后闭式里只剩`Kc`。

    传体电流密度（A/m²）当片电流密度（A/m）用会差一个`d`——
    微米量级即百万倍，而且不报任何错。这条门锁的是那道换算真的在。
    """

    thickness = 1.0e-6
    density = 3.0e10
    assert sheet_critical_current_a_per_m(
        critical_current_density_a_per_m2=density, layer_thickness_m=thickness
    ) == density * thickness
    assert strip_critical_current_a(
        width_m=WIDTH_M, critical_sheet_current_a_per_m=SHEET_A_PER_M
    ) == WIDTH_M * SHEET_A_PER_M


# ---------------------------------------------------------------------------
# 三、两条缺口的绊线
# ---------------------------------------------------------------------------
def _probe_property(name: str, value: float) -> MaterialProperty:
    return MaterialProperty(
        name=name, value=value, domains=("em",),
        evidence=EvidenceRef(
            grade="estimated", evidence_id="evidence/gap_tripwire",
            method="缺口绊线的占位证据，不是物性",
        ),
    )


def test_the_material_record_still_cannot_carry_a_critical_current():
    """**缺口绊线（决策0047第四节）**：`Ic`今天进不了材料记录。

    `Jc`与`Ic`是材料属性不是物理常数，按spec/14规则1本该与杨氏模量同处一条记录。
    实测拒收点：`identity.BASE_UNIT_SUFFIXES`里没有安培，
    `critical_current_a`落进"没有单位后缀又没声明无量纲"那条。

    **这条门在缺口被补上的那天变红。** 红了不是坏事：回去更新决策0047第四节，
    把"临时避开"改成"已长期解决"，并把本案例的`Ic`改成从记录里读。
    """

    with pytest.raises(MaterialError) as failure:
        MaterialRecord(
            material_id="material/gap_tripwire_rebco",
            applicable_domains=("em",),
            length_unit="mm",
            properties=(_probe_property("critical_current_a", 120.0),),
        )
    assert "critical_current_a" in str(failure.value)
    assert "unit suffix" in str(failure.value)


def test_the_length_system_check_still_lets_an_area_denominator_through():
    """**缺口绊线之二**：`..._a_per_m2`（米制）能悄悄进一条mm制记录。

    `materials._SYSTEM_SUFFIXES['m']`只枚举了`m`/`m2`/`m3`/`per_m`，
    **没有`per_m2`**，于是长度制隔离检查放行；而同一个物理量的片版本
    `..._a_per_m`（`per_m`）会被当场拒收。**同一族量两种判决，
    差别只在于哪个后缀恰好被枚举过**——这正是spec/14第五节要防的那类静默。

    本案例不踩这个坑（`Jc`根本没进记录），但缺口是实测出来的，必须留绊线。
    补上枚举的那天这条门变红，回去更新决策0047第四节。
    """

    accepted = MaterialRecord(
        material_id="material/gap_tripwire_area_denominator",
        applicable_domains=("em",),
        length_unit="mm",
        properties=(_probe_property("critical_current_density_a_per_m2", 3.0e10),),
    )
    assert accepted.length_unit == "mm"
    with pytest.raises(MaterialError) as failure:
        MaterialRecord(
            material_id="material/gap_tripwire_line_denominator",
            applicable_domains=("em",),
            length_unit="mm",
            properties=(_probe_property("critical_sheet_current_a_per_m", 3.0e4),),
        )
    assert "mixing length systems" in str(failure.value)


#: 幂律指数`n`"典型20—30"这一说的**未核实**标记必须寸步不离。
#: research/09那次事故的形制：一个未核实的数脱离标记被转述，就变成了事实。
_N_RANGE = re.compile(r"20\s*[—–-]\s*30")
_MARKER = "未核实"
_MARKER_WINDOW = 500

_GUARDED_PATHS: tuple[str, ...] = (
    "src/physics_engine/electromagnetics/superconductor.py",
    "src/physics_engine/electromagnetics/__init__.py",
    "cases/norris_thin_strip/case.md",
    "cases/norris_thin_strip/generate_oracle.py",
    "tests/cases/test_norris_thin_strip.py",
    "docs/decisions/0047_Norris薄带解析基准_20260805.md",
)


def test_the_unverified_n_value_never_appears_without_its_marker():
    """本轨道产出的每一处"20—30"附近都必须带《未核实》。

    这条门守的是决策0047第五节点名的三条边界之二。它按**字符窗**判而不是按段落，
    因为段落在代码注释与Markdown里不是同一个东西。
    """

    for relative in _GUARDED_PATHS:
        path = ROOT / relative
        assert path.is_file(), f"被守的文件不见了：{relative}"
        text = path.read_text(encoding="utf-8")
        for match in _N_RANGE.finditer(text):
            window = text[
                max(0, match.start() - _MARKER_WINDOW) : match.end() + _MARKER_WINDOW
            ]
            assert _MARKER in window, (
                f"{relative}第{text[: match.start()].count(chr(10)) + 1}行的"
                f"『{match.group()}』附近没有《{_MARKER}》标记——"
                "research/09那次事故就是这么发生的"
            )
