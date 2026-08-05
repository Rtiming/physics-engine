"""Norris 1970薄带的临界态闭式：片电流分布与两条交流损耗式。

一句话：宽`w`、厚`d`的**无限薄超导带**通传输电流`I`时，临界态模型给出的
片电流密度`K(x)`有conformal mapping闭式，磁滞损耗`Ql`也有闭式；
薄带与椭圆截面的小幅值渐近分别是`i^4/3`与`i^3/3`（`i = I/Ic`），
**三次方与四次方的差别就是这两条判据的招牌**。

## 参考解出处（本轮逐条核实过，不是抄二手）

* **片电流分布**：W. T. Norris, *J. Phys. D: Appl. Phys.* **3** (1970) 489。
  本仓采信的写法取自Pardo等《Electromagnetic modelling of superconductors
  with a smooth current-voltage relation》（arXiv:1410.0772）——
  research/12把它记作"式(69)"，**本轮公开检索到的引用给的是式(67)**，
  节号存疑而形式一致；下面的形式是按两个零容差极限反推确认过的（见`FLUX_FRONT`一段）。
* **交流损耗两式**：Grilli等《Computation of Losses in HTS Under the Action of
  Varying Magnetic Fields and Currents》（arXiv:1306.6251）式46、47。
  本轮独立核到的等价写法（单位长度损耗功率）是
  `P/l = f·μ0·Ic²·g(i)/π`，`g(i) = (1−i)ln(1−i) + (1+i)ln(1+i) − i²`（薄带）与
  `P/l = f·μ0·Ic²·e(i)/(2π)`，`e(i) = (2−i)·i + 2(1−i)ln(1−i)`（椭圆截面）。
  除以频率`f`得每周期损耗`Ql`，即本模块归一化量
  `2π·Ql/(μ0·Ic²)`分别等于`2·g(i)`与`e(i)`——**与任务书给的形式逐字一致**。

## 临界态（Bean）模型的适用边界——**必须先读**

本模块算的是**临界态极限**（幂律指数`n → ∞`），它假设：

1. `Jc`与局部磁场无关（真实REBCO的`Jc(B, θ)`随场显著下降，各向异性可达数倍）；
2. 忽略磁通蠕动（有限`n`、有限频率下损耗与分布都会偏离）；
3. 带材**无限薄**（`d ≪ w`）且**无限长**、无邻近导体、无外加磁场；
4. 单调加载的**传输电流**，没有外场分量、没有历史回溯。

第3条正是本闭式与"三维带厚度"之间的张力：同行绕开HTS带材长细比的主流手段
（T-A薄带、H-φ薄壳）就是**把厚度降维掉不画进网格**，与"我们是带厚度的三维"
直接相抵（research/12第3.1节）。

**幂律指数`n`的"典型20—30"这一说，本仓标《未核实》**：research/12核不到权威出处，
核到的具体数字是Pardo等取`n = 1000`模拟临界态、扫描`n = 5…200`、以及一份实测`n = 15`。
本模块**不使用`n`**（临界态是`n → ∞`的极限），此处出现它只为把这条标记留在使用点上。

## 单位边界（本模块存在的另一半理由）

本仓力学是mm制（`node0_x_mm`），电磁量天然米制。本域按**米**
（`SUPERCONDUCTOR_LENGTH_UNIT`），每个公开参数名自带单位后缀，
`K`是`a_per_m`、损耗是`j_per_m_per_cycle`。

**`per_`那个坑**：`materials.unit_suffix_of`曾把`current_density_a_per_m2`
的后缀判成面积单位`m2`，于是换算按面积走`×1e6`，而`A/m² → A/mm²`是`×1e-6`——
**方向反了、差1e12、且不报错**（2026-08-05修，research/07审计发现）。
本模块的量正落在这条边界上，因此逐条实测过（`tests/test_superconductor.py`
的量纲门）：`..._a_per_m → 'per_m'`、`..._a_per_m2 → 'per_m2'`，
两个都**没有**退化成裸`m`/`m2`，坑没踩上。

**安培与材料记录的缺口**：`Jc`与`Ic`是材料属性不是物理常数，本该进材料记录，
但`identity.BASE_UNIT_SUFFIXES`里**没有任何电学单位**（无A、V、Ω、T、H、Wb、S），
`critical_current_a`当场被拒。缺口的完整实测与裁决请求写在
`docs/decisions/0047_Norris薄带解析基准_20260805.md`第四节。
**本模块因此把`Kc`/`Ic`收成显式函数参数，不自建平行的电磁材料通道。**

## 数值形态（0016甲案：纯Python、零运行时依赖）

两处刻意的写法，都是实测出来的，不是风格：

* `K`的arctan自变量写成`(a·i)/sqrt((b−|x|)(b+|x|))`而不是
  `sqrt((a²−b²)/(b²−x²))`。因为`a²−b² = (a·i)²`**恒等**，直接用它就没有相消；
  而`b²−x²`按`(b−|x|)(b+|x|)`算时，`b−|x|`在`|x|`接近`b`时由Sterbenz引理**精确相减**。
  实测样点最坏相对误差从1.07e-12降到**4.48e-14**（24倍）；
* 损耗两式在`i ≪ 1`时是**对数相消**（结果`~i^4/3`，中间项`~i`，
  相消因子达`6/i^3`量级）。`i ≤ LOSS_SERIES_LIMIT`走**无相消的幂级数**，
  以上走`log1p`。实测最坏相对误差从`log(1±i)`写法的6.6e-4（`i=1e-3`处，
  **research/12记的"3.331e-13"正是这个数值噪声，不是物理**）降到**2.96e-15**。
"""

from __future__ import annotations

import math

#: 本域的长度制。电磁量天然米制（μ0是H/m，`K`是A/m），
#: 与力学的mm制**不混record**——见模块docstring的单位边界一段。
SUPERCONDUCTOR_LENGTH_UNIT: str = "m"

#: 本模块公开名（参数名与函数名）用到、而`identity.BASE_UNIT_SUFFIXES`里
#: **没有**的后缀。轴2规则3的基础集不含任何电学单位（无A、V、Ω、T、H、Wb、S），
#: 也不含`per_m`/`per_m2`/`per_cycle`；`identity.has_unit_suffix`明写调用方
#: 可传补充集，本域就在这里显式补，**不去改那个冻结的基础集**
#: （那是spec/14的面，改它要走决策记录，且是并行波次的共享冲突面）。
SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES: frozenset[str] = frozenset(
    {"a", "per_m", "per_m2", "per_cycle"}
)

#: 返回无量纲量的公开函数——轴2规则5：无量纲必须**显式**列出，留空装有是禁止的形状。
SUPERCONDUCTOR_DIMENSIONLESS_RESULTS: frozenset[str] = frozenset(
    {"current_ratio", "norris_strip_normalised_loss", "norris_ellipse_normalised_loss"}
)

#: 真空磁导率，取**2019年SI重新定义之前的定义值**`4π×10⁻⁷ H/m`。
#: CODATA的实测值与它相差约1.3e-10（相对），而本案例的损耗容差是1e-14——
#: **换成CODATA值本案例会当场红**，那是对的：换物理常数是一次决定，不是优化。
#: Norris 1970的推导用的就是这个定义值。
#: **不从本子包的``__init__``导出**：Track-E的互感模块很可能也需要同一个常量，
#: 收口时统一到一处（决策0047第六节登记）。
VACUUM_PERMEABILITY_H_PER_M: float = 4.0e-7 * math.pi

#: 损耗两式的级数/对数分支切换点（形制同`optics.bessel.SERIES_LIMIT`）。
#: **实测定的**：`i ≤ 0.7`用级数最坏3.5 eps，`i > 0.7`用`log1p`最坏10.6 eps；
#: 切到0.5则`log1p`一侧最坏22.9 eps，切到0.9则级数一侧要200余项才收敛。
LOSS_SERIES_LIMIT: float = 0.7

#: 级数项数上限：到不了收敛判据就失败关闭，绝不悄悄截断。
#: `i = 0.7`实测54项，留约7倍余量。
_LOSS_SERIES_MAX_TERMS: int = 400

#: 级数收敛判据：末项相对和小于它即停。
_LOSS_SERIES_FLOOR: float = 1.0e-18

#: 片电流密度的申报相对精度（`|x| ≤ 0.999·b`区间，对50位Decimal参考实测）。
#: 实测最坏4.48e-14（202 eps）落在`i = 0.01`、`|x|/b = 0.999`，取22倍余量。
#: **域外不申报**：`|x|/b → 1`时闭式本身病态，见模块docstring与案例页第四节。
SHEET_CURRENT_RELATIVE_ACCURACY: float = 1.0e-12

#: 两条损耗式的申报相对精度（`0 < i ≤ 1`全区间，对50位Decimal参考实测）。
#: 实测最坏2.96e-15（13.3 eps）落在`i = 0.709`——**刚过切换点的`log1p`分支**，
#: 那里`log1p`一侧的相消最重（`i`再大相消变轻，`i`再小就走级数了）。取3.4倍余量。
NORRIS_LOSS_RELATIVE_ACCURACY: float = 1.0e-14


class SuperconductorError(ValueError):
    """薄带临界态闭式的一切失败关闭：非物理输入、超临界电流、域外取值。"""


def _finite_positive(value: object, what: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0.0:
        raise SuperconductorError(f"{what}必须是有限正数：{value!r}")
    return number


def strip_critical_current_a(
    *, width_m: float, critical_sheet_current_a_per_m: float
) -> float:
    """带材的临界电流``Ic = w·Kc``（安培）。

    `Kc = Jc·d`是**片**临界电流密度（A/m），不是体电流密度（A/m²）——
    薄带近似把厚度`d`吸收进`Kc`，这正是同行"把厚度降维掉"的那一步。
    传体电流密度进来会差一个`d`（微米量级，即百万倍）且不报任何错，
    所以这道换算有名字、参数名带`per_m`后缀。
    """

    width = _finite_positive(width_m, "带宽width_m")
    sheet = _finite_positive(
        critical_sheet_current_a_per_m, "临界片电流密度critical_sheet_current_a_per_m"
    )
    return width * sheet


def sheet_critical_current_a_per_m(
    *, critical_current_density_a_per_m2: float, layer_thickness_m: float
) -> float:
    """``Kc = Jc·d``（A/m）——把厚度吸收进片电流密度的那一步。

    **这是薄带近似的定义式，不是一个方便函数**：`Jc`与`d`此后不再单独出现，
    闭式里只剩`Kc`。分开写是为了让"厚度到哪里去了"这件事在代码里看得见。
    """

    density = _finite_positive(
        critical_current_density_a_per_m2, "临界电流密度critical_current_density_a_per_m2"
    )
    thickness = _finite_positive(layer_thickness_m, "超导层厚度layer_thickness_m")
    return density * thickness


def current_ratio(*, transport_current_a: float, critical_current_a: float) -> float:
    """无量纲传输电流``i = I/Ic``，域是`[0, 1]`。

    超临界（`i > 1`）在临界态模型里**没有稳态解**——磁通前沿已经贯穿整条带，
    再多的电流无处可放。这里失败关闭而不是夹到1：
    "这个构型下判据不成立"与"判据成立且等于1"是两件事。
    """

    critical = _finite_positive(critical_current_a, "临界电流critical_current_a")
    current = float(transport_current_a)
    if not math.isfinite(current) or current < 0.0:
        raise SuperconductorError(f"传输电流必须是有限非负数（安培）：{transport_current_a!r}")
    ratio = current / critical
    if ratio > 1.0:
        raise SuperconductorError(
            f"传输电流{current!r} A超过临界电流{critical!r} A（i = {ratio!r}）："
            "临界态模型在超临界下没有稳态解，拒算"
        )
    return ratio


def flux_free_half_width_m(
    *, width_m: float, transport_current_a: float, critical_current_a: float
) -> float:
    """磁通前沿位置``b = (w/2)·sqrt(1 − (I/Ic)²)``（米）。

    `|x| < b`是**未穿透**的中心区（那里电流密度低于`Kc`），
    `b ≤ |x| ≤ w/2`是已饱和的边缘区（那里恰为`Kc`）。

    两个零容差极限**同时也是这条公式形式的判据来源**：
    `I = 0`给`b = w/2`（电流全挤在两条边、中间完全没穿透），
    `I = Ic`给`b = 0`（完全穿透）。research/12记原文把分母写作`I_m`，
    正是靠这两个极限反推出它应当是`Ic`——本仓按`Ic`写。

    **警告（决策0047第三节实测）**：这两个极限对"`sqrt`漏写"这条错法
    **一条都抓不住**（`b = (w/2)(1 − i²)`在`i = 0`与`i = 1`上给出同样的值，
    而且照样单调）。抓住它的是电流守恒`∫K dx = I`。
    """

    width = _finite_positive(width_m, "带宽width_m")
    ratio = current_ratio(
        transport_current_a=transport_current_a, critical_current_a=critical_current_a
    )
    return 0.5 * width * math.sqrt((1.0 - ratio) * (1.0 + ratio))


def sheet_current_density_a_per_m(
    *,
    position_m: float,
    width_m: float,
    transport_current_a: float,
    critical_sheet_current_a_per_m: float,
) -> float:
    """Norris薄带的片电流密度``K(x)``（A/m），横坐标`x`从带中心量起。

        K(x) = (2·Kc/π)·arctan( sqrt( ((w/2)² − b²) / (b² − x²) ) )    |x| < b
        K(x) = Kc                                                      b ≤ |x| ≤ w/2

    偶函数，且在`|x| = b`处连续（内支的极限恰为`Kc`）。

    **实现按代数等价的第二形写**：`(w/2)² − b² = ((w/2)·i)²`恒等，
    于是分子无相消；分母按`(b − |x|)(b + |x|)`算，`b − |x|`由Sterbenz引理精确。
    实测把样点最坏相对误差从1.07e-12压到4.48e-14。
    这不是等价重写的"顺手优化"——两条写法在`|x| → b`处差24倍。
    """

    width = _finite_positive(width_m, "带宽width_m")
    sheet = _finite_positive(
        critical_sheet_current_a_per_m, "临界片电流密度critical_sheet_current_a_per_m"
    )
    position = float(position_m)
    if not math.isfinite(position):
        raise SuperconductorError(f"横坐标position_m必须有限：{position_m!r}")
    half_width = 0.5 * width
    if abs(position) > half_width:
        raise SuperconductorError(
            f"横坐标{position!r} m落在带外（半宽{half_width!r} m）："
            "本闭式只在带材截面上有定义，不外推"
        )
    critical = strip_critical_current_a(
        width_m=width, critical_sheet_current_a_per_m=sheet
    )
    ratio = current_ratio(
        transport_current_a=transport_current_a, critical_current_a=critical
    )
    front = flux_free_half_width_m(
        width_m=width, transport_current_a=transport_current_a, critical_current_a=critical
    )
    distance = abs(position)
    if distance >= front:
        return sheet
    penetrated = half_width * ratio
    return (2.0 * sheet / math.pi) * math.atan(
        penetrated / math.sqrt((front - distance) * (front + distance))
    )


def _strip_series(ratio: float) -> float:
    """``2·[(1+i)ln(1+i) + (1−i)ln(1−i) − i²] = Σ_{n≥2} 2·i^(2n)/(n(2n−1))``。

    级数由`(1+x)ln(1+x) + (1−x)ln(1−x) = Σ_{n≥1} x^(2n)/(n(2n−1))`减去`x²`得到
    （`n = 1`项恰为`x²`，与`−i²`相消——**相消在纸上做掉了，浮点里就没有了**）。
    """

    total = 0.0
    for index in range(2, _LOSS_SERIES_MAX_TERMS + 1):
        term = 2.0 * ratio ** (2 * index) / (index * (2 * index - 1))
        total += term
        if term <= _LOSS_SERIES_FLOOR * total:
            return total
    raise SuperconductorError(
        f"薄带损耗级数在i = {ratio!r}处{_LOSS_SERIES_MAX_TERMS}项未收敛——"
        f"切换点LOSS_SERIES_LIMIT = {LOSS_SERIES_LIMIT!r}被改大了？拒答不截断"
    )


def _ellipse_series(ratio: float) -> float:
    """``(2−i)·i + 2(1−i)ln(1−i) = Σ_{n≥3} 2·i^n/(n(n−1))``。

    `n = 1`与`n = 2`两项与`(2−i)·i`逐项相消，同样是在纸上做掉的。
    """

    total = 0.0
    for index in range(3, _LOSS_SERIES_MAX_TERMS + 1):
        term = 2.0 * ratio**index / (index * (index - 1))
        total += term
        if term <= _LOSS_SERIES_FLOOR * total:
            return total
    raise SuperconductorError(
        f"椭圆截面损耗级数在i = {ratio!r}处{_LOSS_SERIES_MAX_TERMS}项未收敛"
    )


def _checked_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise SuperconductorError(
            f"归一化电流幅值i = I/Ic必须落在[0, 1]：{value!r}"
        )
    return ratio


def norris_strip_normalised_loss(current_ratio_value: float) -> float:
    """薄带的归一化磁滞损耗``2π·Ql/(μ0·Ic²)``（无量纲），`i = Im/Ic`。

        2π·Ql/(μ0·Ic²) = 2·[ (1+i)·ln(1+i) + (1−i)·ln(1−i) − i² ]   → i⁴/3  (i ≪ 1)

    小幅值渐近的**下一阶是`O(i²)`相对修正**（系数恰为`2/5`：
    级数次项`2i⁶/15`比首项`i⁴/3`是`0.4·i²`）。`i = 1e-3`实测相对偏差
    4.000002e-07，与`0.4·i² = 4.0e-07`吻合到六位。

    `i = 1`处两式都有限：本式给`2(2ln2 − 1) = 0.7725887…`。
    """

    ratio = _checked_ratio(current_ratio_value)
    if ratio == 0.0:
        return 0.0
    if ratio == 1.0:
        return 2.0 * (2.0 * math.log(2.0) - 1.0)
    if ratio <= LOSS_SERIES_LIMIT:
        return _strip_series(ratio)
    return 2.0 * (
        (1.0 + ratio) * math.log1p(ratio)
        + (1.0 - ratio) * math.log1p(-ratio)
        - ratio * ratio
    )


def norris_ellipse_normalised_loss(current_ratio_value: float) -> float:
    """椭圆截面线材的归一化磁滞损耗``2π·Ql/(μ0·Ic²)``（无量纲）。

        2π·Ql/(μ0·Ic²) = (2−i)·i + 2·(1−i)·ln(1−i)                   → i³/3  (i ≪ 1)

    Norris证明它对**任意半轴比**的椭圆成立（`Ic = π·a·b·Jc`）。

    **下一阶是`O(i)`相对修正不是`O(i²)`**（级数次项`i⁴/6`比首项`i³/3`是`i/2`）：
    `i = 1e-3`实测相对偏差5.003002e-04，与`i/2 = 5.0e-04`吻合到四位。
    research/12的判据表把两条渐近的理由都写成"下一阶是`O(i²)`相对修正"，
    **对薄带成立、对椭圆不成立**——两条容差因此必须差三个数量级，
    不能共用一个1e-3（决策0047第三节）。

    `i = 1`处本式恰为`1`。
    """

    ratio = _checked_ratio(current_ratio_value)
    if ratio == 0.0:
        return 0.0
    if ratio == 1.0:
        return 1.0
    if ratio <= LOSS_SERIES_LIMIT:
        return _ellipse_series(ratio)
    return (2.0 - ratio) * ratio + 2.0 * (1.0 - ratio) * math.log1p(-ratio)


def norris_strip_loss_j_per_m_per_cycle(
    *, critical_current_a: float, current_amplitude_a: float
) -> float:
    """薄带的每周期每单位长度磁滞损耗``Ql = μ0·Ic²·F(i)/(2π)``（J/m/周期）。

    单位边界就在这一行：归一化量是无量纲的，乘上`μ0·Ic²/(2π)`才落回焦耳。
    `μ0`是**物理常数**可以放在模块里；`Ic`是**材料属性**必须由调用方给，
    今天还进不了材料记录（决策0047第四节）。
    """

    critical = _finite_positive(critical_current_a, "临界电流critical_current_a")
    amplitude = float(current_amplitude_a)
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise SuperconductorError(f"电流幅值必须是有限非负数（安培）：{current_amplitude_a!r}")
    normalised = norris_strip_normalised_loss(amplitude / critical)
    return VACUUM_PERMEABILITY_H_PER_M * critical * critical * normalised / (2.0 * math.pi)


def norris_ellipse_loss_j_per_m_per_cycle(
    *, critical_current_a: float, current_amplitude_a: float
) -> float:
    """椭圆截面线材的每周期每单位长度磁滞损耗（J/m/周期）。"""

    critical = _finite_positive(critical_current_a, "临界电流critical_current_a")
    amplitude = float(current_amplitude_a)
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise SuperconductorError(f"电流幅值必须是有限非负数（安培）：{current_amplitude_a!r}")
    normalised = norris_ellipse_normalised_loss(amplitude / critical)
    return VACUUM_PERMEABILITY_H_PER_M * critical * critical * normalised / (2.0 * math.pi)


__all__ = [
    "LOSS_SERIES_LIMIT",
    "NORRIS_LOSS_RELATIVE_ACCURACY",
    "SHEET_CURRENT_RELATIVE_ACCURACY",
    "SUPERCONDUCTOR_DIMENSIONLESS_RESULTS",
    "SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES",
    "SUPERCONDUCTOR_LENGTH_UNIT",
    "VACUUM_PERMEABILITY_H_PER_M",
    "SuperconductorError",
    "current_ratio",
    "flux_free_half_width_m",
    "norris_ellipse_loss_j_per_m_per_cycle",
    "norris_ellipse_normalised_loss",
    "norris_strip_loss_j_per_m_per_cycle",
    "norris_strip_normalised_loss",
    "sheet_critical_current_a_per_m",
    "sheet_current_density_a_per_m",
    "strip_critical_current_a",
]
