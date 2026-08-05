"""电磁域的长度制边界与真空磁导率——**本域最先撞上的那条诚实边界**。

## 为什么这个模块存在

plans/03第四节D早就登记了一条边界：**同一块铜的杨氏模量与电导率放不进
同一条材料记录**——六个电磁本构量（μ、ε、σ、E、M、ρ）的字段名都带米制后缀，
被mm制记录的隔离规则拒收（research/07实测）。那一条写着"光学没跑它，
**电磁会是第一个跑不通的域**"。

本块是电磁域的第一块代码，所以它有义务如实交代自己**撞上了哪一半**：

* **撞上的是"引擎的几何是mm制、电磁量天然米制"这一条**。μ₀的单位是H/m，
  `M = μ₀·√(r1·r2)·f(k)`里那个`√(r1·r2)`必须是**米**，否则M不是亨利。
  两条回路的半径若按spec/11的主单位写成mm，直接代进去会差1000倍——
  与0024那条真bug是**同一个形状**；
* **没撞上材料记录那一半**。真空里两条回路的互感只要几何与μ₀，
  **一个材料字段都不读**。所以本块不建`parameters.py`、不读`MaterialRecord`——
  按0001第二前提，没有真实需求的通用性不预支。plans/03第四节D那条边界
  因此仍然开着，**本块把它撞开了一半、如实说清另一半没碰**。

## 三条处理，全部是"显式且有门守着"

1. **长度制钉死为米**（`EM_LENGTH_UNIT`），与`optics`同制。这不是风格选择：
   μ₀按定义就是H/**m**；
2. **换算只有一个入口且有名字**（`metres_from_millimetres`），
   而且是`CircularLoop.from_millimetres`的唯一实现——mm几何进本域**只有这一条路**；
3. **换算方向由门守着**：`cases/mutual_inductance_coaxial`有一条往返判据
   （mm→m→mm逐位复原）与一条"同一几何用mm声明与用m声明给出**逐位相同**的M"。
   本仓有过三次单位事故（1000倍、被抵消、静默1e12方向反），
   方向写反不会报错、只会给出一个看起来合理的数——**只有往返判据能抓**。

## 关于μ₀：它**不是**按定义精确的常数（2019 SI重定义之后）

research/08第4.4节的结论在这里第一次成为代码：2019年SI重定义把元电荷定为精确值，
μ₀随之**不再**是精确的4π×10⁻⁷ H/m，而是由精细结构常数定出的**测量量**
（μ₀ = 2αh/(ce²)）。

**直接后果（spec/14第三节证据分级）**：μ₀**不得**登记为`benchmark_constant`
——那一档明写是"按定义精确"的解析基准常数。本模块把它申报为`measured`，
并把旧的按定义值一并留在`LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M`里，
让"两者差多少"成为一个**可被判据钉住的数**而不是一句注释。

这一条同时是μ₀量级写错的捕手：数量级错一位，
`VACUUM_PERMEABILITY_RELATIVE_DEVIATION_FROM_LEGACY`那条判据会红到看不懂为止。
"""

from __future__ import annotations

import math

from physics_engine.electromagnetics.errors import ElectromagneticsError

#: 本域的长度制。μ₀的单位是H/**m**，所以这不是风格选择。
#: 与`optics.OPTICS_LENGTH_UNIT`同为米，与spec/11的几何主单位mm**不同**——
#: 这条差别就是本模块存在的理由。
EM_LENGTH_UNIT: str = "m"

#: spec/11的几何主单位。写在这里是为了让换算的两端都有名字。
GEOMETRY_LENGTH_UNIT: str = "mm"

#: 唯一的长度换算因子。**只写一次**：写两次就会有一次写反。
MILLIMETRES_PER_METRE: float = 1.0e3

#: 真空磁导率（CODATA 2022推荐值，H/m）。
#: **不是按定义精确**——2019 SI重定义之后它由μ₀ = 2αh/(ce²)定出，是测量量。
VACUUM_PERMEABILITY_H_PER_M: float = 1.25663706127e-6

#: CODATA 2022给出的相对标准不确定度（约1.6×10⁻¹⁰）。
#: 它是本域一切互感值的**物理精度地板**：算得再准也不会比这个准。
VACUUM_PERMEABILITY_RELATIVE_UNCERTAINTY: float = 1.6e-10

#: 2019年之前按定义精确的旧值4π×10⁻⁷ H/m。留着不是为了用，
#: 是为了让"新旧差多少"成为一个可被判据钉住的数。
LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M: float = 4.0e-7 * math.pi

#: 证据分级（spec/14第三节）。**`measured`不是`benchmark_constant`**——
#: 后者按定义精确，而μ₀自2019 SI重定义起不再是。研究依据见research/08第4.4节。
VACUUM_PERMEABILITY_EVIDENCE_GRADE: str = "measured"


def vacuum_permeability_relative_deviation_from_legacy() -> float:
    """本仓采信的μ₀与旧的按定义值4π×10⁻⁷的相对偏差。

    量级约1.3×10⁻¹⁰，**落在CODATA的不确定度之内**——即两个值在物理上不冲突，
    冲突的只是"它是不是精确的"这条声明。

    这条同时是**μ₀量级写错的捕手**：把1.2566e-6写成1.2566e-5，
    本函数从1.3e-10跳到8.0，任何有理由的容差都拦得住。
    """

    legacy = LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M
    return abs(VACUUM_PERMEABILITY_H_PER_M - legacy) / legacy


def require_em_length_unit(length_unit: str) -> str:
    """本域只收米制，别的一律拒跑。

    照`optics/parameters.py`那条纪律：**不在这里替调用方换制**。
    换制是调用方代码里必须显式发生的一件事（`metres_from_millimetres`），
    在这里偷偷换掉，出错时没有任何地方能看出来换过。
    """

    if length_unit != EM_LENGTH_UNIT:
        raise ElectromagneticsError(
            f"电磁域按{EM_LENGTH_UNIT!r}制，收到{length_unit!r}——"
            f"若几何是{GEOMETRY_LENGTH_UNIT!r}制，请显式调用"
            f"metres_from_millimetres()，不要指望本域替你换"
        )
    return length_unit


def metres_from_millimetres(length_mm: float) -> float:
    """mm → m。**mm几何进入电磁域的唯一入口。**

    名字里两端的单位都写出来了（`..._from_...`），因为方向写反
    （乘1000而不是除1000）在数值上完全合法、不会报任何错，
    只会让互感大出一百万倍——而互感本来就跨十几个数量级，
    "看起来不对"这件事在这里不成立。
    """

    value = _require_finite(length_mm, "length_mm")
    return value / MILLIMETRES_PER_METRE


def millimetres_from_metres(length_m: float) -> float:
    """m → mm。与`metres_from_millimetres`成对存在，供往返判据用。

    **两个方向都用同一个`MILLIMETRES_PER_METRE`**，一个乘一个除——
    因子只有一份，抄错一处两处都错，往返判据当场红。
    """

    value = _require_finite(length_m, "length_m")
    return value * MILLIMETRES_PER_METRE


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ElectromagneticsError(f"{name}必须是实数：{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ElectromagneticsError(f"{name}必须是有限值：{value!r}")
    return number


__all__ = [
    "EM_LENGTH_UNIT",
    "GEOMETRY_LENGTH_UNIT",
    "LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M",
    "MILLIMETRES_PER_METRE",
    "VACUUM_PERMEABILITY_EVIDENCE_GRADE",
    "VACUUM_PERMEABILITY_H_PER_M",
    "VACUUM_PERMEABILITY_RELATIVE_UNCERTAINTY",
    "metres_from_millimetres",
    "millimetres_from_metres",
    "require_em_length_unit",
    "vacuum_permeability_relative_deviation_from_legacy",
]
