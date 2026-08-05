"""第一、二类完全椭圆积分（AGM），外加互感专用的**无相消**组合。

## 一、为什么不复用大挠度悬臂那份Carlson实现

`cases/large_deflection_cantilever/generate_oracle.py`里确实有一份Carlson对称形式
（`carlson_rf`/`carlson_rd`/`complete_k`/`complete_e`）。**不能复用，两条独立理由**：

1. **它在`cases/`里不在`src/`里，而`cases/`不是包的一部分**。
   引擎的生产路径import一个案例目录的脚本，`pip install`出来的wheel里根本没有它；
2. **就算搬进`src/`也不该复用**——spec/12第3.2节要求每块物理带一条
   **与生产路径不共享代码**的独立求值路径。那份Carlson是大挠度悬臂案例的
   **金标生成器**；本模块若与它共用代码，本域案例的"独立oracle"就名存实亡。

所以本模块走**AGM（算术-几何平均）**，与Carlson重复化是两条不同的算法；
而`cases/mutual_inductance_coaxial/generate_oracle.py`那一侧**复用**了
悬臂案例的Carlson（生成器之间复用是正当的，两侧仍然无共用代码）。

## 二、约定：本模块的公开函数收的是**模k**，不是参数m = k²

**这条不写清必错，而且错出来的数不会离谱到一眼看出**——两种约定下都是同一族函数，
`K(k=0.5) = 1.6858`而`K(m=0.5) = K(k=0.7071) = 1.8541`，差10%，
在一条本来就跨十几个数量级的互感值上没有任何"看起来不对"的提示。

本仓**两种约定都已经在用**：

* `cases/large_deflection_cantilever`的`complete_k(m)`收的是**参数m**；
* Maxwell互感闭式按传统写作`K(k)`、`E(k)`，收的是**模k**。

处理照0031第3.3节那条纪律（**每一处单位/约定边界都要有名字**）：
两种约定各给一对函数，名字里写死收的是哪个，**不提供"聪明地猜"的入口**。
`cases/mutual_inductance_coaxial`有一条判据把
`complete_elliptic_k(0.5)`与`complete_elliptic_k_of_parameter(0.25)`
钉成**逐位相同**、并同时钉住`complete_elliptic_k_of_parameter(0.5)`是另一个数——
一条判据同时守住"换算对"与"两者确实不同"。

**顺带一条本仓特有的坑**：椭圆积分的参数传统上叫`m`，而本仓`_m`后缀是**米**
（轴2单位后缀）。所以本模块一个参数都不叫`m`——叫`parameter`、叫`modulus`。

## 三、互感组合`(2/k − k)·K − (2/k)·E`必须换一种算法，不是风格问题

Maxwell闭式里那个方括号在小k下**灾难性相消**：两项各是`O(1/k)·O(1)`量级，
而差是`O(k³)`，相消放大约`1/k⁴`。实测（对60位Decimal高精度）：

| k | 教科书分组的相对误差 | 本模块的相对误差 |
|---|---|---|
| 0.1 | 8.9e-12 | 1.9e-17 |
| 0.02 | 2.4e-9 | 1.8e-16 |
| 0.01 | 7.6e-8 | 1.6e-16 |
| 1e-3 | 1.8e-3 | 2.4e-17 |
| 1e-4 | **19.5（1950%）** | 7.7e-16 |

**小k正是远场**（k² = 4r1r2/((r1+r2)²+d²)，d ≫ r 即 k → 0），
也就是"退化到偶极子"那条判据要用的整个区间。**照教科书分组写，那条判据做不了**：
d增大两三个倍频程之后误差就淹没掉待测的偏差本身。

无相消的写法来自AGM自己的中间量。设AGM序列
`a₀ = 1`、`b₀ = k' = √(1−k²)`、`aₙ₊₁ = (aₙ+bₙ)/2`、`bₙ₊₁ = √(aₙbₙ)`、
`cₙ₊₁ = (aₙ−bₙ)/2`，则（A&S 17.6）

    K = π/(2·a_∞)
    E = K·(1 − Σ_{n≥0} 2^{n−1}·cₙ²)   其中 c₀ = k

记 **T = Σ_{n≥1} 2ⁿ·cₙ²**（**去掉n=0那一项**）。于是 2·Σ2^{n−1}cₙ² = k² + T，

    (2/k − k)·K − (2/k)·E = (2/k)·(K − E) − k·K
                          = (2/k)·K·Σ2^{n−1}cₙ² − k·K
                          = K·(k² + T − k²)/k
                          = **K·T/k**

`T`是一串**全正**项之和，`K`为正——**整条路上一次减法都没有**。
这不是近似，是同一个量的另一种分组。

## 四、AGM本身有两个坑，两个都是实测撞出来的，不是抄来的

**坑一：循环不终止，而`2ⁿ`权重会把浮点地板放大成真误差。**
`cₙ`本该二次收敛到0，但浮点下`aₙ`与`bₙ`最终停在相差一个ulp，`cₙ`卡在`2⁻⁵⁴`附近不动，
而`2ⁿ`还在翻倍。**它注入的是`T`上的一个绝对地板**约`2^N·ε²`（`N`是迭代上限）——
所以**它的危害与k成反比**：`T ≈ k⁴/8`，k大时那个地板淹没在有效数字里，
k小时它就是全部。实测（迭代上限40，扫513个模）：

| k | 不停机注入的相对误差 |
|---|---|
| 0.1及以上 | 0（`cₙ`恰好落到0，从未停滞） |
| 0.01 | 0 |
| 1e-4 | **5.4e-4** |
| 1e-8 | **5.4e12** |

**注意这条与迭代上限直接挂钩**：把上限从40写成64，同一个地板放大`2²⁴ ≈ 1.7e7`倍，
于是连k=0.035（r=10mm、d=800mm那一组）都被污染到5.8e-7——**本模块最早的一版
正是这么写的，实测撞出来才发现**。处理：`cₙ`一旦不再变小就立刻停
（`c >= previous`），并另配一条迭代上限**失败关闭**；两条都留着，
因为停机判定管正常收敛、上限管病态输入。

**坑二：`c₁ = (1 − √(1−k²))/2`在小k下相消。**
k=1e-4时`√(1−k²)`与1差5e-9，相减丢掉8位。而`T ≈ 2c₁²`，
于是误差翻倍进`T`、再原样进M。实测这一条单独造成的相对误差：
k=0.1处4.4e-15、k=0.01处4.6e-13、k=1e-3处3.2e-10、k=1e-4处2.7e-8——
**它比坑一更早开始咬，而且在整个远场区间连续存在**。
处理：`c₁`按恒等式`(1−k')/2 = k²/(2(1+k'))`直接算，**不做那次减法**；
`k'`本身也按`√((1−k)(1+k))`算，在k→1一侧少一次相消。

## 五、精度申报（实测，不是推断）

对60位`decimal`高精度AGM逐点对拍，`k ∈ [1e-12, 1−1e-8]`共410个点：

* `complete_elliptic_k`最坏相对误差 **3.4e-16**；
* `complete_elliptic_e`最坏相对误差 **9.8e-16**；
* `maxwell_mutual_bracket`最坏相对误差 **7.7e-16**。

域外（`k > MODULUS_MAX`）**拒跑**，不返回nan也不夹边界。
物理上`k → 1`意味着两条丝状回路重合，互感对数发散——那不是精度问题，
是丝状回路模型在该处失效（见`inductance.py`）。
"""

from __future__ import annotations

import math

from physics_engine.electromagnetics.errors import ElectromagneticsError

#: 模的上界。`k → 1`时`K`对数发散，且`1−k²`在浮点下相消到失去有效位。
#: 取`1 − 1e-12`：此处`1−k²`仍有约2e-12，保住约4位十进制余量。
#: **超过就拒跑**——夹到边界会给出一个有限的、看起来正常的错值。
MODULUS_MAX: float = 1.0 - 1.0e-12

#: AGM迭代上限。二次收敛下double精度实测最多用9次；40是活性护栏不是SLA。
#: 到顶仍未收敛=失败关闭，不许"用当前值凑合"。
AGM_ITERATION_BOUND: int = 40

#: 申报精度（第五节实测值），供案例页与判据表引用，避免两处各写一个数。
COMPLETE_K_RELATIVE_ACCURACY: float = 3.4e-16
COMPLETE_E_RELATIVE_ACCURACY: float = 9.8e-16
MAXWELL_BRACKET_RELATIVE_ACCURACY: float = 7.7e-16


def _require_modulus(modulus: object) -> float:
    if isinstance(modulus, bool) or not isinstance(modulus, (int, float)):
        raise ElectromagneticsError(f"modulus必须是实数：{modulus!r}")
    value = float(modulus)
    if not math.isfinite(value):
        raise ElectromagneticsError(f"modulus必须是有限值：{modulus!r}")
    if value < 0.0:
        raise ElectromagneticsError(
            f"modulus必须落在[0, {MODULUS_MAX!r}]：{modulus!r}——"
            "本模块收的是**模k**不是参数m=k²，负模没有意义"
        )
    if value > MODULUS_MAX:
        raise ElectromagneticsError(
            f"modulus={modulus!r}超过{MODULUS_MAX!r}：K(k)在k→1处对数发散，"
            "此处1−k²已相消到失去有效位——拒跑，不夹边界也不返回nan"
        )
    return value


def _require_parameter(parameter: object) -> float:
    if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
        raise ElectromagneticsError(f"parameter必须是实数：{parameter!r}")
    value = float(parameter)
    if not math.isfinite(value):
        raise ElectromagneticsError(f"parameter必须是有限值：{parameter!r}")
    if value < 0.0 or value > MODULUS_MAX * MODULUS_MAX:
        raise ElectromagneticsError(
            f"parameter必须落在[0, {MODULUS_MAX * MODULUS_MAX!r}]：{parameter!r}——"
            "本函数收的是**参数m = k²**；若手上是模k请改用不带_of_parameter的那一对"
        )
    return value


def agm_kernel(modulus: float) -> tuple[float, float]:
    """AGM一次跑完，返回``(K, tail)``。

    ``tail`` = ``Σ_{n≥1} 2ⁿ·cₙ²``（**不含n=0的k²项**）。三个公开量都从它出：

    * ``K = π/(2·a_∞)``
    * ``E = K·(1 − k²/2 − tail/2)``
    * ``(2/k − k)·K − (2/k)·E = K·tail/k``

    返回中间量而不是只返回K与E，正是为了第三条能不做减法——
    见模块docstring第三节。
    """

    k = _require_modulus(modulus)
    # `(1−k)(1+k)`而不是`1−k*k`：k接近1时前者少一次相消。
    complement = math.sqrt((1.0 - k) * (1.0 + k))
    a, b = 1.0, complement
    # c₁ = (1 − k')/2 = k²/(2(1+k'))。右式**不做减法**（模块docstring第四节坑二）。
    c = k * k / (2.0 * (1.0 + complement))
    tail = 0.0
    power = 2.0
    for _ in range(AGM_ITERATION_BOUND):
        if c <= 0.0:
            break
        tail += power * c * c
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        previous, c = c, 0.5 * (a - b)
        # cₙ本该二次收敛；一旦不再变小就是踩到浮点地板了，而2ⁿ还在翻倍——
        # 继续加下去是把地板噪声放大成真误差（坑一，实测5.8e-7）。
        if c >= previous:
            break
        power *= 2.0
    else:
        raise ElectromagneticsError(
            f"AGM在{AGM_ITERATION_BOUND}次迭代内没有收敛：modulus={modulus!r}"
        )
    return math.pi / (2.0 * a), tail


def complete_elliptic_k(modulus: float) -> float:
    """第一类完全椭圆积分``K(k)``——**收的是模k**。

    ``K(k) = ∫₀^{π/2} dθ / √(1 − k²·sin²θ)``。
    参数m = k²的那一版是``complete_elliptic_k_of_parameter``。
    """

    return agm_kernel(modulus)[0]


def complete_elliptic_e(modulus: float) -> float:
    """第二类完全椭圆积分``E(k)``——**收的是模k**。

    ``E(k) = ∫₀^{π/2} √(1 − k²·sin²θ) dθ``。
    """

    k = _require_modulus(modulus)
    complete, tail = agm_kernel(k)
    return complete * (1.0 - 0.5 * k * k - 0.5 * tail)


def complete_elliptic_k_of_parameter(parameter: float) -> float:
    """``K``按**参数m = k²**取值——与`cases/large_deflection_cantilever`同约定。

    ``complete_elliptic_k_of_parameter(m) == complete_elliptic_k(sqrt(m))``。
    两个入口并存是**故意的**：本仓两种约定都在用，逼调用点说清要哪一个，
    比让一个函数去猜安全（0031第3.3节的做法）。
    """

    return complete_elliptic_k(math.sqrt(_require_parameter(parameter)))


def complete_elliptic_e_of_parameter(parameter: float) -> float:
    """``E``按**参数m = k²**取值。见``complete_elliptic_k_of_parameter``。"""

    return complete_elliptic_e(math.sqrt(_require_parameter(parameter)))


def maxwell_mutual_bracket(modulus: float) -> float:
    """Maxwell互感闭式里的方括号``(2/k − k)·K(k) − (2/k)·E(k)``，**无相消求值**。

    数学上与教科书分组逐字相同；数值上不是——教科书分组在小k处相消放大约``1/k⁴``，
    k=1e-4时相对误差达1950%（模块docstring第三节有实测表）。
    本函数按恒等式返回``K·tail/k``，全程无减法。

    ``modulus = 0``返回精确的``0.0``：两条回路无限远（或有一条半径为零）时互感为零，
    这是极限的精确值，不走除法。
    """

    k = _require_modulus(modulus)
    if k == 0.0:
        return 0.0
    complete, tail = agm_kernel(k)
    return complete * tail / k


__all__ = [
    "AGM_ITERATION_BOUND",
    "COMPLETE_E_RELATIVE_ACCURACY",
    "COMPLETE_K_RELATIVE_ACCURACY",
    "MAXWELL_BRACKET_RELATIVE_ACCURACY",
    "MODULUS_MAX",
    "agm_kernel",
    "complete_elliptic_e",
    "complete_elliptic_e_of_parameter",
    "complete_elliptic_k",
    "complete_elliptic_k_of_parameter",
    "maxwell_mutual_bracket",
]
