"""切向：库仑return-map（各向同性圆与各向异性椭圆）与粘着弹簧。

两条切向屈服面并列，**不是一条替换另一条**：

| 函数 | 屈服面 | 返回方向 |
|---|---|---|
| `coulomb_return_map` | 圆``\\|f_t\\| ≤ μN`` | 径向 |
| `anisotropic_return_map` | 椭圆``(f_∥/μ_∥N)² + (f_⊥/μ_⊥N)² ≤ 1`` | **最近点（外法向）** |

后者在``μ_∥ == μ_⊥``时**把参数原样转交前者**，故退化是逐位的
（决策0068第三节；那条不是"算出来恰好相等"，是构造出来的）。

拆分自原`contact.py`（2026-08-17）——**当时函数体逐字节未动**；
各向异性那一片是2026-08-17后加的新代码（决策0068）。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.contact.layout import (
    NORMAL_UNIT_TOLERANCE,
    REGIME_SEPARATED,
    REGIME_SLIP,
    REGIME_STICK,
)
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.state import State

#: 面内纵向轴的最小可辨正弦。``along_direction``几乎与法向平行时，
#: 它投影到切平面后剩下的那一点长度**由舍入决定**——那不是一个方向，是噪声。
#: 取1e-6：此时面内轴的方向误差约``eps/sin ≈ 2e-10``，仍远小于任何物理量的容差；
#: 再小就开始拿噪声当纵向轴用。
IN_PLANE_DIRECTION_MIN_SINE = 1e-6

#: 试探力允许的法向分量（相对``|f_trial|``）。摩擦力住在切平面里，
#: 试探力带着法向分量进来意味着**调用方忘了投影**——那是个静默错值的入口，
#: 不是一个需要被容忍的数值现象。取1e-9：比fp64舍入高七个数量级，
#: 正常调用（`TangentialStickSpring.tangential_force_n`已扣掉法向）永远碰不到它。
TRIAL_OUT_OF_PLANE_TOLERANCE = 1e-9

#: η迭代的残差地板。残差``1/√Φ − 1``是无量纲O(1)量，它落到几个ulp时
#: **再迭代是在磨一个不影响答案的量**：η的相对变化再大，
#: 力的变化也被``a²+η``的分母压到1e-17以下（2026-08-17实测）。
_PLASTIC_RESIDUAL_FLOOR = 8.0 * sys.float_info.epsilon

#: η迭代的趟数上限。实测最坏17趟（``μ_∥:μ_⊥``从1:1到1e4:1、
#: 试探力从超出1e-15倍到1e12倍、逐度扫一圈），均值约3趟。
#: 取64是**绊线不是预算**：走满就说明推导错了，此时抛比给一个错数好。
_PLASTIC_MAX_ITERATIONS = 64


@dataclass(frozen=True)
class FrictionOutcome:
    """一次return-map的结果：实际切向力、粘/滑判别、以及**锚点该挪到哪**。

    ``anchor_correction_mm``是滑移那一步锚点要平移的量（粘住时为零矢量）。
    **它是这一步产生的不可逆位移**——把它写回状态，历史就被记住了；
    不写回，下一步会以为自己还粘在原处，于是摩擦力凭空多出一截。
    """

    tangential_force_n: tuple[float, float, float]
    regime: float
    anchor_correction_mm: tuple[float, float, float]

    @property
    def is_stick(self) -> bool:
        return self.regime == REGIME_STICK


def coulomb_return_map(
    *,
    trial_force_n: tuple[float, float, float],
    normal_force_n: float,
    friction_coefficient: float,
    tangential_stiffness_n_per_mm: float,
) -> FrictionOutcome:
    """库仑摩擦的return-map：试探力落在摩擦锥内就粘，超出就投影回锥面并挪锚点。

    ## 这一段为什么不是能量项（0050第二节）

    库仑摩擦**耗散且非associative**——它做的功依赖路径，写不成任何位置的势函数。
    所以接触在本仓是"**半个能量项**"：法向在`PenaltyNormalContact`里，
    切向在这里，而这里**不满足**`EnergyTerm`四方法协议，也不该假装满足。

    ## 判据

    ``|T_trial| ≤ μN`` → **粘**：实际力就是试探力，锚点不动；
    否则 → **滑**：``T = μN · T_trial/|T_trial|``，并把锚点沿滑移方向挪
    ``(|T_trial| − μN)/k_t``——挪完之后，用新锚点重算的试探力恰好落在锥面上。
    **这一条是return-map的定义，也是它可被验证的地方**（见`tests/`那条自洽门）。

    ## 边界

    ``N = 0``（分离）时摩擦锥退化成一个点：**没有法向力就没有摩擦**，
    一切试探力都被投影成零，判别是`REGIME_SEPARATED`而不是`REGIME_SLIP`——
    "分离"与"在滑"是两件事，混起来会让案例分不清"飞出去了"和"在蹭着走"。
    """

    #: **试探力此前完全不校验**，而同一函数对``N``/``μ``/``k_t``都很严
    #: （2026-08-06对抗审核）：nan进来会原样变成锚点修正**写进状态向量**，
    #: 而状态是复现契约；长度2或4的元组会被原样返回。
    if len(trial_force_n) != 3 or not all(math.isfinite(v) for v in trial_force_n):
        raise ContactError(f"trial force must be a finite 3-vector: {trial_force_n!r}")
    if normal_force_n < 0.0 or not math.isfinite(normal_force_n):
        raise ContactError(f"normal force must be finite and nonnegative: {normal_force_n!r}")
    if friction_coefficient < 0.0 or not math.isfinite(friction_coefficient):
        raise ContactError(
            f"friction coefficient must be finite and nonnegative: {friction_coefficient!r}"
        )
    if not (tangential_stiffness_n_per_mm > 0.0 and math.isfinite(tangential_stiffness_n_per_mm)):
        raise ContactError(
            f"tangential stiffness must be positive: {tangential_stiffness_n_per_mm!r}"
        )

    zero = (0.0, 0.0, 0.0)
    if normal_force_n == 0.0:
        return FrictionOutcome(zero, REGIME_SEPARATED, zero)

    limit = friction_coefficient * normal_force_n
    magnitude = math.sqrt(sum(component * component for component in trial_force_n))
    if magnitude <= limit:
        return FrictionOutcome(trial_force_n, REGIME_STICK, zero)

    scale = limit / magnitude
    force = tuple(component * scale for component in trial_force_n)
    #: 超出锥面的那一截除以切向刚度，就是这一步滑掉的距离。
    slip_mm = (magnitude - limit) / tangential_stiffness_n_per_mm
    correction = tuple(component / magnitude * slip_mm for component in trial_force_n)
    return FrictionOutcome(force, REGIME_SLIP, correction)


@dataclass(frozen=True)
class FrictionEllipse:
    """各向异性摩擦的屈服面：切平面里的一个椭圆，**外加它自己的朝向**。

    ``(f_∥/(μ_∥·N))² + (f_⊥/(μ_⊥·N))² ≤ 1``

    ## 朝向是一个显式字段，**不许从别的量的符号里推**

    本仓在这一条上吃过一次亏：`PenaltyAnnulusLimit`曾把朝向编码在坐标符号里，
    于是**单元门永远抓不到取错朝向**，端到端跑一次才炸。
    椭圆的长轴指哪边是这个类的一等信息，所以它是一个有名字、被校验、
    出现在`repr`里的字段——`along_direction`。

    ``normal``也必须显式给：椭圆住在**切平面**里，而`coulomb_return_map`
    根本不知道法向是谁（它只取三维模长）。这是两条路径签名不同的根因，
    不是接口不齐。

    ## 四条校验各自挡什么

    * 两个``μ``都必须**严格为正**：``μ = 0``让椭圆退化成一条线段，
      那是一次**不同的**投影（往线段上夹紧），今天没有消费方要它，
      而按现有公式算会直接除零。**登记为边界**，触发条件：
      第一个真的要"某方向完全无摩擦"的消费方；
    * ``along_direction``与``normal``都必须是有限单位矢量：nan混进来
      会让整条链路静默变nan（`TangentialStickSpring`吃过这一条）；
    * ``along_direction``不许（近乎）平行于``normal``：投影完剩下的
      那一点长度由舍入决定，**拿噪声当纵向轴用比报错坏得多**；
    * 面内轴在`__post_init__`里算一次就缓存：它是构造期定死的常量，
      每次return-map重算等于把常量放进内层循环。
    """

    #: 沿``along_direction``（带长方向）的摩擦系数。
    mu_along: float
    #: 垂直于``along_direction``、仍在切平面内（带宽方向）的摩擦系数。
    mu_across: float
    #: 带长方向。**不要求已经落在切平面内**——本类把它投影进去。
    along_direction: tuple[float, float, float]
    #: 接触面的单位法向，定义椭圆所在的切平面。
    normal: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name, value in (("mu_along", self.mu_along), ("mu_across", self.mu_across)):
            if not (value > 0.0 and math.isfinite(value)):
                raise ContactError(
                    f"{name} must be finite and strictly positive: {value!r} — "
                    "零摩擦系数把椭圆压成线段，那是另一次投影（本类不做，见docstring）"
                )
        for name, vector in (
            ("along_direction", self.along_direction),
            ("normal", self.normal),
        ):
            if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
                raise ContactError(f"{name} must be a finite 3-vector: {vector!r}")
            norm = math.sqrt(sum(component * component for component in vector))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(f"{name} must be a unit vector (|v| = {norm!r})")

        normal = self.normal
        along_normal = sum(
            self.along_direction[axis] * normal[axis] for axis in range(3)
        )
        in_plane = tuple(
            self.along_direction[axis] - along_normal * normal[axis] for axis in range(3)
        )
        length = math.sqrt(sum(component * component for component in in_plane))
        if length < IN_PLANE_DIRECTION_MIN_SINE:
            raise ContactError(
                f"along_direction与normal夹角太小（投影后长度{length!r}）"
                "——面内纵向轴会由舍入决定，那不是方向是噪声"
            )
        axis_along = tuple(component / length for component in in_plane)
        #: 右手系：``e_⊥ = n × e_∥``。定死它是因为**符号约定必须是一件被写下来的事**，
        #: 而不是"读代码自己看出来"的事——椭圆本身对``e_⊥``反号不变
        #: （见`test_the_ellipse_does_not_care_about_the_sign_of_along_direction`），
        #: 但下一个往这里加"横向落位偏差取正负"的人需要这句话。
        axis_across = (
            normal[1] * axis_along[2] - normal[2] * axis_along[1],
            normal[2] * axis_along[0] - normal[0] * axis_along[2],
            normal[0] * axis_along[1] - normal[1] * axis_along[0],
        )
        object.__setattr__(self, "_axes", (axis_along, axis_across))

    def in_plane_axes(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """``(e_∥, e_⊥)``：椭圆两个主轴在世界坐标下的单位矢量。"""

        return self._axes  # type: ignore[attr-defined]

    @property
    def is_isotropic(self) -> bool:
        """两个``μ``**逐位相等**。这一条为真时椭圆就是圆，走旧路径。"""

        return self.mu_along == self.mu_across


def _positive_zero(value: float) -> float:
    """把``-0.0``收成``+0.0``；其余值逐位不变（``x + 0.0``在IEEE-754下是精确的）。

    **这不是洁癖，是产物契约。** 实测：`canonical_bytes`把``-0.0``写成``-0.0``、
    把``0.0``写成``0.0``，两者字节不同，于是`State.fingerprint()`不同。
    而``f_∥·e_∥ + f_⊥·e_⊥``在面外那一格上恰好是``(±0.0) + (∓0.0)``——
    **它的符号跟着``along_direction``的符号翻**，尽管两个值都精确等于零。
    没有这一步，"翻转纵向轴结果不变"这句话在字节层面是假的
    （`test_the_ellipse_does_not_care_about_the_sign_of_along_direction`当场抓到）。

    **`coulomb_return_map`没有这一步**，因为给它加等于改既有产物的指纹
    （三前提第三条）。两条路径在这一点上的差别只出现在精确为零的分量上。
    """

    return value + 0.0


def _plastic_multiplier(
    *,
    along: float,
    across: float,
    semi_along: float,
    semi_across: float,
) -> float:
    """解出塑性乘子``η = k_t·Δγ``，使返回点恰好落在椭圆上。

    ## 参数化

    关联流动``Δu_slip = Δγ·∂Φ/∂f``代进``f = f_trial − k_t·Δu_slip``后逐分量解出：

        f_∥ = A·a²/(a² + η)      f_⊥ = B·b²/(b² + η)

    （``A``、``B``是试探力的两个面内分量，``a = μ_∥N``、``b = μ_⊥N``。）
    **这就是"η-return"里的η**——它是塑性乘子，不是任何一次坐标缩放。

    代回``Φ = 1``得一个关于η的标量方程。它在η ≥ 0上**严格单调下降且凸**
    （两项各自单调下降、各自二阶导为正），故根唯一。

    ## 为什么解的是``1/√Φ − 1``而不是``Φ − 1``

    三个等价形式的根相同，**收敛速度差三倍**（2026-08-17实测均值趟数）：

    | 残差 | 远场行为 | 均值趟数（``μ_∥:μ_⊥ = 5:1``） |
    |---|---|---|
    | ``Φ − 1`` | ``~η⁻²`` | 10.58 |
    | ``√Φ − 1`` | ``~η⁻¹`` | 6.40 |
    | ``1/√Φ − 1`` | **恰好线性** | **4.47** |

    远场``a², b² ≪ η``时``√Φ → √((Aa)²+(Bb)²)/η``，于是第三式是``η/η_hi − 1``——
    **一条直线，牛顿一步到位**。前两式在同一处是``η⁻¹``与``η⁻²``，
    从η=0出发要爬很多步才够得着根。

    ## 括号与初值

    * 上界``η_hi = √((Aa)² + (Bb)²)``：``a²+η ≥ η``且``b²+η ≥ η``，
      故``Φ(η_hi) ≤ 1``。**它在远场恰好收敛到真根**，所以不是一个松上界；
    * 初值套圆的闭式``η₀ = ρ(|f_trial| − ρ)``，``ρ``是椭圆沿``f_trial``方向的半径。
      **圆上它就是精确解**——实测``μ_∥ = μ_⊥``时通用路径恒1趟收敛。

    牛顿步跳出括号就退回二分，故收敛是**有保证的**（括号始终夹住根）。
    """

    #: ``lo``一侧恒有``Φ > 1``、``hi``一侧恒有``Φ ≤ 1``。
    hi = math.sqrt((along * semi_along) ** 2 + (across * semi_across) ** 2)
    lo = 0.0
    magnitude = math.sqrt(along * along + across * across)
    quadratic = (along / semi_along) ** 2 + (across / semi_across) ** 2
    radius = magnitude / math.sqrt(quadratic)
    eta = radius * (magnitude - radius)
    if not (lo < eta < hi):
        eta = 0.5 * (lo + hi)

    for _ in range(_PLASTIC_MAX_ITERATIONS):
        u = semi_along * semi_along + eta
        v = semi_across * semi_across + eta
        phi = (along * semi_along / u) ** 2 + (across * semi_across / v) ** 2
        root = math.sqrt(phi)
        residual = 1.0 / root - 1.0
        if abs(residual) <= _PLASTIC_RESIDUAL_FLOOR:
            return eta
        derivative = (
            (along * semi_along) ** 2 / (u * u * u)
            + (across * semi_across) ** 2 / (v * v * v)
        ) / (root * root * root)
        if residual < 0.0:
            lo = eta
        else:
            hi = eta
        step = eta - residual / derivative
        if not (lo < step < hi):
            step = 0.5 * (lo + hi)
        if step == eta:
            return eta
        eta = step

    raise ContactError(
        f"塑性乘子迭代走满{_PLASTIC_MAX_ITERATIONS}趟仍未落到残差地板"
        f"（a={semi_along!r}, b={semi_across!r}, A={along!r}, B={across!r}）"
        "——实测最坏17趟，走满说明推导或输入超出了已验过的范围"
    )


def anisotropic_return_map(
    *,
    trial_force_n: tuple[float, float, float],
    normal_force_n: float,
    ellipse: FrictionEllipse,
    tangential_stiffness_n_per_mm: float,
) -> FrictionOutcome:
    """各向异性摩擦的return-map：**往椭圆上做最近点投影**，不是沿径向缩回去。

    ## 一、为什么径向缩是错的

    各向同性时三件事**恰好重合**，于是"径向返回"这个名字看起来像是算法本身：

    1. 沿``f_trial``方向缩回圆；
    2. 欧氏度规下到圆的最近点投影；
    3. 最大耗散原理要求的关联流动（滑移增量沿屈服面外法向）。

    椭圆上这三件**互不相同**，而**只有第3件是物理**。
    最大耗散原理（Hill）说：给定实际滑移增量``Δu``，实际切向力在所有
    容许力里使``f·Δu``最大。它的KKT条件正是``Δu = Δγ·∂Φ/∂f``，
    即**滑移增量必须沿屈服面的外法向**。

    椭圆的外法向``∂Φ/∂f ∝ (f_∥/a², f_⊥/b²)``**不平行于**``f``
    （除非``a = b``或``f``落在主轴上）。所以径向缩给出的滑移方向是错的，
    实测最大偏差``|sin| = 0.923``，即**67.4°**（``μ_∥:μ_⊥ = 5:1``，逐度扫）。

    ## 二、"把椭圆拉成圆、径向返回、再变回来"**就是径向缩**

    这一条是本轮实测出来的，写在这里是因为它是个很容易踩的推理：

        g = S·f（S = diag(1/a, 1/b)）；圆上径向返回 g → g/|g|；回去 f = S⁻¹g

    展开即``f = f_trial/|S·f_trial|``——**一个纯标量乘在**``f_trial``**上**，
    方向一个字没变。逐分量实测两条路径的最大差**8.88e-16**（纯舍入）。

    **度规变换不改变"沿哪个方向返回"，它只改变"缩多少"。**
    要让最近点投影变成径向，需要``T·S² = c·T``，即``S² ∝ I``——只有各向同性。
    **不存在这样的线性变换**，所以这条捷径不是"近似"，是**同一个错映射的另一种写法**。

    ## 三、真的算法：沿η族返回

    最近点投影在欧氏度规下等价于关联流动（因为切向弹性刚度是各向同性的
    ``k_t·I``——**这一步依赖它**，若哪天切向刚度也各向异性，这句话要重推）。
    解法见`_plastic_multiplier`：一个凸单调的标量方程，括号二分兜底的牛顿，
    实测最坏17趟、均值约3趟。

    ## 四、退化是**构造出来的**，不是算出来的

    ``μ_∥ == μ_⊥``（逐位相等）时本函数**把参数原样转交**`coulomb_return_map`，
    连校验与报错都是它的。所以"既有案例的指纹不变"不需要任何数值论证。

    通用路径本身也退化得干净（关掉这条捷径实测：``μ_∥ = μ_⊥``时
    与圆的闭式最大相对逐分量差**2.96e-16**，且恒1趟收敛）——
    那条由`test_the_general_path_also_degenerates`守着，
    **两条一起才叫"退化被验过"**：捷径保证逐位，通用路径保证捷径没有掩盖错误。

    ## 五、边界

    * ``N = 0``（分离）→ `REGIME_SEPARATED`，与各向同性同口径
      （由转交路径或本函数各自给出，两处约定必须一致）；
    * 试探力的**法向分量**超过``TRIAL_OUT_OF_PLANE_TOLERANCE``即报错。
      摩擦住在切平面里；带法向分量的"切向试探力"是调用方忘了投影，
      静默扔掉它就是又一个静默错值入口；
    * 屈服面边界（``Φ = 1``恰好成立）判**粘**，与各向同性的``|T| ≤ μN``同口径。
    """

    #: **先转交，再校验**。转交路径必须连报错行为都是`coulomb_return_map`的，
    #: 否则"逐位相同"就只覆盖了正常路径而漏掉了异常路径。
    if ellipse.is_isotropic:
        return coulomb_return_map(
            trial_force_n=trial_force_n,
            normal_force_n=normal_force_n,
            friction_coefficient=ellipse.mu_along,
            tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
        )

    if len(trial_force_n) != 3 or not all(math.isfinite(v) for v in trial_force_n):
        raise ContactError(f"trial force must be a finite 3-vector: {trial_force_n!r}")
    if normal_force_n < 0.0 or not math.isfinite(normal_force_n):
        raise ContactError(f"normal force must be finite and nonnegative: {normal_force_n!r}")
    if not (tangential_stiffness_n_per_mm > 0.0 and math.isfinite(tangential_stiffness_n_per_mm)):
        raise ContactError(
            f"tangential stiffness must be positive: {tangential_stiffness_n_per_mm!r}"
        )

    zero = (0.0, 0.0, 0.0)
    if normal_force_n == 0.0:
        return FrictionOutcome(zero, REGIME_SEPARATED, zero)

    normal = ellipse.normal
    out_of_plane = sum(trial_force_n[axis] * normal[axis] for axis in range(3))
    trial_magnitude = math.sqrt(sum(value * value for value in trial_force_n))
    if abs(out_of_plane) > TRIAL_OUT_OF_PLANE_TOLERANCE * trial_magnitude:
        raise ContactError(
            f"试探力有{out_of_plane!r} N的法向分量（|f_trial| = {trial_magnitude!r} N）"
            "——摩擦住在切平面里，这说明调用方没扣掉法向"
        )

    axis_along, axis_across = ellipse.in_plane_axes()
    along = sum(trial_force_n[axis] * axis_along[axis] for axis in range(3))
    across = sum(trial_force_n[axis] * axis_across[axis] for axis in range(3))

    semi_along = ellipse.mu_along * normal_force_n
    semi_across = ellipse.mu_across * normal_force_n
    quadratic = (along / semi_along) ** 2 + (across / semi_across) ** 2
    if quadratic <= 1.0:
        #: 粘：**报回投影后的面内力**而不是入参。两者只差被扣掉的法向分量，
        #: 而那一截刚被判定为可忽略——报入参等于把它偷偷留在切向力里。
        force = tuple(
            _positive_zero(along * axis_along[axis] + across * axis_across[axis])
            for axis in range(3)
        )
        return FrictionOutcome(force, REGIME_STICK, zero)

    eta = _plastic_multiplier(
        along=along,
        across=across,
        semi_along=semi_along,
        semi_across=semi_across,
    )
    returned_along = along * semi_along * semi_along / (semi_along * semi_along + eta)
    returned_across = across * semi_across * semi_across / (semi_across * semi_across + eta)
    force = tuple(
        _positive_zero(
            returned_along * axis_along[axis] + returned_across * axis_across[axis]
        )
        for axis in range(3)
    )
    #: 锚点修正就是``(f_trial − f)/k_t``。**它自动沿外法向**——那是最近点投影的
    #: 定义性质，不是这里额外补的一步。判它的门在`tests/test_contact_friction_anisotropic.py`。
    slip_along = (along - returned_along) / tangential_stiffness_n_per_mm
    slip_across = (across - returned_across) / tangential_stiffness_n_per_mm
    correction = tuple(
        _positive_zero(
            slip_along * axis_along[axis] + slip_across * axis_across[axis]
        )
        for axis in range(3)
    )
    return FrictionOutcome(force, REGIME_SLIP, correction)


@dataclass(frozen=True)
class TangentialStickSpring:
    """粘着弹簧：``U = Σ ½·k_t·|P(x − a)|²``，``P = I − n⊗n``是切平面投影。单位N·mm。

    ## 它为什么**是**能量项，而滑移不是

    0033调研的结论：带粘着的库仑摩擦把切向相对位移**分解成可逆的"粘"分量与
    不可逆的"滑"分量**（与塑性力学的形制是同一个）。

    **可逆的那一半是弹性的，因此有势函数**——就是这个类。
    不可逆的那一半（滑移）耗散、非associative，**写不成任何位置的势**——
    那是`coulomb_return_map`。

    这就是0050第二节"接触是**半个能量项**"那句话的具体形状：
    法向 + 粘着在能量里（进`EnergyRegistry`、进牛顿的切线刚度），
    滑移在return-map里（改锚点，即改状态里的历史）。
    **把这条分界写在这里，是因为下一个人最可能犯的错是想把整个摩擦写成势能。**

    ## 它给对了什么

    与法向项同构：平衡时``k_t·|Δ| = T_理论``，于是**切向力精确、切向位移是``O(1/k_t)``**。
    斜面上实测：``T = W·sinθ``与``k_t``无关。

    ## 锚点是**输入**，不是这里算出来的

    本项读锚点，不写锚点。写锚点的是return-map——那一步才是历史发生的地方。
    锚点住在状态向量的槽位里（`ContactLayout`），本项从调用方拿到它的值。
    """

    name: str = "tangential_stick"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点索引, 锚点mm, 面法向单位矢量, 切向刚度N/mm)
    springs: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float], float], ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.springs:
            raise ContactError("tangential_stick needs at least one spring")
        for node, anchor, normal, stiffness in self.springs:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"stick node index must be a nonnegative int: {node!r}")
            if len(anchor) != 3 or not all(math.isfinite(value) for value in anchor):
                raise ContactError(f"anchor must be a finite 3-vector: {anchor!r}")
            #: **单位矢量那道门挡不住nan**：``abs(nan − 1.0) > tol``是``False``，
            #: 于是nan法向一路通过、能量与梯度全变nan（2026-08-06对抗审核实测）。
            #: 同门的`PenaltyNormalContact`有这两条检查，这里此前没有。
            if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
                raise ContactError(f"stick normal must be a finite 3-vector: {normal!r}")
            norm = math.sqrt(sum(component * component for component in normal))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(f"stick normal must be a unit vector (|n| = {norm!r})")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"tangential stiffness must be positive: {stiffness!r}")

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _ in self.springs) + 1

    @staticmethod
    def _tangential_offset_mm(
        vector: tuple[float, ...],
        node: int,
        anchor: tuple[float, float, float],
        normal: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """``P(x − a)``：位移里**扣掉法向分量**的那一部分。

        扣掉法向是这个项的全部要害：不扣，粘着弹簧会连法向也一起拉，
        于是它与法向罚函数**重复计入法向刚度**，而两者的刚度通常差好几个数量级——
        结果是法向力悄悄变成``(k_n + k_t)·δ``，**而`normal_force_n`报的仍是``k_n·δ``**。
        """

        base = 3 * node
        delta = tuple(vector[base + axis] - anchor[axis] for axis in range(3))
        along_normal = sum(delta[axis] * normal[axis] for axis in range(3))
        return tuple(delta[axis] - along_normal * normal[axis] for axis in range(3))

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(state.vector, node, anchor, normal)
            total += 0.5 * stiffness * sum(value * value for value in offset)
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(state.vector, node, anchor, normal)
            base = 3 * node
            for axis in range(3):
                result[base + axis] += stiffness * offset[axis]
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        size = len(state.vector)
        result = [[0.0] * size for _ in range(size)]
        for row, column, value in self.hessian_entries(state, context):
            result[row][column] += value
        return tuple(tuple(row) for row in result)

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        """``k_t·(I − n⊗n)``。**常量**——粘着是线性的，这是它比滑移好对付的原因。"""

        entries: list[tuple[int, int, float]] = []
        for node, _, normal, stiffness in self.springs:
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    value = stiffness * ((1.0 if a == b else 0.0) - normal[a] * normal[b])
                    entries.append((base + a, base + b, value))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(vector, node, anchor, normal)
            total += 0.5 * stiffness * sum(value * value for value in offset)
            if gradient is not None:
                base = 3 * node
                for axis in range(3):
                    gradient[base + axis] += stiffness * offset[axis]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def tangential_force_n(self, state: State) -> tuple[tuple[float, float, float], ...]:
        """每根弹簧的切向力矢量``k_t·P(x − a)``——**摩擦锥要判的就是它的模**。"""

        return tuple(
            tuple(
                stiffness * value
                for value in self._tangential_offset_mm(state.vector, node, anchor, normal)
            )
            for node, anchor, normal, stiffness in self.springs
        )


__all__ = [
    "FrictionOutcome",
    "TangentialStickSpring",
    "coulomb_return_map",
]
