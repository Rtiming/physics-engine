"""时间推进——spec/12第四节的参考实现（T4，引擎第一次能推进物理状态）。

**这是本包第一块会算物理的代码。** 在它之前引擎只有声明、校验、溯源与几何量。

三条形制条款落在这里：

* **一份公式源，两个求值后端**（spec/12第5.2节，0016甲案）：积分公式只写一遍，
  按`VectorOps`求值。纯Python后端与NumPy加速档因此**逐字节一致是构造保证的**，
  不是碰巧对上——两边执行的是同一串运算、同一个次序。按spec/12第5.3节
  "能逐字节就必须逐字节"，本模块的对拍档位是逐字节，不是容差。
* **积分器出生带五项声明**（spec/12第4.2节，缺一不得进仓）：适用域、
  形式阶与实测阶分开、稳定性分类、耗散记账、失败阶梯。
* **显式积分器不写步长上界就是埋雷**——三个积分器的上界都在声明里。

**本模块不管什么**：无接触、无约束、无隐式求解。隐式族（Newmark、隐式中点、
generalized-α）随WDS内核搬迁进来，不在本片。二阶系统专用：状态是位置与速度，
力由调用方给。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Vector = tuple[float, ...]

#: 加速度回调：``a(x, v, t) -> 加速度``。它是**调用方的代码**，不在加速档边界内
#: （两个后端都用元组调它）——所以今天加速档只向量化积分器自己的算术。
#: 这是有意的诚实边界：真正值得加速的是能量装配，那随内核搬迁进来时再谈。
Acceleration = Callable[[Vector, Vector, float], Sequence[float]]
#: 物理耗散率回调，单位N·mm/s；返回值必须有限且非负。
DissipationRate = Callable[[Vector, Vector, float], float]


class IntegrateError(ValueError):
    """时间推进的一切失败关闭。"""


class VectorOps(Protocol):
    """数值命名空间（spec/13规则2：收窄到可替换）。"""

    name: str

    def scale(self, factor: float, vector: object) -> object: ...
    def add(self, left: object, right: object) -> object: ...
    def load(self, values: Vector) -> object: ...
    def dump(self, vector: object) -> Vector: ...


class PurePythonOps:
    """纯Python后端。零依赖，永远可用（0014零设施承诺）。"""

    name = "pure_python"

    def scale(self, factor, vector):
        return tuple(factor * value for value in vector)

    def add(self, left, right):
        return tuple(a + b for a, b in zip(left, right, strict=True))

    def load(self, values):
        return tuple(values)

    def dump(self, vector):
        return tuple(vector)


class NumpyOps:
    """NumPy加速档。**只换求值方式，不换数学**（spec/13规则1）。

    逐元素IEEE 754 float64运算、无归约、无次序重排，因此与纯Python后端
    **逐位相同**；确定性等级按spec/13规则5申报为``bitwise``。
    """

    name = "numpy"
    determinism = "bitwise"

    def __init__(self) -> None:
        import numpy  # 只在构造加速档时import——核心路径永不要求NumPy

        self._numpy = numpy

    def scale(self, factor, vector):
        return factor * vector

    def add(self, left, right):
        return left + right

    def load(self, values):
        return self._numpy.array(values, dtype=self._numpy.float64)

    def dump(self, vector):
        return tuple(float(value) for value in vector)


def default_ops() -> VectorOps:
    """核心默认纯Python。加速档要显式选，不做静默切换——静默切换等于把
    "结果从哪来"藏起来，与轴3的复现纪律冲突。"""

    return PurePythonOps()


@dataclass(frozen=True)
class IntegratorDeclaration:
    """出生五项声明（spec/12第4.2节）。缺一构造即拒。"""

    name: str
    #: 一句话说清它**不管**什么。
    scope_excludes: str
    formal_order: int
    #: 实测阶单列——形式二阶而实测1.4的事在WDS发生过，不许混写。
    measured_order: str
    stability: Literal["explicit_conditional", "symplectic", "implicit_unconditional"]
    #: 条件稳定的必须给步长上界表达式；无条件稳定的写"无"。
    step_bound: str
    #: 有算法耗散必须显式记账，不许一边耗散一边宣称守恒。
    dissipation_accounting: str
    #: 有没有步长拒绝/二分；没有就写没有（失败关闭v1也要说出来）。
    failure_ladder: str
    production_ready: bool
    #: `step_bound`的**可计算部分**：让线性振子能量保持有界的最大``h·ω``。
    #:
    #: 立它的理由（决策0052第五节）：`step_bound`是**字符串**，
    #: 而步长顾问要算数。**一个字段同时干两件事，结果是两件都干不好**——
    #: 在它之前，"这个积分器的步长上界是多少"只能靠人读一句话。
    #:
    #: **``None``表示顾问拒绝为这个积分器建议步长**，理由写在`step_bound`里。
    #: 它不是"忘了填"——`explicit_euler`就是`None`，因为**实测它在任何步长下都发散**。
    #:
    #: 值取自**本仓实测**，不取自文献：2026-08-12在线性振子上跑40个周期，
    #: 逐档量能量比。见`OSCILLATORY_COEFFICIENT_EVIDENCE`。
    oscillatory_step_coefficient: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "scope_excludes", "measured_order", "step_bound",
            "dissipation_accounting", "failure_ladder",
        ):
            if not str(getattr(self, field_name)).strip():
                raise IntegrateError(
                    f"integrator {self.name!r} must declare {field_name} "
                    "— 五项声明缺一不得进仓（spec/12第4.2节）"
                )
        if self.stability == "explicit_conditional" and self.step_bound == "无":
            raise IntegrateError(
                f"{self.name!r} is conditionally stable but declares no step bound "
                "— 显式积分器不写步长上界就是埋雷"
            )
        coefficient = self.oscillatory_step_coefficient
        if coefficient is not None and not (coefficient > 0.0):
            raise IntegrateError(
                f"{self.name!r} declares oscillatory_step_coefficient={coefficient!r} "
                "— 它是`h·ω`的上界，必须为正；'没有可用的界'请写None并在step_bound说明"
            )


@dataclass(frozen=True)
class Integrator:
    declaration: IntegratorDeclaration
    step: Callable[[Vector, Vector, float, float, Acceleration, VectorOps],
                   tuple[Vector, Vector]]


@dataclass(frozen=True)
class DissipativeIntegrationResult:
    """带物理耗散账的推进结果；旧``integrate``三元组保持原样。"""

    x: Vector
    v: Vector
    t_s: float
    #: 按端点梯形求积累计的物理耗散，单位N·mm。它不掩盖算法误差；
    #: 调用方仍应以``初始机械能-末态机械能-本值``检查数值残差。
    dissipated_energy_nmm: float


def _explicit_euler_step(x, v, t, h, acceleration, ops):
    a = ops.load(acceleration(x, v, t))
    xv, vv = ops.load(x), ops.load(v)
    # 先用**旧**速度推位置——这一步的次序就是它与半隐式的全部差别。
    next_x = ops.add(xv, ops.scale(h, vv))
    next_v = ops.add(vv, ops.scale(h, a))
    return ops.dump(next_x), ops.dump(next_v)


def _symplectic_euler_step(x, v, t, h, acceleration, ops):
    a = ops.load(acceleration(x, v, t))
    xv, vv = ops.load(x), ops.load(v)
    # 先更新速度，再用**新**速度推位置。
    next_v = ops.add(vv, ops.scale(h, a))
    next_x = ops.add(xv, ops.scale(h, next_v))
    return ops.dump(next_x), ops.dump(next_v)


def _velocity_verlet_step(x, v, t, h, acceleration, ops):
    a0 = ops.load(acceleration(x, v, t))
    xv, vv = ops.load(x), ops.load(v)
    # x + v·h + a·h²/2 —— 常加速度下这一式是**精确**的，误差只剩浮点噪声。
    next_x = ops.add(ops.add(xv, ops.scale(h, vv)), ops.scale(0.5 * h * h, a0))
    next_x_t = ops.dump(next_x)
    a1 = ops.load(acceleration(next_x_t, ops.dump(vv), t + h))
    next_v = ops.add(vv, ops.scale(0.5 * h, ops.add(a0, a1)))
    return next_x_t, ops.dump(next_v)


def _velocity_verlet_damped_step(x, v, t, h, acceleration, ops):
    """速度Verlet的显式预测-校正扩展，速度相关加速度下保持二阶。

    位置预测仍是``x + hv + h²a/2``；末端加速度使用一阶速度预测
    ``v* = v + ha``。当加速度与速度无关时，预测速度不参与求值，运算退化为
    原``velocity_verlet``的同一串公式；原积分器对象与注册名仍保持不动。
    """

    a0 = ops.load(acceleration(x, v, t))
    xv, vv = ops.load(x), ops.load(v)
    next_x = ops.add(ops.add(xv, ops.scale(h, vv)), ops.scale(0.5 * h * h, a0))
    next_x_t = ops.dump(next_x)
    predicted_v = ops.add(vv, ops.scale(h, a0))
    a1 = ops.load(acceleration(next_x_t, ops.dump(predicted_v), t + h))
    next_v = ops.add(vv, ops.scale(0.5 * h, ops.add(a0, a1)))
    return next_x_t, ops.dump(next_v)


EXPLICIT_EULER = Integrator(
    declaration=IntegratorDeclaration(
        name="explicit_euler",
        scope_excludes="不管刚性问题、不管接触、不管长时间守恒；只作教学案例与漂移排序的对照组",
        formal_order=1,
        measured_order="1（本仓B档实测，常加速度误差恰为−a·T·h/2）",
        stability="explicit_conditional",
        step_bound=(
            "**振荡问题上没有可用的步长上界**。2/ω_max是实轴稳定区半径，"
            "而线性振子的特征值在虚轴上，显式Euler的放大因子恒为"
            "sqrt(1+h²ω²) > 1——**任何h都增长**。"
            "本仓实测（2026-08-12，40个周期）：h·ω=0.1时能量已涨到7.2e10倍，"
            "h·ω=2.0时1.2e88倍。**此前这一格写的是"
            "'h < 2/ω_max（线性振子）'，读起来像个可用的界，那是误导**"
        ),
        dissipation_accounting="无算法耗散；能量**单调增长**（反耗散），这是它被排除在生产之外的原因",
        failure_ladder="无步长拒绝阶梯",
        production_ready=False,
        #: None不是忘了填：实测任何步长都发散，顾问必须拒绝为它建议步长。
        oscillatory_step_coefficient=None,
    ),
    step=_explicit_euler_step,
)

SYMPLECTIC_EULER = Integrator(
    declaration=IntegratorDeclaration(
        name="symplectic_euler",
        scope_excludes="不管刚性问题、不管接触；一阶精度，只适合定性演示",
        formal_order=1,
        measured_order="1（常加速度误差恰为+a·T·h/2，与显式同幅反号）",
        stability="symplectic",
        step_bound=(
            "**稳定界**：h < 2/ω_max（辛但仍条件稳定）。"
            "本仓实测（2026-08-12，40个周期）h·ω=2.00时能量比1.28e5、"
            "2.01时2.8e23——边界在2.0上。"
            "**接触分辨界比稳定界紧一个数量级**：要分辨接触时长t_c=π/ω需"
            "h ≤ π/(N·ω)，N=20时约为稳定界的1/12.7。"
            "plans/08实测N=2（h·ω≈1.57）时恢复系数已错14.3%，N=20时降到1.6e-4。"
            "**用`advise_step()`算，不要读这句话估**"
        ),
        dissipation_accounting="无算法耗散；能量有界振荡不漂移",
        failure_ladder="无步长拒绝阶梯",
        production_ready=False,
        oscillatory_step_coefficient=2.0,
    ),
    step=_symplectic_euler_step,
)

VELOCITY_VERLET = Integrator(
    declaration=IntegratorDeclaration(
        name="velocity_verlet",
        scope_excludes="不管接触、不管强耗散、不管刚性问题；速度相关的力（阻尼）下不再是二阶",
        formal_order=2,
        measured_order="2（本仓B档实测cos(ωT)误差比3.9985与3.9996，渐近趋4）",
        stability="symplectic",
        step_bound=(
            "**稳定界**：h < 2/ω_max。本仓实测（2026-08-12，40个周期）"
            "h·ω=2.00时能量比1.0000（临界有界）、2.01时1.3e21。"
            "**接触分辨界比稳定界紧一个数量级**：h ≤ π/(N·ω)，"
            "N=20时约为稳定界的1/12.7。plans/08实测N=2时恢复系数错14.3%。"
            "**用`advise_step()`算，不要读这句话估**"
        ),
        dissipation_accounting="无算法耗散；长时间能量有界",
        failure_ladder="无步长拒绝阶梯",
        production_ready=False,
        oscillatory_step_coefficient=2.0,
    ),
    step=_velocity_verlet_step,
)

VELOCITY_VERLET_DAMPED = Integrator(
    declaration=IntegratorDeclaration(
        name="velocity_verlet_damped",
        scope_excludes=(
            "不管刚性约束、不做碰撞时刻定位、不做自适应拒步；"
            "用于速度相关耗散力时必须同时接耗散记账与步长顾问"
        ),
        formal_order=2,
        measured_order=(
            "2（本仓2026-08-13在ζ=0.2线性阻尼振子上实测，步长减半位置误差比"
            "4.036/4.018/4.009）"
        ),
        stability="explicit_conditional",
        step_bound=(
            "全阻尼范围的保守稳定界h·ω_fast < 1；最紧点ζ=1在本仓实测"
            "h·ω0=0.995衰减、1.005发散。ζ>1的ω_fast取"
            "(ζ+sqrt(ζ²−1))·ω0；接触分辨界另由实际接触时长/N给出。"
            "用advise_step()算，不要从本字符串抄数"
        ),
        dissipation_accounting=(
            "物理耗散由integrate_with_dissipation按耗散率端点梯形累计；"
            "算法截断误差不冒充物理耗散，须另查机械能平衡残差"
        ),
        failure_ladder="无自适应拒步；非有限或负耗散率立即失败关闭",
        production_ready=False,
        #: ζ=1是ζ∈[0,∞)按ω_fast归一后的最紧点；测试在1两侧实跑。
        oscillatory_step_coefficient=1.0,
    ),
    step=_velocity_verlet_damped_step,
)

INTEGRATORS: dict[str, Integrator] = {
    integrator.declaration.name: integrator
    for integrator in (
        EXPLICIT_EULER,
        SYMPLECTIC_EULER,
        VELOCITY_VERLET,
        VELOCITY_VERLET_DAMPED,
    )
}


# ---------------------------------------------------------------------------
# 步长顾问（决策0052第五节）
# ---------------------------------------------------------------------------

#: `oscillatory_step_coefficient`各值的出处——**本仓实测，不是引文**。
#:
#: 方法：线性振子`ẍ = −ω²x`，`ω=1`，初条件`x=1, v=0`，跑40个周期，
#: 量末态与初态的能量比。2026-08-12实测：
#:
#: | 积分器 | h·ω=0.1 | h·ω=1.99 | h·ω=2.00 | h·ω=2.01 |
#: |---|---|---|---|---|
#: | `explicit_euler` | **7.2e10** | 4.3e87 | 1.2e88 | 6.4e87 |
#: | `symplectic_euler` | 0.992 | 3.74 | **1.3e5** | 2.8e23 |
#: | `velocity_verlet` | 1.000 | 0.994 | **1.0000** | 1.3e21 |
#:
#: 两条读出来的结论：
#:
#: 1. 辛的两个边界**正好在2.0**（2.00有界、2.01发散）；
#: 2. `explicit_euler`**在任何步长下都发散**——`h·ω=0.1`就已经涨到7.2e10倍。
#:    2/ω是**实轴**稳定区半径，而线性振子的特征值在**虚轴**上，
#:    显式Euler的放大因子恒为`sqrt(1+h²ω²) > 1`。
#:    **它此前那句`h < 2/ω_max（线性振子）`读起来像个可用的界，那是误导。**
OSCILLATORY_COEFFICIENT_EVIDENCE = (
    "2026-08-12本仓实测：线性振子40周期能量比。"
    "symplectic_euler与velocity_verlet在h·ω=2.00有界、2.01发散；"
    "explicit_euler在h·ω=0.1已发散（7.2e10倍），故其系数为None。"
)

#: 默认每次接触要走的步数。
#:
#: 取20不是取整：plans/08实测**2步/接触时恢复系数错14.3%**、20步时1.6e-4、
#: 40步时8.0e-6；而O'Sullivan & Bray 2004的`Δt ≤ 0.17·sqrt(m/K)`换算过来
#: 约18.5步/接触——**同行经验数与本仓实测落在同一格**。
DEFAULT_STEPS_PER_CONTACT = 20

#: 低于这个步数就不只是"不准"，是**定性错**：plans/08实测2步/接触时
#: 恢复系数是1.1433（理论1），**大于1**——积分误差把能量喂进了碰撞。
MIN_MEANINGFUL_STEPS_PER_CONTACT = 4


@dataclass(frozen=True)
class StepAdvice:
    """步长建议。**两个界并排给出，并说明哪个在管事**。

    只给一个数会让使用者不知道"再放松一点会先撞上什么"——
    而这两个界的失败方式完全不同：撞稳定界是**爆掉**，
    撞分辨界是**静默地算出一个错的恢复系数**。
    """

    #: 稳定界：`coefficient / ω_max`。超过它数值解发散。
    stability_bound_s: float
    #: 接触分辨界：`π / (steps_per_contact · ω_max)`。超过它恢复系数静默错。
    contact_resolution_bound_s: float
    #: 建议值 = 两者取小。
    advised_step_s: float
    #: 哪一条在管事。
    binding: Literal["stability", "contact_resolution"]
    steps_per_contact: int
    omega_max_rad_per_s: float
    #: 分辨界使用的接触时长；无显式传入时沿用无阻尼``π/ω_max``。
    contact_duration_s: float
    #: 建议值相对稳定界的倍数——用来一眼看出"离爆掉还有多远"。
    stability_margin: float


def advise_step(
    omega_max_rad_per_s: float,
    *,
    oscillatory_step_coefficient: float,
    steps_per_contact: int = DEFAULT_STEPS_PER_CONTACT,
    contact_duration_s: float | None = None,
) -> StepAdvice:
    """把`step_bound`那个**字符串**变成可计算的数。

    **只吃纯数字**（决策0052第五节）：不收接触对象、不收场景、
    不import包内任何东西。本模块今天的包内import是**0**，
    而那条独立性意味着积分器可以被单独拿走用——顾问不该是破它的那个。

    代价如实登记：`ω_max`由调用方自己算，**算错了顾问看不出来**。
    缓解只有一条——对非正/非有限的入参失败关闭。
    算`ω_max`的辅助放在接触那边，不放这里。

    参数
    ----
    omega_max_rad_per_s
        系统最高频模态的角频率。接触上是`sqrt(k_eff/m_eff)`。
    oscillatory_step_coefficient
        取自`integrator.declaration.oscillatory_step_coefficient`。
        该字段为``None``的积分器**不能被建议步长**——
        `explicit_euler`就是这种，实测它在任何步长下都发散。
    steps_per_contact
        每次接触要走的步数，默认20（见`DEFAULT_STEPS_PER_CONTACT`）。
    """

    if not math.isfinite(omega_max_rad_per_s) or omega_max_rad_per_s <= 0.0:
        raise IntegrateError(
            f"omega_max_rad_per_s must be finite and positive, got "
            f"{omega_max_rad_per_s!r} — 零频或非有限频不是'不限步长'，是没有定义"
        )
    if oscillatory_step_coefficient is None:
        raise IntegrateError(
            "这个积分器的oscillatory_step_coefficient是None——"
            "**它没有可用的步长上界**，顾问拒绝建议。理由见它的step_bound。"
            "（`explicit_euler`就是这种：实测h·ω=0.1时40周期能量已涨7.2e10倍）"
        )
    if not math.isfinite(oscillatory_step_coefficient) or oscillatory_step_coefficient <= 0.0:
        raise IntegrateError(
            f"oscillatory_step_coefficient must be finite and positive, got "
            f"{oscillatory_step_coefficient!r}"
        )
    if steps_per_contact < MIN_MEANINGFUL_STEPS_PER_CONTACT:
        raise IntegrateError(
            f"steps_per_contact={steps_per_contact} < {MIN_MEANINGFUL_STEPS_PER_CONTACT} "
            "— 低于这个数不只是不准，是定性错：plans/08实测2步/接触时"
            "恢复系数1.1433（理论1），**大于1**，积分误差把能量喂进了碰撞"
        )
    if contact_duration_s is not None and (
        not math.isfinite(contact_duration_s) or contact_duration_s <= 0.0
    ):
        raise IntegrateError(
            f"contact_duration_s must be finite and positive, got {contact_duration_s!r}"
        )

    stability = oscillatory_step_coefficient / omega_max_rad_per_s
    duration = (
        contact_duration_s
        if contact_duration_s is not None
        else math.pi / omega_max_rad_per_s
    )
    resolution = duration / steps_per_contact
    advised = min(stability, resolution)
    return StepAdvice(
        stability_bound_s=stability,
        contact_resolution_bound_s=resolution,
        advised_step_s=advised,
        binding="stability" if stability <= resolution else "contact_resolution",
        steps_per_contact=steps_per_contact,
        omega_max_rad_per_s=omega_max_rad_per_s,
        contact_duration_s=duration,
        stability_margin=advised / stability,
    )


#: `production_ready`翻成True的**四条条件，全中才翻**（决策0052第六节）。
#:
#: 立它的理由：不定条件，它就是个**永远翻不了的死字段**。
#: 本仓立过同源的规矩（0039："绊线一旦长期不响就等于被拆了"）。
#: 定了条件而2026-08-12这一轮不翻，是因为四条今天**一条都没全中**。
PRODUCTION_READY_CONDITIONS = (
    "实测阶与声明阶一致（有case，不是有注释）",
    "step_bound已是可计算的数（oscillatory_step_coefficient非None），且步长顾问覆盖它",
    "至少一个第1档解析判据案例走完state→energies→solve或state→integrate",
    "能量行为已被测过并如实声明——罚接触+显式积分会涨能量（Acary ZAMM 2016），"
    "**不许把'没测'写成'守恒'**",
)


def integrate(
    integrator: Integrator,
    *,
    x0: Sequence[float],
    v0: Sequence[float],
    dt_s: float,
    steps: int,
    acceleration: Acceleration,
    t0_s: float = 0.0,
    ops: VectorOps | None = None,
) -> tuple[Vector, Vector, float]:
    """推进``steps``步，返回``(位置, 速度, 末时刻)``。

    步长与步数是**显式的**，不做自适应——步长二分是抢救不是收敛证据
    （spec/12第4.3节），要自适应就得先有失败阶梯的声明。
    """

    if steps < 0:
        raise IntegrateError("steps must be nonnegative")
    if not (dt_s > 0.0):
        raise IntegrateError("dt_s must be positive — 零步长不是不动，是没有定义")
    if len(x0) != len(v0):
        raise IntegrateError("x0 and v0 must have the same length")
    backend = ops or default_ops()
    x, v, t = tuple(float(value) for value in x0), tuple(float(value) for value in v0), t0_s
    for _ in range(steps):
        x, v = integrator.step(x, v, t, dt_s, acceleration, backend)
        t += dt_s
    return x, v, t


def integrate_with_dissipation(
    integrator: Integrator,
    *,
    x0: Sequence[float],
    v0: Sequence[float],
    dt_s: float,
    steps: int,
    acceleration: Acceleration,
    dissipation_rate: DissipationRate,
    t0_s: float = 0.0,
    ops: VectorOps | None = None,
) -> DissipativeIntegrationResult:
    """推进并累计**物理**耗散，旧``integrate``的返回形制一个字节不动。

    耗散率在每步两端求值并作梯形积分。它与二阶状态推进同阶；但接触启停有分段
    光滑点，调用方仍须报告能量平衡残差，不能把本数当作“算法绝对守恒”的证明。
    """

    if steps < 0:
        raise IntegrateError("steps must be nonnegative")
    if not (dt_s > 0.0):
        raise IntegrateError("dt_s must be positive — 零步长不是不动，是没有定义")
    if len(x0) != len(v0):
        raise IntegrateError("x0 and v0 must have the same length")

    backend = ops or default_ops()
    x = tuple(float(value) for value in x0)
    v = tuple(float(value) for value in v0)
    t = t0_s
    dissipated = 0.0
    if steps == 0:
        return DissipativeIntegrationResult(x, v, t, dissipated)

    def checked_rate(at_x: Vector, at_v: Vector, at_t: float) -> float:
        rate = float(dissipation_rate(at_x, at_v, at_t))
        if not math.isfinite(rate) or rate < 0.0:
            raise IntegrateError(
                f"dissipation rate must be finite and nonnegative, got {rate!r}"
            )
        return rate

    rate0 = checked_rate(x, v, t)
    for _ in range(steps):
        next_x, next_v = integrator.step(x, v, t, dt_s, acceleration, backend)
        next_t = t + dt_s
        rate1 = checked_rate(next_x, next_v, next_t)
        dissipated += 0.5 * dt_s * (rate0 + rate1)
        x, v, t, rate0 = next_x, next_v, next_t, rate1
    return DissipativeIntegrationResult(x, v, t, dissipated)


__all__ = [
    "DEFAULT_STEPS_PER_CONTACT",
    "EXPLICIT_EULER",
    "INTEGRATORS",
    "SYMPLECTIC_EULER",
    "VELOCITY_VERLET",
    "VELOCITY_VERLET_DAMPED",
    "Acceleration",
    "DissipationRate",
    "DissipativeIntegrationResult",
    "IntegrateError",
    "Integrator",
    "IntegratorDeclaration",
    "MIN_MEANINGFUL_STEPS_PER_CONTACT",
    "NumpyOps",
    "OSCILLATORY_COEFFICIENT_EVIDENCE",
    "PRODUCTION_READY_CONDITIONS",
    "PurePythonOps",
    "StepAdvice",
    "VectorOps",
    "advise_step",
    "default_ops",
    "integrate",
    "integrate_with_dissipation",
]
