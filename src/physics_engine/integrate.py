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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Vector = tuple[float, ...]

#: 加速度回调：``a(x, v, t) -> 加速度``。它是**调用方的代码**，不在加速档边界内
#: （两个后端都用元组调它）——所以今天加速档只向量化积分器自己的算术。
#: 这是有意的诚实边界：真正值得加速的是能量装配，那随内核搬迁进来时再谈。
Acceleration = Callable[[Vector, Vector, float], Sequence[float]]


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


@dataclass(frozen=True)
class Integrator:
    declaration: IntegratorDeclaration
    step: Callable[[Vector, Vector, float, float, Acceleration, VectorOps],
                   tuple[Vector, Vector]]


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


EXPLICIT_EULER = Integrator(
    declaration=IntegratorDeclaration(
        name="explicit_euler",
        scope_excludes="不管刚性问题、不管接触、不管长时间守恒；只作教学案例与漂移排序的对照组",
        formal_order=1,
        measured_order="1（本仓B档实测，常加速度误差恰为−a·T·h/2）",
        stability="explicit_conditional",
        step_bound="h < 2/ω_max（线性振子）；一般系统由最高频模态定",
        dissipation_accounting="无算法耗散；能量**单调增长**（反耗散），这是它被排除在生产之外的原因",
        failure_ladder="无步长拒绝阶梯",
        production_ready=False,
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
        step_bound="h < 2/ω_max（辛但仍条件稳定）",
        dissipation_accounting="无算法耗散；能量有界振荡不漂移",
        failure_ladder="无步长拒绝阶梯",
        production_ready=False,
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
        step_bound="h < 2/ω_max",
        dissipation_accounting="无算法耗散；长时间能量有界",
        failure_ladder="无步长拒绝阶梯",
        production_ready=False,
    ),
    step=_velocity_verlet_step,
)

INTEGRATORS: dict[str, Integrator] = {
    integrator.declaration.name: integrator
    for integrator in (EXPLICIT_EULER, SYMPLECTIC_EULER, VELOCITY_VERLET)
}


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


__all__ = [
    "EXPLICIT_EULER",
    "INTEGRATORS",
    "SYMPLECTIC_EULER",
    "VELOCITY_VERLET",
    "Acceleration",
    "IntegrateError",
    "Integrator",
    "IntegratorDeclaration",
    "NumpyOps",
    "PurePythonOps",
    "VectorOps",
    "default_ops",
    "integrate",
]
