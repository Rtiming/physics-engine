"""零依赖前向jet：一阶与二阶，tuple-of-float实现（决策0064第4.1节的执行面）。

**本模块不是新代码。** `Jet1`/`Jet2`与六个`ad_*`包装从2026-08-13起就在
`section_beam.py`里跑着easy-axis曲率的解析链式法则；决策0064第4.1节裁定整杆
各向异性弯曲的梯度/Hessian也走它，于是它从一个模块的私有实现升成共享面。

裁决的理由不是"jet比解析好"，是本仓性能条款第一句**优化先profile**：
各向异性版是11个变量、两个曲率分量、还含边扭角γ，二阶解析链式法则很难写对；
一上来写它等于在没有任何实测的情况下押注常数因子，而押错的代价是几百行难验的
二阶导。先用好写对的这条拿到正确答案与一条真实墙钟，"要不要解析"才有分母。

**这次提升只搬运，不改数学。** `section_beam.py`的曲率值、梯度与Hessian在提升
前后逐字节相同——`tests/test_section_beam.py`里那三串WDS对拍常数就是判据。

### 唯一的行为差别，写在这里以免它静默

jet内部的失败（对非正数开方、除零、两个宽度不一致的jet相运算）此前抛
`KirchhoffSectionError`，现在抛`AutodiffError`。两者都是`ValueError`的子类，
所以`solve_equilibrium`线搜索里那句`except (ValueError, ZeroDivisionError)`
两边都接得住；只catch `KirchhoffSectionError`的调用方会漏掉——
**仓内没有这样的调用方**（查过：两个jet类与六个包装此前只有`section_beam.py`
一个引用点，且现存三条`pytest.raises(KirchhoffSectionError)`断言的都是
`section_beam`自己抛的消息：orthonormal／antiparallel／committed history）。

### 二阶jet的代价先写清楚，再谈要不要换掉它

`Jet2`每次乘法或除法都要建一个``size × size``的tuple-of-tuples。11变量时
一次乘法是121次浮点乘加外加一次元组构造，而弯曲曲率核里这样的运算有几十次，
逐顶点逐牛顿迭代地发生。**这不是可以忽略的常数**——它正是0064第4.1节
要求同批交出"逐顶点求值墙钟中位数"的原因。数字见决策0065第三节，
**那里的数是量出来的，不是这里估的**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class AutodiffError(ValueError):
    """jet层的一切失败关闭。是`ValueError`的子类，见模块docstring那条行为差别。"""


@dataclass(frozen=True)
class Jet1:
    """只携一阶导的轻量jet；残差装配不为未请求的二阶量付费。"""

    value: float
    gradient: tuple[float, ...]

    @property
    def size(self) -> int:
        return len(self.gradient)

    @classmethod
    def constant(cls, value: float, size: int) -> Jet1:
        return cls(float(value), (0.0,) * size)

    @classmethod
    def variable(cls, value: float, index: int, size: int) -> Jet1:
        gradient = [0.0] * size
        gradient[index] = 1.0
        return cls(float(value), tuple(gradient))

    def _coerce(self, other) -> Jet1:
        if isinstance(other, Jet1):
            if other.size != self.size:
                raise AutodiffError("first-order jets have inconsistent widths")
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Jet1.constant(float(other), self.size)
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Jet1(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.gradient, other.gradient, strict=True)),
        )

    __radd__ = __add__

    def __neg__(self):
        return Jet1(-self.value, tuple(-value for value in self.gradient))

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Jet1(
            self.value * other.value,
            tuple(
                self.gradient[index] * other.value
                + self.value * other.gradient[index]
                for index in range(self.size)
            ),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> Jet1:
        if self.value == 0.0:
            raise AutodiffError("division by zero in a first-order jet")
        inverse = 1.0 / self.value
        return Jet1(
            inverse,
            tuple(-inverse * inverse * component for component in self.gradient),
        )

    def __truediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        inverse = other.reciprocal()
        # 导数用乘倒数展开，但值通道保留Python直接除法的舍入；这样0/1/2阶
        # 入口在屈服分支上逐位选择同一个标量曲率。
        return Jet1(
            self.value / other.value,
            tuple(
                self.gradient[index] * inverse.value
                + self.value * inverse.gradient[index]
                for index in range(self.size)
            ),
        )

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other / self


@dataclass(frozen=True)
class Jet2:
    """一阶＋二阶的前向jet。二阶量是稠密``size × size``，代价见模块docstring。"""

    value: float
    gradient: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]

    @property
    def size(self) -> int:
        return len(self.gradient)

    @classmethod
    def constant(cls, value: float, size: int) -> Jet2:
        return cls(float(value), (0.0,) * size, tuple((0.0,) * size for _ in range(size)))

    @classmethod
    def variable(cls, value: float, index: int, size: int) -> Jet2:
        gradient = [0.0] * size
        gradient[index] = 1.0
        return cls(float(value), tuple(gradient), tuple((0.0,) * size for _ in range(size)))

    def _coerce(self, other) -> Jet2:
        if isinstance(other, Jet2):
            if other.size != self.size:
                raise AutodiffError("second-order jets have inconsistent widths")
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Jet2.constant(float(other), self.size)
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Jet2(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.gradient, other.gradient, strict=True)),
            tuple(
                tuple(a + b for a, b in zip(left, right, strict=True))
                for left, right in zip(self.hessian, other.hessian, strict=True)
            ),
        )

    __radd__ = __add__

    def __neg__(self):
        return Jet2(
            -self.value,
            tuple(-value for value in self.gradient),
            tuple(tuple(-value for value in row) for row in self.hessian),
        )

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        gradient = tuple(
            self.gradient[index] * other.value + self.value * other.gradient[index]
            for index in range(self.size)
        )
        hessian = tuple(
            tuple(
                self.hessian[row][column] * other.value
                + self.gradient[row] * other.gradient[column]
                + other.gradient[row] * self.gradient[column]
                + self.value * other.hessian[row][column]
                for column in range(self.size)
            )
            for row in range(self.size)
        )
        return Jet2(self.value * other.value, gradient, hessian)

    __rmul__ = __mul__

    def reciprocal(self) -> Jet2:
        if self.value == 0.0:
            raise AutodiffError("division by zero in a second-order jet")
        inverse = 1.0 / self.value
        first = -inverse * inverse
        second = 2.0 * inverse * inverse * inverse
        return jet_unary(self, inverse, first, second)

    def __truediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        inverse = other.reciprocal()
        gradient = tuple(
            self.gradient[index] * inverse.value
            + self.value * inverse.gradient[index]
            for index in range(self.size)
        )
        hessian = tuple(
            tuple(
                self.hessian[row][column] * inverse.value
                + self.gradient[row] * inverse.gradient[column]
                + inverse.gradient[row] * self.gradient[column]
                + self.value * inverse.hessian[row][column]
                for column in range(self.size)
            )
            for row in range(self.size)
        )
        return Jet2(self.value / other.value, gradient, hessian)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other / self


def jet_unary(value: Jet2, result: float, first: float, second: float) -> Jet2:
    """一元函数的二阶链式法则：``f(g)``的一二阶量由``f'``、``f''``与``g``的量装出来。"""

    return Jet2(
        result,
        tuple(first * component for component in value.gradient),
        tuple(
            tuple(
                second * value.gradient[row] * value.gradient[column]
                + first * value.hessian[row][column]
                for column in range(value.size)
            )
            for row in range(value.size)
        ),
    )


def ad_sqrt(value):
    """``sqrt``；非正实参**失败关闭**（那是长度为零或近零的构型，导数不存在）。"""

    raw = value.value if isinstance(value, (Jet1, Jet2)) else value
    if raw <= 0.0:
        raise AutodiffError("sqrt of a non-positive value — 长度为零的边上导数不存在")
    result = math.sqrt(raw)
    if isinstance(value, Jet2):
        return jet_unary(value, result, 0.5 / result, -0.25 / (raw * result))
    if isinstance(value, Jet1):
        return Jet1(
            result,
            tuple(0.5 / result * component for component in value.gradient),
        )
    return result


def ad_sin(value):
    raw = value.value if isinstance(value, (Jet1, Jet2)) else value
    result = math.sin(raw)
    if isinstance(value, Jet2):
        return jet_unary(value, result, math.cos(raw), -result)
    if isinstance(value, Jet1):
        return Jet1(
            result,
            tuple(math.cos(raw) * component for component in value.gradient),
        )
    return result


def ad_cos(value):
    raw = value.value if isinstance(value, (Jet1, Jet2)) else value
    result = math.cos(raw)
    if isinstance(value, Jet2):
        return jet_unary(value, result, -math.sin(raw), -result)
    if isinstance(value, Jet1):
        return Jet1(
            result,
            tuple(-math.sin(raw) * component for component in value.gradient),
        )
    return result


def ad_dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def ad_cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def ad_norm(vector):
    return ad_sqrt(ad_dot(vector, vector))


__all__ = [
    "AutodiffError",
    "Jet1",
    "Jet2",
    "ad_cos",
    "ad_cross",
    "ad_dot",
    "ad_norm",
    "ad_sin",
    "ad_sqrt",
    "jet_unary",
]
