"""``CircularLoop``——引擎里第一次有"回路"这个几何词汇。

## 与`shapes.py`的关系：**不进`shapes`，理由是环序不是偏好**

`shapes.py`在spec/15的登记表里属**基座**（modelgen圈），而基座**不许依赖物理域**
（域隔离门③）。回路带`turns`与`current_a`——这两个字段没有任何非电磁的含义，
把它们塞进`shapes.py`就是让基座长出电磁语义。所以回路住在本域里。

这条判断与0035给`sensors`定环、0038给`motion`/`actuators`定环用的是同一把尺子：
**import决定环，不是愿望决定环**。触发条件写在这里：
若将来出现一个非电磁的消费方需要"圆形回路"这个纯几何词汇
（例如碰撞查询要一个圆环），那时把纯几何的那一半升进`shapes.py`、
把`turns`/`current_a`留在本域——**走决策记录，不许由某个基座模块直接import本域
来事实上完成升格**（域隔离门当场红，所以它也偷不成）。

## 单位：本类型的每个长度字段都带`_m`后缀，且mm只有一条入口

`radius_m`、`axial_position_m`——**不是`_mm`**。引擎的几何主单位是mm（spec/11），
本域是米（`units.EM_LENGTH_UNIT`），所以这里是那条边界的落点。
mm几何进来**只有`from_millimetres`一条路**，它内部只调
`units.metres_from_millimetres`，因子只存在一份。

## 明确不做的

* **不做倾斜与偏心**。本类型按定义共轴：所有回路的轴是同一根，回路平面垂直于它。
  一般位形要Neumann双回路线积分（plans/04第五节第3条），不在本块；
* **不做导线半径**，因此**不做自感**。丝状回路的自感对数发散，
  必须引入导线截面半径才有限——那是另一条闭式（Maxwell/Kirchhoff），另一块；
* **不做多层线圈的等效**（匝数只是线性倍数，见`inductance.py`的负空间）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.electromagnetics.errors import ElectromagneticsError
from physics_engine.electromagnetics.units import metres_from_millimetres


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ElectromagneticsError(f"{name}必须是实数：{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ElectromagneticsError(f"{name}必须是有限值：{value!r}")
    return number


@dataclass(frozen=True)
class CircularLoop:
    """一条共轴圆形回路：半径、轴向位置、匝数、载流。

    ``turns``是**集中匝**的理想化：N匝全部绕在同一条几何回路上，
    彼此之间没有轴向或径向间距。真实线圈不是这样——见``inductance.py``
    对匝数判据的负空间声明。

    ``current_a``只被``flux_linkage_wb``用到；互感本身与电流无关
    （互感是几何量）。它出现在这里是因为"回路"这个词汇在物理上就带电流，
    而且它让磁链有一个可算的定义——**不是为想象中的需求预留的字段**。
    """

    radius_m: float
    axial_position_m: float
    turns: int = 1
    current_a: float = 0.0

    def __post_init__(self) -> None:
        radius = _require_finite(self.radius_m, "radius_m")
        if radius <= 0.0:
            raise ElectromagneticsError(
                f"radius_m必须为正：{self.radius_m!r}——"
                "半径为零的回路互感恒为零，那不是一条回路"
            )
        _require_finite(self.axial_position_m, "axial_position_m")
        _require_finite(self.current_a, "current_a")
        turns = self.turns
        if isinstance(turns, bool) or not isinstance(turns, int):
            raise ElectromagneticsError(f"turns必须是整数：{self.turns!r}")
        if turns < 1:
            raise ElectromagneticsError(f"turns必须≥1：{self.turns!r}")

    @classmethod
    def from_millimetres(
        cls,
        *,
        radius_mm: float,
        axial_position_mm: float,
        turns: int = 1,
        current_a: float = 0.0,
    ) -> CircularLoop:
        """从mm制几何构造——**mm进入电磁域的唯一入口**。

        关键字参数是强制的：``CircularLoop.from_millimetres(50, 20)``这种写法
        在半径与轴向位置之间没有任何防错，而这两个量在数值上完全可互换。
        """

        return cls(
            radius_m=metres_from_millimetres(radius_mm),
            axial_position_m=metres_from_millimetres(axial_position_mm),
            turns=turns,
            current_a=current_a,
        )

    def axial_separation_m(self, other: CircularLoop) -> float:
        """两条回路的轴向间距``|z₂ − z₁|``。

        取绝对值：互感与"谁在上谁在下"无关。这也是互易判据能逐位成立的一半原因
        （另一半是``k²``的表达式对两个半径对称）。
        """

        if not isinstance(other, CircularLoop):
            raise ElectromagneticsError(f"axial_separation_m只接受CircularLoop：{other!r}")
        return abs(other.axial_position_m - self.axial_position_m)


__all__ = ["CircularLoop"]
