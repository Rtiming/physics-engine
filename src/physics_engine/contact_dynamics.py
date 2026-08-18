"""接触力矩装配层——把接触的**几何量**接到`rigidbody`的两个回调上。

决策0063第五节第3步（路径B），执行计划见
`docs/plans/16_自主长跑_路径B与性能模块化_20260818.md`的M1。
同行依据逐条在`docs/research/17`，本模块的每一处形制选择都指得回那一页的某一节。

## 本模块存在的理由：**不许把一个裸的力矩数喂进积分器**

`rigidbody.integrate_free_flight`收两个回调：世界系合力与**体系**力矩。
今天没有任何东西在算后者。最省事的接法是让接触那一侧直接交一个力矩数出来——
**而research/17第二节说明那是错的接法**：

* Bullet的切向摩擦把力矩**显式**算成``rel_pos1.cross(contactNormal1)``
  （`btSequentialImpulseConstraintSolver.cpp:537`，杆臂来自第1040行"接触点减质心"）；
* Fang & Negrut 2021 Appendix A.2原话："…which conserves angular momentum of the pair.
  **The pseudo-force itself does not act on the center of the mass.**"

**共同点是"力与矩来自同一个物理事件、共用同一份杆臂"。** 若两边各算各的，
线动量与角动量两条账各自看着都对，而**耦合关系没有被任何东西验过**。

所以本模块的输入是**（作用点的杆臂、世界系力）**，输出才是那两个回调。
``τ = r × F``只在这里出现一次。

## 杆臂取几何半径`R`，**不取`R − δ`**——本模块最要紧的一条

research/17第一节实测到同行分两派：

| 谁 | 杆臂 |
|---|---|
| Bullet／Fang & Negrut／TinyDEM | **几何半径`R`**（渗透量只进法向） |
| MuJoCo | `R − δ/2`（接触点放在两面中点） |

**而本仓已经是前一派，只是只站了一半**：`contact/penalty.py`的``_pair_state``
交出的是``length − radii_sum``——**半径不折算、渗透量单独扣**。

于是这不是"跟同行学的细节"，是**跟本仓自己保持一致**：

> 若摩擦/力矩这一半另起一套``R − δ``，同一帧里就有**两个互不相认的半径**。
> 接触点速度是``v_c = v_cm + ω × r``，`r`小一个`δ/2`就会凭空多出一个
> **量级为`δ·ω`的虚假滑移分量**——而"无滑"正是本轮要判的那件事。
> **一个用来判无滑的量，自己带着一个与穿透量同阶的伪滑移，那条判据就没有意义了。**

**因此本模块的杆臂一律是``r = R·n̂``（未折算），并且它与法向间隙用的是同一个`R`。**

## 球的法向罚力对质心的力矩**恒为零**，这是结构性的不是巧合

球的杆臂沿``n̂``、法向力也沿``n̂``，于是``r × F_n ≡ 0``——**逐位的零，不是近似的零**。
本模块把它写成一条可判的事实（见`sphere_plane_contact`的返回值），
理由与`rotation.py`那条"θ=0时耦合项恰为0"同源：
**一个结构性的零如果不被判，实现里多出一项来也没人知道。**

## 本模块**不做**什么

* **不做滚动阻力。** research/17第三节逐家核过：Bullet、Chrono、MuJoCo、
  Fang & Negrut 全都把它做成**显式声明的第二套力偶**，
  没有一家在裸的点接触＋库仑摩擦上会自动长出滚动阻力。
  本仓"点接触不传力偶"这句话**与所有主流同行的默认状态一致**——
  但也因此**不许声称覆盖滚动阻力**。要做就另立一个可关的力偶模型，
  不是指望摩擦项自己产生它。
* **不做检测。** 接触对由调用方声明，与`contact/`那边同纪律。
* **不做多体。** 一个体对一个静止平面，够本轮的靶子（无滑滚球与翻倒）用。
  多体要先裁"两个都在动时杆臂各取谁的半径"，那是另一件事。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from physics_engine.rigidbody import (
    RIGID_BODY_LAYOUT,
    cross,
    rotate_world_to_body,
)

Vector3 = tuple[float, float, float]

#: 与`rigidbody`同源的单位换算：力矩用`N·mm`，而Euler方程里的惯量是`kg·mm²`。
#: 这个常量本模块**不自己定义**——它属于`energies`，从那里传下来的链条见
#: `rigidbody.py`第50—52行。本模块只用`rigidbody`已经处理好的接口，
#: 所以这里不需要它；**写在这里是为了让读的人知道它已经被处理过，不要再乘一次**。


class ContactDynamicsError(ValueError):
    """接触动力学装配的失败关闭。**不返回"尽力而为"的力**。"""


def _require_vec3(value: object, what: str) -> Vector3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ContactDynamicsError(f"{what}必须是三个数：{value!r}")
    out = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContactDynamicsError(f"{what}第{index}个分量不是数：{item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ContactDynamicsError(f"{what}第{index}个分量不是有限数：{item!r}")
        out.append(number)
    return (out[0], out[1], out[2])


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _scale(vector: Vector3, factor: float) -> Vector3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _unit(vector: Vector3, what: str) -> Vector3:
    length = _norm(vector)
    if length == 0.0:
        raise ContactDynamicsError(f"{what}是零向量，方向没有定义")
    return _scale(vector, 1.0 / length)


@dataclass(frozen=True)
class ContactResponse:
    """一次接触求值的全部产物。**每一项都是可判的量，没有中间态。**

    分成这么多字段不是啰嗦：`gap_mm`与`slip_speed_mm_per_s`是**判据要读的**
    （前者判有没有接触、后者判无滑），而`normal_torque_body_nmm`存在的唯一理由
    是把"球的法向力不产生力矩"这条结构性事实变成可判的——见模块docstring。
    """

    #: 间隙。`< 0`才有接触；口径与`contact/penalty.py`一致（半径不折算）。
    gap_mm: float
    #: 世界系合力（法向＋切向）。
    force_world_n: Vector3
    #: **体系**力矩，可直接交给`rigidbody`的`torque_body_nmm`回调。
    torque_body_nmm: Vector3
    #: 法向力单独产生的体系力矩。**球上它必须逐位为零**（`r ∥ n̂`）。
    normal_torque_body_nmm: Vector3
    #: 接触点的相对滑移速率。**"无滑"判的就是它**，不是"我们假定无滑"。
    slip_speed_mm_per_s: float
    #: 切向力是否已经饱和在摩擦锥上（`|F_t| = μ|F_n|`）。滑与不滑的布尔量。
    sliding: bool


def sphere_plane_contact(
    vector: tuple[float, ...],
    *,
    radius_mm: float,
    plane_point_mm: Vector3,
    plane_normal: Vector3,
    normal_stiffness_n_per_mm: float,
    tangential_stiffness_n_per_mm: float,
    friction_coefficient: float,
    normal_damping_n_s_per_mm: float = 0.0,
) -> ContactResponse:
    """一个刚体球对一个静止平面的接触响应。**几何量在这里算，力矩在这里装。**

    ``vector``是`rigidbody.RIGID_BODY_LAYOUT`的13维状态向量。

    ## 切向力用的是**速度型**库仑摩擦，不是锚点型

    `contact/friction.py`那一族是准静态路径上的**锚点＋return-map**：
    它记住"上一次粘住的位置"，因为准静态里没有时间、只有载荷步。
    动态路径上有时间，于是切向力可以直接由**接触点相对速度**给出：

        F_t = −min(k_t·|v_t|·dt_eq, μ|F_n|) · v̂_t     （粘）
        F_t = −μ|F_n| · v̂_t                            （滑）

    **本函数取更简单也更保守的那一档：`F_t = −min(k_t·|v_t|, μ|F_n|)·v̂_t`**，
    即把`k_t`当成"切向阻尼"而不是"切向弹簧"。理由写在明处：
    速度型摩擦**不需要历史**，于是它是纯函数、能直接进RK4的导数回调；
    锚点型要历史，而历史进不了一个"给我状态、还我导数"的回调
    （spec/12第11.2节登记的那条"无状态纯函数在接触层不成立"正是说的这件事）。

    **代价也写在明处**：速度型摩擦在真正的静止粘着上会有一个残余滑移
    ``|v_t| ≈ μ|F_n|/k_t``，它**不是零**。所以"无滑"这条判据判的是
    **`|v_t|`相对`ωR`足够小**，不是`|v_t| == 0`——见`ContactResponse.slip_speed_mm_per_s`
    的用法与M2的验收标准。

    ## 杆臂

    ``r = −R·n̂``（从质心指向接触点；`n̂`是平面指向球那一侧的外法向）。
    **`R`是几何半径，不折算渗透量**——理由见模块docstring那一节。
    """

    if radius_mm <= 0.0 or not math.isfinite(radius_mm):
        raise ContactDynamicsError(f"radius_mm必须是正的有限数：{radius_mm!r}")
    if normal_stiffness_n_per_mm <= 0.0:
        raise ContactDynamicsError("normal_stiffness_n_per_mm必须为正")
    if tangential_stiffness_n_per_mm < 0.0:
        raise ContactDynamicsError("tangential_stiffness_n_per_mm不能为负")
    if friction_coefficient < 0.0:
        raise ContactDynamicsError("friction_coefficient不能为负")
    if len(vector) != RIGID_BODY_LAYOUT.dof_count:
        raise ContactDynamicsError(
            f"状态向量长度{len(vector)}不是刚体布局的{RIGID_BODY_LAYOUT.dof_count}"
        )

    normal = _unit(_require_vec3(plane_normal, "plane_normal"), "plane_normal")
    point = _require_vec3(plane_point_mm, "plane_point_mm")

    centre = (vector[0], vector[1], vector[2])
    velocity = (vector[3], vector[4], vector[5])
    omega_body = (vector[6], vector[7], vector[8])
    attitude = (vector[9], vector[10], vector[11], vector[12])

    #: 间隙：质心到平面的有符号距离减半径。**与`_pair_state`同口径**。
    height = _dot(_sub(centre, point), normal)
    gap = height - radius_mm

    zero: Vector3 = (0.0, 0.0, 0.0)
    if gap >= 0.0:
        return ContactResponse(
            gap_mm=gap,
            force_world_n=zero,
            torque_body_nmm=zero,
            normal_torque_body_nmm=zero,
            slip_speed_mm_per_s=0.0,
            sliding=False,
        )

    #: 杆臂：质心 → 接触点，沿`−n̂`、长度是**几何半径**。
    lever = _scale(normal, -radius_mm)

    #: 体系角速度换到世界系才能与质心速度相加。
    #: `rotate_world_to_body`的逆用姿态的共轭——`rigidbody`已有这两个方向的换算，
    #: 本模块不自己写第三份（research/17第六节：不让接触层去猜姿态该怎么用）。
    conjugate = (-attitude[0], -attitude[1], -attitude[2], attitude[3])
    omega_world = rotate_world_to_body(conjugate, omega_body)

    #: 接触点速度 `v_c = v_cm + ω × r`。
    contact_velocity = _add(velocity, cross(omega_world, lever))
    normal_speed = _dot(contact_velocity, normal)
    tangential_velocity = _sub(contact_velocity, _scale(normal, normal_speed))
    slip_speed = _norm(tangential_velocity)

    #: 法向：罚力 ＋ 可选的线性阻尼（只在压缩时耗散，不在分离时"吸住"）。
    normal_magnitude = -normal_stiffness_n_per_mm * gap
    if normal_damping_n_s_per_mm > 0.0 and normal_speed < 0.0:
        normal_magnitude += -normal_damping_n_s_per_mm * normal_speed
    if normal_magnitude < 0.0:
        normal_magnitude = 0.0
    normal_force = _scale(normal, normal_magnitude)

    #: 切向：速度型库仑，饱和在摩擦锥上。
    cone = friction_coefficient * normal_magnitude
    sliding = False
    if slip_speed == 0.0 or cone == 0.0:
        tangential_force: Vector3 = zero
    else:
        wanted = tangential_stiffness_n_per_mm * slip_speed
        magnitude = wanted
        if wanted >= cone:
            magnitude = cone
            sliding = True
        direction = _scale(tangential_velocity, 1.0 / slip_speed)
        tangential_force = _scale(direction, -magnitude)

    force = _add(normal_force, tangential_force)

    #: **力矩在这里装，且只在这里装一次。**
    normal_torque_world = cross(lever, normal_force)
    torque_world = cross(lever, force)
    return ContactResponse(
        gap_mm=gap,
        force_world_n=force,
        torque_body_nmm=rotate_world_to_body(attitude, torque_world),
        normal_torque_body_nmm=rotate_world_to_body(attitude, normal_torque_world),
        slip_speed_mm_per_s=slip_speed,
        sliding=sliding,
    )


def sphere_plane_callbacks(
    *,
    radius_mm: float,
    plane_point_mm: Vector3,
    plane_normal: Vector3,
    normal_stiffness_n_per_mm: float,
    tangential_stiffness_n_per_mm: float,
    friction_coefficient: float,
    gravity_world_n: Vector3 = (0.0, 0.0, 0.0),
    normal_damping_n_s_per_mm: float = 0.0,
) -> tuple[Callable[[tuple[float, ...], float], Vector3], Callable[[tuple[float, ...], float], Vector3]]:
    """把上面那个响应包成`integrate_free_flight`要的两个回调。

    **重力在这里加，不在接触函数里加**：接触函数回答的是"接触给了什么"，
    重力不是接触给的。混在一起会让"法向力"这个量不再等于接触法向力，
    而摩擦锥`μ|F_n|`用的正是它——**那样摩擦会被重力污染，且不报任何错**。
    """

    def force(vector: tuple[float, ...], _t: float) -> Vector3:
        response = sphere_plane_contact(
            vector,
            radius_mm=radius_mm,
            plane_point_mm=plane_point_mm,
            plane_normal=plane_normal,
            normal_stiffness_n_per_mm=normal_stiffness_n_per_mm,
            tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
            friction_coefficient=friction_coefficient,
            normal_damping_n_s_per_mm=normal_damping_n_s_per_mm,
        )
        return _add(response.force_world_n, gravity_world_n)

    def torque(vector: tuple[float, ...], _t: float) -> Vector3:
        response = sphere_plane_contact(
            vector,
            radius_mm=radius_mm,
            plane_point_mm=plane_point_mm,
            plane_normal=plane_normal,
            normal_stiffness_n_per_mm=normal_stiffness_n_per_mm,
            tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
            friction_coefficient=friction_coefficient,
            normal_damping_n_s_per_mm=normal_damping_n_s_per_mm,
        )
        return response.torque_body_nmm

    return force, torque


__all__ = [
    "ContactDynamicsError",
    "ContactResponse",
    "sphere_plane_callbacks",
    "sphere_plane_contact",
]
