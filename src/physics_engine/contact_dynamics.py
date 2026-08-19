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

## 多接触点对同一平面：**翻倒要的是"底面"，而底面是一组物质点**

球对平面只有一个接触点，于是`r × F_n ≡ 0`、法向力**永远不产生力矩**——
一个球不会翻倒，它只会滚。**翻倒要的是第二个接触点**：底面两侧的法向力不等，
合力作用线离开质心投影，那个不平衡才是倾覆力矩。

所以本模块的第二档是`support_points_plane_contact`：**一组由调用方声明的
体系支承点**对同一个静止平面。形制逐条照抄球那一档——

| 口径 | 球那一档 | 支承点这一档 |
|---|---|---|
| 杆臂 | `r = −R·n̂`，**几何半径不折算** | `r_i = R(q)·p_i`，**体系声明值不折算** |
| 间隙 | `(c − p)·n̂ − R` | `(c + r_i − p)·n̂` |
| 摩擦 | 速度型库仑，饱和在`μ|F_n|` | **逐点**同一条，各点各自饱和 |
| 力矩 | `τ = r × F`，只装一次 | `τ = Σ r_i × F_i`，仍只装一次 |

**"不折算"在这一档更要紧**：支承点是**物质点**（决策0080第二节的同一句话——
锚点记的是物质点，不是空间点），它随姿态刚性地转，穿透只进`gap_i`。
若杆臂改用"接触点被压到平面上"的那个投影点，杆臂就会随载荷变，
而`v_c = v_cm + ω × r_i`里凭空多出一个与穿透同阶的伪滑移——
与球那一档要挡的是**同一个**错。

**法向力矩在这一档不是零，而那正是判据要读的量**：
`normal_torque_body_nmm`在球上是结构性的零、在支承点组上是**倾覆/复位力矩本身**。
同一个字段两档语义不同，因此两档各配一条判据（球判它逐位为零，
支承点组判它在阈值两侧**变号**）。

## 底面形状怎么声明：**由调用方给一组点，本模块不猜**

plans/16的M4写着"底面形状怎么声明未裁"。**本模块的裁法是最小的那一种**：
调用方交一组体系点，本模块只对这组点求值。`box_corner_points_mm`是**一个**
便利构造器（长方体八角），不是唯一形制。

**因此明确不覆盖**（GAP，登记在决策0082第五节）：
支承多边形的凸性与冗余不被校验（给三个共线点也照算）；
不从网格自动导出接触足印；圆底/线接触要的是分布压力而不是有限个点，本档给不出。

## 步长上限：**力矩装配看到的模态与罚接触的刚度模态不是同一个**（决策0087）

`contact_pipeline`已经有一条"h相对接触刚度"的上限（`CONTACT_STIFFNESS_STEP_BOUND`），
**本模块仍然需要自己那一条**，理由是一句话：**那一层没有杆臂。**

`contact_pipeline`的两端都是绑在**平动**节点上的球，接触力只改质心速度，
于是它的模态永远是一个标量：``ω0 = √(1000·k/m_eff)``，`m_eff`是约化质量。
**本模块的力作用在一个有杆臂的点上**，``τ = r × F``把同一个力送进了转动方程，
于是同一个`k`看到的是一个**六维**算子的谱。推导逐行如下：

1. 第`i`个支承点的速度是``u_i = v + ω × r_i = J_i·(v, ω)``，其中
   ``J_i = [I₃ | −[r_i]×]``（3×6）；
2. 该点的接触力是``F_i = −κ·P·u_i``（`P`是投影：法向取``n̂n̂ᵀ``、切向取``I₃−n̂n̂ᵀ``；
   `κ`法向取`k_n`或`c_n`、切向取`k_t`），广义力旋量因此是``−κ·J_iᵀ P J_i·(v,ω)``；
3. 运动方程（单位换算与`rigidbody`第610/625行同一个`MM_PER_M`）是
   ``d(v,ω)/dt = 1000·G·W``，``G = diag(I₃/m, I⁻¹)``；
4. 于是速率算子是``1000·κ·G·Σ_i J_iᵀ P J_i``，它的特征值就是要挡的模态速率。
   记``ρ = λ(G·Σ J_iᵀ n̂n̂ᵀ J_i)``、``σ = λ(G·Σ J_iᵀ (I₃−n̂n̂ᵀ) J_i)``（单位1/kg）；
5. **法向族是二阶的**（有刚度也有阻尼），走`contact_pipeline`那条逐字相同的读法：
   ``ω0 = √(1000·k_n·ρ)``、``ζ = (c_n/2)·√(1000·ρ/k_n)``、
   速率取`ω0`（ζ≤1）或``(ζ+√(ζ²−1))·ω0``（ζ>1）。
   ``D_n = (c_n/k_n)·K_n``是**恒等**关系（同一个投影），所以两个算子严格共特征向量，
   这一步不是近似；
6. **切向族是一阶的**（速度型摩擦只有阻尼没有刚度），速率直接是``1000·k_t·σ``。

单点无杆臂时``ρ → 1/m``、``σ → 1/m``，第5步逐字退化成`contact_pipeline`那条式子——
**两条上限同形不同分母，这正是"两条独立、取更紧的那个"的意思**。
有杆臂时它们分道扬镳：实测本仓那只箱子的底面四角组`σ = 272`／kg，
而"按平动约化质量算"给的是`4/m = 80`／kg，**差3.4倍，全在杆臂上**。

`k_t`那条上限是同一个不等式解`k_t`而不是解`h`，见`tangential_coefficient_window`。

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
* **步长上限不做两族之间的耦合。** 法向族（刚度＋阻尼）与切向族（只有阻尼）
  各自在自己的投影上被精确对角化，但完整系统是一个二次特征值问题
  ``M q̈ + (D_n + D_t) q̇ + K_n q = 0``，两族的特征向量一般不重合。
  **这不是保守的**：分开算有可能比真值松。今天没有做，登记为GAP
  （决策0087第六节），触发条件是第一个把`c_n`与`k_t`调到同一量级的消费方。
  实测两条案例上分开算都是**保守**的（发散/颤振起点比声明的上限还高10.6%与15.7%）。
* **步长上限不做几何刚度**（杆臂随姿态转出来的那一项）与**陀螺项**
  （``ω × Iω``）。前者在小倾角上是高阶量，后者由`rigidbody`那条转动模态上限管，
  两条上限取更紧的那个（`governing_assembly_step_bound`）。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from physics_engine.energies import MM_PER_M
from physics_engine.rigidbody import (
    RIGID_BODY_LAYOUT,
    cross,
    rotate_body_to_world,
    rotate_world_to_body,
)

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]

#: 与`rigidbody`同源的单位换算：力矩用`N·mm`，而Euler方程里的惯量是`kg·mm²`。
#: 这个常量本模块**不自己定义**——它属于`energies`，从那里传下来的链条见
#: `rigidbody.py`第50—52行。响应那一档（力与力矩）只用`rigidbody`已经处理好的
#: 接口，所以那一半不需要它；**步长上限那一档需要**（它要自己写出运动方程的
#: 速率算子），因此从`energies`原样import——与`rigidbody.py`第67行同一条纪律，
#: **不在本文件里再写一个1000.0**。


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




# ---------------------------------------------------------------------------
# 多支承点对同一平面——翻倒那一档（决策0082）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportPointContact:
    """一个支承点这一步的全部产物。**逐点都要能单独判。**

    翻倒的可观测量正是"**哪几个点还在承载**"——把它们合并成一个总力就再也看不见了。
    球那一档只有一个点，所以`ContactResponse`是扁的；这一档必须逐点带回来。
    """

    #: 体系声明的支承点，**原样带回**。调用方按序号对上自己的声明，
    #: 而不是靠"我记得第三个是右前角"——那正是`_slice_of`要挡的同一族错。
    point_body_mm: Vector3
    #: 杆臂：质心 → 该支承点，**世界系**，`r_i = R(q)·p_i`。**不折算穿透**。
    lever_world_mm: Vector3
    #: 该点到平面的有符号距离。`< 0`才承载。
    gap_mm: float
    #: 该点的世界系接触力（法向＋切向）。不承载时**逐位**是零向量。
    force_world_n: Vector3
    #: 该点的相对滑移速率（mm/s）。
    slip_speed_mm_per_s: float
    #: 该点的切向力是否已经饱和在摩擦锥上。
    sliding: bool

    @property
    def in_contact(self) -> bool:
        """`gap_mm < 0`。**写成属性而不是让调用方自己比**：
        "接触与否"在本仓已经有一个口径（`contact/penalty.py`的`gap < 0`），
        不许在判据里长出第二个（比如某处写成`<= 0`，恰好在阈值上分道扬镳）。"""

        return self.gap_mm < 0.0


@dataclass(frozen=True)
class SupportSetResponse:
    """一组支承点对同一平面的合成产物。

    `points`的**次序与调用方声明的次序逐一对应**，这是形制的一部分：
    判据要读"第几个点还在承载"，次序换了判据就换了意思。
    """

    #: 逐点产物，次序同声明。
    points: tuple[SupportPointContact, ...]
    #: 世界系合力（所有承载点的法向＋切向之和）。
    force_world_n: Vector3
    #: **体系**合力矩`τ = Σ r_i × F_i`，可直接交给`rigidbody`的回调。
    torque_body_nmm: Vector3
    #: 只由法向力产生的那一半体系力矩。**这一档它不是零**——
    #: 它就是倾覆/复位力矩本身，翻倒判据读的正是它的**符号**。
    normal_torque_body_nmm: Vector3

    @property
    def contact_count(self) -> int:
        """还在承载的点数。翻倒的第一可观测量：4 → 2 → 换一组点。"""

        return sum(1 for point in self.points if point.in_contact)


def box_corner_points_mm(half_extents_mm: Vector3) -> tuple[Vector3, ...]:
    """长方体八个角点的体系坐标。**次序是形制的一部分，写在这里一次。**

    次序：`z`最慢、`x`最快，各轴从负到正。于是

    * **索引0—3是底面**（`z = −c`），索引4—7是顶面；
    * 同一面内索引0/1是`y = −b`那一边、2/3是`y = +b`那一边。

    判据会写"底面那四个点全部离地"，而"底面是哪四个"必须是**被声明的**、
    不是被数出来的——`rigidbody._slice_of`那条"偏移量由布局算、调用方永不手写"
    在这里是同一条纪律。

    这个构造器是**一个**便利形制，不是唯一形制：`support_points_plane_contact`
    收的是任意一组体系点，长方体只是本仓第一个有底面的靶子。
    """

    half = _require_vec3(half_extents_mm, "half_extents_mm")
    if any(extent <= 0.0 for extent in half):
        raise ContactDynamicsError(f"half_extents_mm必须全为正：{half_extents_mm!r}")
    return tuple(
        (sx * half[0], sy * half[1], sz * half[2])
        for sz in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sx in (-1.0, 1.0)
    )


def support_points_plane_contact(
    vector: tuple[float, ...],
    *,
    support_points_body_mm: tuple[Vector3, ...],
    plane_point_mm: Vector3,
    plane_normal: Vector3,
    normal_stiffness_n_per_mm: float,
    tangential_stiffness_n_per_mm: float,
    friction_coefficient: float,
    normal_damping_n_s_per_mm: float = 0.0,
) -> SupportSetResponse:
    """一组体系支承点对一个静止平面的接触响应。**力矩仍然只装一次。**

    ``vector``是`rigidbody.RIGID_BODY_LAYOUT`的13维状态向量。
    每个点各自算间隙、法向罚力、速度型库仑切向力；**杆臂是`r_i = R(q)·p_i`，
    不折算穿透**——理由与球那一档的"取几何半径`R`不取`R − δ`"是同一条，
    见模块docstring。

    ## 为什么支承点在**体**系声明

    支承点是**物质点**（决策0080第二节）。在体系声明意味着它随姿态刚性地转，
    而这正是翻倒要的：翻过去之后贴地的是**另外几个**物质点，
    若支承点在世界系声明，它们就会在体转过去之后留在原地——那不是一个刚体。

    ## 空支承集**失败关闭**

    一组零个点不是"没有接触"，是**没有声明底面**。前者应该由`gap >= 0`表达、
    后者是调用方漏了参数；把两者合并成"返回零力"会让一个配置错误安静地
    变成一次自由落体。
    """

    points_in = tuple(support_points_body_mm)
    if not points_in:
        raise ContactDynamicsError(
            "support_points_body_mm是空的——空支承集不是『没有接触』，是没有声明底面"
        )
    if normal_stiffness_n_per_mm <= 0.0:
        raise ContactDynamicsError("normal_stiffness_n_per_mm必须为正")
    if tangential_stiffness_n_per_mm < 0.0:
        raise ContactDynamicsError("tangential_stiffness_n_per_mm不能为负")
    if friction_coefficient < 0.0:
        raise ContactDynamicsError("friction_coefficient不能为负")
    if normal_damping_n_s_per_mm < 0.0:
        raise ContactDynamicsError("normal_damping_n_s_per_mm不能为负")
    if len(vector) != RIGID_BODY_LAYOUT.dof_count:
        raise ContactDynamicsError(
            f"状态向量长度{len(vector)}不是刚体布局的{RIGID_BODY_LAYOUT.dof_count}"
        )

    normal = _unit(_require_vec3(plane_normal, "plane_normal"), "plane_normal")
    plane = _require_vec3(plane_point_mm, "plane_point_mm")
    body_points = tuple(
        _require_vec3(point, f"support_points_body_mm[{index}]")
        for index, point in enumerate(points_in)
    )

    centre = (vector[0], vector[1], vector[2])
    velocity = (vector[3], vector[4], vector[5])
    omega_body = (vector[6], vector[7], vector[8])
    attitude = (vector[9], vector[10], vector[11], vector[12])
    #: 体→世界直接走`rigidbody.rotate_body_to_world`（本模块不自己写第三份换算，
    #: research/17第六节）。球那一档写成`rotate_world_to_body(共轭, ·)`，
    #: 两条路**逐位等价**（`R(q*) = R(q)ᵀ`，求和次序也相同），
    #: 这里取直接那一条只因为它少造一个四元数。
    omega_world = rotate_body_to_world(attitude, omega_body)

    zero: Vector3 = (0.0, 0.0, 0.0)
    total_force = zero
    total_torque_world = zero
    normal_torque_world = zero
    results: list[SupportPointContact] = []
    for point in body_points:
        lever = rotate_body_to_world(attitude, point)
        gap = _dot(_sub(_add(centre, lever), plane), normal)
        if gap >= 0.0:
            results.append(
                SupportPointContact(
                    point_body_mm=point,
                    lever_world_mm=lever,
                    gap_mm=gap,
                    force_world_n=zero,
                    slip_speed_mm_per_s=0.0,
                    sliding=False,
                )
            )
            continue

        contact_velocity = _add(velocity, cross(omega_world, lever))
        normal_speed = _dot(contact_velocity, normal)
        tangential_velocity = _sub(contact_velocity, _scale(normal, normal_speed))
        slip_speed = _norm(tangential_velocity)

        normal_magnitude = -normal_stiffness_n_per_mm * gap
        if normal_damping_n_s_per_mm > 0.0 and normal_speed < 0.0:
            normal_magnitude += -normal_damping_n_s_per_mm * normal_speed
        if normal_magnitude < 0.0:
            normal_magnitude = 0.0
        normal_force = _scale(normal, normal_magnitude)

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
        total_force = _add(total_force, force)
        #: **力矩在这里装，且只在这里装一次**——与球那一档同一句话。
        total_torque_world = _add(total_torque_world, cross(lever, force))
        normal_torque_world = _add(normal_torque_world, cross(lever, normal_force))
        results.append(
            SupportPointContact(
                point_body_mm=point,
                lever_world_mm=lever,
                gap_mm=gap,
                force_world_n=force,
                slip_speed_mm_per_s=slip_speed,
                sliding=sliding,
            )
        )

    return SupportSetResponse(
        points=tuple(results),
        force_world_n=total_force,
        torque_body_nmm=rotate_world_to_body(attitude, total_torque_world),
        normal_torque_body_nmm=rotate_world_to_body(attitude, normal_torque_world),
    )


def support_points_plane_callbacks(
    *,
    support_points_body_mm: tuple[Vector3, ...],
    plane_point_mm: Vector3,
    plane_normal: Vector3,
    normal_stiffness_n_per_mm: float,
    tangential_stiffness_n_per_mm: float,
    friction_coefficient: float,
    gravity_world_n: Vector3 = (0.0, 0.0, 0.0),
    normal_damping_n_s_per_mm: float = 0.0,
) -> tuple[
    Callable[[tuple[float, ...], float], Vector3],
    Callable[[tuple[float, ...], float], Vector3],
]:
    """把上面那个响应包成`integrate_free_flight`要的两个回调。

    **重力在这里加，不在接触函数里加**——与球那一档同一条理由：
    摩擦锥`μ|F_n|`用的正是接触法向力，混进重力它就被污染了，且不报任何错。

    ## 那个一格记忆：**它是精确的，不是近似的**

    `_derivative_factory`每算一次导数会**先后**调用力回调与力矩回调，
    参数是**同一个**状态元组对象。支承点这一档一次求值要过`N`个点，
    照球那一档的写法就会把整组点算两遍。

    这里的记忆键是``vector is 上一次的vector`` ——**对象同一性，不是数值相等**，
    并且把那个对象的引用**留着**（所以`id`不会被回收后重用）。
    于是：命中时返回的必然是同一次求值的产物，**逐位相同不是巧合而是同一个对象**；
    不命中就老老实实重算。

    **拆掉这个记忆，产物逐位不变**——这一条被
    `test_the_one_slot_memo_changes_nothing`守着，因为"为了快改了数"
    正是本仓性能条款第二句要挡的事。
    """

    cache: list[object] = [None, None, None]

    def evaluate(vector: tuple[float, ...], t: float) -> SupportSetResponse:
        if cache[0] is vector and cache[1] == t:
            return cache[2]  # type: ignore[return-value]
        response = support_points_plane_contact(
            vector,
            support_points_body_mm=support_points_body_mm,
            plane_point_mm=plane_point_mm,
            plane_normal=plane_normal,
            normal_stiffness_n_per_mm=normal_stiffness_n_per_mm,
            tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
            friction_coefficient=friction_coefficient,
            normal_damping_n_s_per_mm=normal_damping_n_s_per_mm,
        )
        cache[0] = vector
        cache[1] = t
        cache[2] = response
        return response

    def force(vector: tuple[float, ...], t: float) -> Vector3:
        return _add(evaluate(vector, t).force_world_n, gravity_world_n)

    def torque(vector: tuple[float, ...], t: float) -> Vector3:
        return evaluate(vector, t).torque_body_nmm

    return force, torque


# ---------------------------------------------------------------------------
# 步长上限与`k_t`上限——决策0087的丙1与丙2
# ---------------------------------------------------------------------------


#: RK4的实轴稳定区半径。**与`rigidbody.RK4_BODY.declaration.step_bound`里的
#: 那个2.785、以及`contact_pipeline.RK4_STABILITY_RADIUS`是同一个数**——
#: 上限的分子属于积分器，几条上限只在分母上不同。三处不许漂，
#: `tests/test_contact_dynamics_step_bound.py`从`rigidbody`的声明字符串里
#: 把数字抠出来对拍（0083注错第M4轮抓到过：这个数当时没有任何东西钉着，
#: 改成2.0全套门都绿，**而那种错只让上限变松、不会变红**）。
RK4_STABILITY_RADIUS = 2.785

#: 显式Euler的实轴稳定区半径（同上，对应`rigidbody.EXPLICIT_EULER_BODY`）。
EXPLICIT_EULER_STABILITY_RADIUS = 2.0

#: 本模块声明的那条步长上限，写成一句可引用的话（形制对齐
#: `integrate.IntegratorDeclaration.step_bound`与`contact_pipeline`的
#: `CONTACT_STIFFNESS_STEP_BOUND`：**上限是被声明的，不是被口头说的**）。
CONTACT_DYNAMICS_STEP_BOUND = (
    "h < stability_radius / max(法向刚度模态速率, 切向阻尼模态速率)，其中"
    "两族速率都由**六自由度**接触点迁移率算子`G·Σ Jᵢᵀ P Jᵢ`的特征值给出"
    "（`Jᵢ = [I₃ | −[rᵢ]×]`、`G = diag(I₃/m, I⁻¹)`、`P`取`n̂n̂ᵀ`或`I₃−n̂n̂ᵀ`）："
    "法向族`ω0 = √(1000·k_n·ρ)`、`ζ = (c_n/2)·√(1000·ρ/k_n)`、"
    "速率`ω0`（ζ≤1）或`(ζ+√(ζ²−1))·ω0`（ζ>1）；切向族速率`1000·k_t·σ`。"
    "**这是力矩装配下的模态，与`contact_pipeline.CONTACT_STIFFNESS_STEP_BOUND`"
    "那条罚刚度模态、与`rigidbody`那条`h < 2.785/|ω|_max`转动模态都是独立的，"
    "实际步长取更紧的那个**（见`governing_assembly_step_bound`）"
)

#: 丙2：`k_t`的统一上限声明。**它与上面那条是同一个不等式的两种解法**——
#: 上面解`h`，这里解`k_t`。
TANGENTIAL_COEFFICIENT_BOUND = (
    "k_t ≤ stability_radius / (1000·h·σ_max)——它与`CONTACT_DYNAMICS_STEP_BOUND`"
    "的切向族是**同一个不等式**，一个解`h`一个解`k_t`。其中`σ_max`是切向接触点迁移率"
    "`G·Σ Jᵢᵀ (I₃−n̂n̂ᵀ) Jᵢ`的最大特征值（单位1/kg）。"
    "**违反它的症状有两种、判据只有一条**：切向力没被摩擦锥截住时是当场发散，"
    "被截住时是钉在锥上的极限环（『颤振』）——两者都是同一个`λh > 稳定区半径`。"
    "本上限是**保守的**：实测发散/颤振起点在`λh ≈ 3.08`（翻倒）与`λh ≈ 3.22`"
    "（滚球），比2.785松10.6%与15.7%（饱和支不再是线性阻尼器，见决策0087第三节）。"
    "另有一条**下限**（`creep_resolution_lower_bound`）：速度型摩擦的稳态蠕滑是"
    "`|v_t| = F_req/k_t`，要它相对参考速度足够小就要`k_t`足够大。"
    "**两条夹出来的窗口可以是空的**，那时候要改的是`h`不是`k_t`"
)


def _skew_times(lever: Vector3, column: int) -> Vector3:
    """``[r]×``的第``column``列。写成取列而不是建矩阵：下面只按列用它。"""

    basis: list[float] = [0.0, 0.0, 0.0]
    basis[column] = 1.0
    return cross(lever, (basis[0], basis[1], basis[2]))


def _cholesky3(matrix: Matrix3) -> Matrix3:
    """对称正定3×3的下三角Cholesky因子``L``（``L·Lᵀ = matrix``）。**闭式，不迭代。**

    非正定当场失败关闭：一个非正定的惯量张量不对应任何质量分布，
    而`RigidBodyInertia.__post_init__`已经在入口挡过一次；这里是第二道，
    因为本模块也收裸的3×3（调用方可能没走那个类）。
    """

    lower = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0.0 or not math.isfinite(total):
                    raise ContactDynamicsError(
                        f"惯量张量不是正定的（Cholesky第{i}个主元为{total!r}）——"
                        "它不对应任何质量分布，上限无从谈起"
                    )
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return (
        (lower[0][0], lower[0][1], lower[0][2]),
        (lower[1][0], lower[1][1], lower[1][2]),
        (lower[2][0], lower[2][1], lower[2][2]),
    )


def _invert_lower3(lower: Matrix3) -> Matrix3:
    """下三角3×3的逆（仍是下三角）。前代，闭式。"""

    inverse = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        inverse[i][i] = 1.0 / lower[i][i]
        for j in range(i):
            inverse[i][j] = (
                -sum(lower[i][k] * inverse[k][j] for k in range(j, i)) / lower[i][i]
            )
    return (
        (inverse[0][0], inverse[0][1], inverse[0][2]),
        (inverse[1][0], inverse[1][1], inverse[1][2]),
        (inverse[2][0], inverse[2][1], inverse[2][2]),
    )


#: Jacobi扫的停机判据：非对角Frobenius范数相对对角范数的阈值，与最大扫数。
#: **两个都写在这里而不是埋在函数里**：`rigidbody._symmetric_eigenvalues`
#: 特意避开了迭代，理由是"不需要声明迭代到什么时候停"。6×6没有闭式，
#: 于是这条声明躲不掉——**那就把它写在明处，并且失败关闭**。
JACOBI_OFF_DIAGONAL_REL_TOL = 1.0e-15
JACOBI_MAX_SWEEPS = 60


def _symmetric_eigenvalues6(matrix: list[list[float]]) -> tuple[float, ...]:
    """对称6×6的六个特征值，降序。循环Jacobi。

    **停机判据与最大扫数是声明**（见上面两个常量），扫满不收敛**失败关闭**——
    一个"尽力而为的上限"比没有上限更坏：它看起来像一条门。
    """

    size = 6
    a = [row[:] for row in matrix]
    diagonal_norm = math.sqrt(sum(a[i][i] * a[i][i] for i in range(size)))
    threshold = JACOBI_OFF_DIAGONAL_REL_TOL * max(diagonal_norm, 1.0)
    for _ in range(JACOBI_MAX_SWEEPS):
        off = math.sqrt(
            sum(a[p][q] * a[p][q] for p in range(size) for q in range(p + 1, size))
        )
        if off <= threshold:
            break
        for p in range(size):
            for q in range(p + 1, size):
                if abs(a[p][q]) <= 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    else:
        raise ContactDynamicsError(
            f"Jacobi扫了{JACOBI_MAX_SWEEPS}遍还没收敛到"
            f"{JACOBI_OFF_DIAGONAL_REL_TOL!r}——不返回一个『大概是这么多』的上限"
        )
    return tuple(sorted((a[i][i] for i in range(size)), reverse=True))


@dataclass(frozen=True)
class ContactModeSpectrum:
    """一组支承点在给定构型下看到的**接触点迁移率谱**（单位1/kg），两族各一支。

    中间量全部进结果，理由与`contact_pipeline.ContactStiffnessStepBound`同源：
    **一个只给最终数字的上限，读的人无法判断它是不是算错了。**

    ## 这个量是什么

    切向阻尼力`F = −k_t·v_c`作用在杆臂`rᵢ`上，于是它同时改质心速度与角速度。
    把这两条写成一个六维一阶系统，速率算子是``1000·k_t·G·Σ Jᵢᵀ P Jᵢ``，
    其中``Jᵢ = [I₃ | −[rᵢ]×]``把体的旋量映到第`i`个接触点的速度、
    ``G = diag(I₃/m, I⁻¹)``是广义逆惯量。**迁移率就是那个算子除掉`k_t`剩下的部分。**

    ## 为什么`contact_pipeline`那条上限读不出这个量

    那一层的两端都是绑在**平动**节点上的球，没有杆臂、没有转动自由度，
    于是它的迁移率恒等于约化质量的倒数`1/m_eff`，一个标量。
    **本模块有杆臂**，于是迁移率是一个六维算子的谱——
    同一个`k_t`在"一个球"与"一个箱底四个角"上给出的速率相差两个量级。
    """

    #: 法向族（投影`n̂n̂ᵀ`）的迁移率，降序，单位1/kg。
    normal_mobilities_per_kg: tuple[float, ...]
    #: 切向族（投影`I₃ − n̂n̂ᵀ`）的迁移率，降序，单位1/kg。
    tangential_mobilities_per_kg: tuple[float, ...]
    #: 求值用的体系支承点，原样带回。
    support_points_body_mm: tuple[Vector3, ...]
    #: 求值用的**体系**接触法向；``None``表示**姿态无关的保守上确界**
    #: （取`P = I₃`，因为两个投影都`⪯ I₃`，见`contact_mode_spectrum`）。
    plane_normal_body: Vector3 | None

    @property
    def normal_mobility_per_kg(self) -> float:
        return self.normal_mobilities_per_kg[0]

    @property
    def tangential_mobility_per_kg(self) -> float:
        return self.tangential_mobilities_per_kg[0]


def _mobilities(
    points: tuple[Vector3, ...],
    mass_kg: float,
    inertia_body_kg_mm2: Matrix3,
    projector: Matrix3,
) -> tuple[float, ...]:
    """``G·Σ Jᵢᵀ P Jᵢ``的六个特征值，降序。

    对称化走``S = Cᵀ K C``（``C·Cᵀ = G``），于是六个特征值可以由对称Jacobi给出：
    ``G``分块对角、平动块是``I₃/m``、转动块是``I⁻¹ = L⁻ᵀL⁻¹``（`L`是`I`的
    Cholesky因子），所以``C = diag(I₃/√m, L⁻ᵀ)``是闭式的。
    ``eig(C Cᵀ K) = eig(Cᵀ K C)``——这不是近似，是`AB`与`BA`同谱。
    """

    stack: list[list[float]] = [[0.0] * 6 for _ in range(6)]
    for lever in points:
        #: `J`的六列：前三列是`I₃`，后三列是`−[r]×`。
        columns: list[Vector3] = []
        for axis in range(3):
            basis: list[float] = [0.0, 0.0, 0.0]
            basis[axis] = 1.0
            columns.append((basis[0], basis[1], basis[2]))
        for axis in range(3):
            column = _skew_times(lever, axis)
            columns.append((-column[0], -column[1], -column[2]))
        #: `Jᵀ P J`：先把每一列过一次`P`，再两两点乘。
        projected = [
            (
                sum(projector[0][k] * column[k] for k in range(3)),
                sum(projector[1][k] * column[k] for k in range(3)),
                sum(projector[2][k] * column[k] for k in range(3)),
            )
            for column in columns
        ]
        for i in range(6):
            for j in range(6):
                stack[i][j] += sum(columns[i][k] * projected[j][k] for k in range(3))

    root_mass = math.sqrt(mass_kg)
    lower = _cholesky3(inertia_body_kg_mm2)
    lower_inverse = _invert_lower3(lower)
    #: ``C``的转动块是``L⁻ᵀ``（上三角）：``L⁻ᵀ(L⁻ᵀ)ᵀ = L⁻ᵀL⁻¹ = I⁻¹``。
    scale = [[0.0] * 6 for _ in range(6)]
    for axis in range(3):
        scale[axis][axis] = 1.0 / root_mass
    for i in range(3):
        for j in range(3):
            scale[3 + i][3 + j] = lower_inverse[j][i]

    #: ``S = Cᵀ K C``。
    middle = [
        [sum(stack[i][k] * scale[k][j] for k in range(6)) for j in range(6)]
        for i in range(6)
    ]
    symmetric = [
        [sum(scale[k][i] * middle[k][j] for k in range(6)) for j in range(6)]
        for i in range(6)
    ]
    #: 对称化：`S`在精确算术下对称，浮点下末位可能差一位，而Jacobi要对称输入。
    for i in range(6):
        for j in range(i + 1, 6):
            averaged = 0.5 * (symmetric[i][j] + symmetric[j][i])
            symmetric[i][j] = averaged
            symmetric[j][i] = averaged
    return _symmetric_eigenvalues6(symmetric)


def _require_inertia(matrix: object) -> Matrix3:
    if not isinstance(matrix, (tuple, list)) or len(matrix) != 3:
        raise ContactDynamicsError(f"inertia_body_kg_mm2必须是3×3：{matrix!r}")
    rows = tuple(_require_vec3(row, "inertia_body_kg_mm2行") for row in matrix)
    scale = max(abs(value) for row in rows for value in row)
    if scale <= 0.0:
        raise ContactDynamicsError("inertia_body_kg_mm2全零")
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(rows[i][j] - rows[j][i]) > 1.0e-12 * scale:
                raise ContactDynamicsError(
                    f"inertia_body_kg_mm2在({i},{j})不对称：{rows[i][j]!r} vs {rows[j][i]!r}"
                )
    return (rows[0], rows[1], rows[2])


def contact_mode_spectrum(
    *,
    support_points_body_mm: tuple[Vector3, ...],
    mass_kg: float,
    inertia_body_kg_mm2: Matrix3,
    plane_normal_body: Vector3 | None = None,
) -> ContactModeSpectrum:
    """一组支承点的接触点迁移率谱。

    ``plane_normal_body``给了就按那个姿态精确求值；给``None``取
    **姿态无关的保守上确界**——两个投影都满足``P ⪯ I₃``，于是
    ``Σ Jᵢᵀ P Jᵢ ⪯ Σ JᵢᵀJᵢ``，后者不含`n̂`，因此它的谱是所有姿态的上界。

    **为什么要有姿态无关那一支**：上限必须在**整条轨迹**上成立，而支承点的
    体系坐标是常量、接触法向在体系里却随姿态转。翻倒那条案例里体真的翻过去了，
    "起手那个姿态上算出来的上限"在末态上不再成立。实测本仓那只箱子的底面四角组：
    精确值272000、姿态无关上确界307513（松13%）——**13%的保守换一条全程成立的话**。
    """

    points = tuple(
        _require_vec3(point, f"support_points_body_mm[{index}]")
        for index, point in enumerate(support_points_body_mm)
    )
    if not points:
        raise ContactDynamicsError(
            "support_points_body_mm是空的——空支承集不是『上限无穷大』，是没有声明底面"
        )
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ContactDynamicsError(f"mass_kg必须是正的有限数：{mass_kg!r}")
    inertia = _require_inertia(inertia_body_kg_mm2)

    identity: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if plane_normal_body is None:
        spectrum = _mobilities(points, mass_kg, inertia, identity)
        return ContactModeSpectrum(
            normal_mobilities_per_kg=spectrum,
            tangential_mobilities_per_kg=spectrum,
            support_points_body_mm=points,
            plane_normal_body=None,
        )
    normal = _unit(_require_vec3(plane_normal_body, "plane_normal_body"), "plane_normal_body")
    outer: Matrix3 = tuple(  # type: ignore[assignment]
        tuple(normal[i] * normal[j] for j in range(3)) for i in range(3)
    )
    tangent: Matrix3 = tuple(  # type: ignore[assignment]
        tuple(identity[i][j] - outer[i][j] for j in range(3)) for i in range(3)
    )
    return ContactModeSpectrum(
        normal_mobilities_per_kg=_mobilities(points, mass_kg, inertia, outer),
        tangential_mobilities_per_kg=_mobilities(points, mass_kg, inertia, tangent),
        support_points_body_mm=points,
        plane_normal_body=normal,
    )


@dataclass(frozen=True)
class ContactDynamicsStepBound:
    """力矩装配下的显式积分步长上限，连同算它用到的全部中间量。"""

    #: 法向罚刚度模态的无阻尼固有角频率（最紧那一支），rad/s。
    normal_omega0_rad_per_s: float
    #: 法向阻尼比（同一支）。
    normal_damping_ratio: float
    #: 法向族要挡的最快速率：ζ≤1取`ω0`，过阻尼取``(ζ+√(ζ²−1))·ω0``。
    #: **过阻尼这一支不能省**：阻尼越大最快模态越快，"加阻尼总是更稳"是错的。
    normal_stability_rate_per_s: float
    #: 切向阻尼族要挡的最快速率``1000·k_t·σ_max``，1/s。
    tangential_stability_rate_per_s: float
    #: 两族里更快的那个。
    stability_rate_per_s: float
    #: 是哪一族在管。只报数字不报出处不行——research/17第五节抓到的正是这个错。
    governed_by: str
    normal_stiffness_n_per_mm: float
    normal_damping_n_s_per_mm: float
    tangential_stiffness_n_per_mm: float
    #: 积分器的实轴稳定区半径（RK4=2.785、显式Euler=2.0）。
    stability_radius: float
    #: 上限本身，秒。
    step_bound_s: float
    spectrum: ContactModeSpectrum


def contact_dynamics_step_bound(
    *,
    support_points_body_mm: tuple[Vector3, ...],
    mass_kg: float,
    inertia_body_kg_mm2: Matrix3,
    normal_stiffness_n_per_mm: float,
    tangential_stiffness_n_per_mm: float,
    normal_damping_n_s_per_mm: float = 0.0,
    plane_normal_body: Vector3 | None = None,
    stability_radius: float = RK4_STABILITY_RADIUS,
) -> ContactDynamicsStepBound:
    """本模块那条步长上限。声明原文见`CONTACT_DYNAMICS_STEP_BOUND`。

    ## 法向族的``ω0``与``ζ``**与`contact_pipeline`逐字同形，但迁移率不同**

    那边``ω0 = √(1000·k/m_eff)``——`1/m_eff`就是两个平动球的迁移率。
    这边``ω0 = √(1000·k_n·ρ)``，`ρ`是**带杆臂的**六维迁移率的最大特征值。
    单点无杆臂时`ρ → 1/m`，两条式子逐字重合；**有杆臂时它们不重合，
    而那正是本条上限存在的理由**。

    ``ζ``的式子由``D_n = (c_n/k_n)·K_n``**恒等**地给出：法向阻尼与法向刚度
    共用同一个投影，于是两个算子严格成比例、同一组特征向量，
    ``ζ = 速率/(2ω0) = (c_n/2)·√(1000·ρ/k_n)``。
    代入单点`ρ = 1/m`退化成`contact_pipeline`那条``ζ = 1000·c/(2·m·ω0)``。
    """

    for name, value in (
        ("normal_stiffness_n_per_mm", normal_stiffness_n_per_mm),
        ("stability_radius", stability_radius),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ContactDynamicsError(f"{name}必须是正的有限数：{value!r}")
    for name, value in (
        ("tangential_stiffness_n_per_mm", tangential_stiffness_n_per_mm),
        ("normal_damping_n_s_per_mm", normal_damping_n_s_per_mm),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ContactDynamicsError(f"{name}必须是非负的有限数：{value!r}")

    spectrum = contact_mode_spectrum(
        support_points_body_mm=support_points_body_mm,
        mass_kg=mass_kg,
        inertia_body_kg_mm2=inertia_body_kg_mm2,
        plane_normal_body=plane_normal_body,
    )
    normal_mobility = spectrum.normal_mobility_per_kg
    omega0 = math.sqrt(MM_PER_M * normal_stiffness_n_per_mm * normal_mobility)
    damping_ratio = (
        0.5
        * normal_damping_n_s_per_mm
        * math.sqrt(MM_PER_M * normal_mobility / normal_stiffness_n_per_mm)
    )
    if damping_ratio <= 1.0:
        normal_rate = omega0
    else:
        root = math.sqrt(damping_ratio - 1.0) * math.sqrt(damping_ratio + 1.0)
        normal_rate = (damping_ratio + root) * omega0
    tangential_rate = (
        MM_PER_M * tangential_stiffness_n_per_mm * spectrum.tangential_mobility_per_kg
    )
    if tangential_rate > normal_rate:
        governed = "tangential_damping"
    elif normal_rate > tangential_rate:
        governed = "normal_stiffness"
    else:
        governed = "both"
    rate = max(normal_rate, tangential_rate)
    if rate <= 0.0:
        raise ContactDynamicsError(
            "两族速率都是零——支承点全都退化在过质心的那条轴上，本上限判不了这种构型"
        )
    return ContactDynamicsStepBound(
        normal_omega0_rad_per_s=omega0,
        normal_damping_ratio=damping_ratio,
        normal_stability_rate_per_s=normal_rate,
        tangential_stability_rate_per_s=tangential_rate,
        stability_rate_per_s=rate,
        governed_by=governed,
        normal_stiffness_n_per_mm=normal_stiffness_n_per_mm,
        normal_damping_n_s_per_mm=normal_damping_n_s_per_mm,
        tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
        stability_radius=stability_radius,
        step_bound_s=stability_radius / rate,
        spectrum=spectrum,
    )


def sphere_plane_step_bound(
    *,
    radius_mm: float,
    mass_kg: float,
    inertia_body_kg_mm2: Matrix3,
    normal_stiffness_n_per_mm: float,
    tangential_stiffness_n_per_mm: float,
    normal_damping_n_s_per_mm: float = 0.0,
    stability_radius: float = RK4_STABILITY_RADIUS,
) -> ContactDynamicsStepBound:
    """球那一档的步长上限。``r = −R·n̂``，与`sphere_plane_contact`同一个杆臂。

    球的杆臂在世界系里恒沿`−n̂`，于是**体系**支承点随姿态转——这看着像
    "上限随姿态变"，但球的惯量是各向同性的，`G`在任何标架下都一样，
    于是取哪个体系标架都给同一个数。**各向异性的惯量在这一档失败关闭**：
    那不是一个球，而球那一档的整套形制（单点、`r ∥ n̂`、法向力矩恒零）
    都建立在"它是球"上。
    """

    if not math.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ContactDynamicsError(f"radius_mm必须是正的有限数：{radius_mm!r}")
    inertia = _require_inertia(inertia_body_kg_mm2)
    scale = max(abs(value) for row in inertia for value in row)
    isotropic = all(
        abs(inertia[i][j] - (inertia[0][0] if i == j else 0.0)) <= 1.0e-12 * scale
        for i in range(3)
        for j in range(3)
    )
    if not isotropic:
        raise ContactDynamicsError(
            f"球那一档要求各向同性惯量，收到{inertia!r}——"
            "非球的杆臂不恒沿法向，本函数的前提不成立"
        )
    return contact_dynamics_step_bound(
        support_points_body_mm=((0.0, 0.0, -radius_mm),),
        mass_kg=mass_kg,
        inertia_body_kg_mm2=inertia,
        normal_stiffness_n_per_mm=normal_stiffness_n_per_mm,
        tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
        normal_damping_n_s_per_mm=normal_damping_n_s_per_mm,
        plane_normal_body=(0.0, 0.0, 1.0),
        stability_radius=stability_radius,
    )


@dataclass(frozen=True)
class GroupStepBound:
    """一组**可以同时承载**的支承点集合里最紧的那条上限，连同是哪一组。"""

    step_bound_s: float
    governing_group_index: int
    bounds: tuple[ContactDynamicsStepBound, ...]


def tightest_step_bound(
    support_point_groups: tuple[tuple[Vector3, ...], ...],
    **kwargs: object,
) -> GroupStepBound:
    """扫**全部声明的支承组**取最紧的一条。

    ## 一组＝"可以同时承载的一批点"，不是"全部声明过的点"

    与`contact_pipeline.stiffness_step_bound`那条"扫候选池不扫活动集"同源、
    但多一层：定步长显式积分器在撞上的那一帧才发现步长太大已经晚了，
    所以**不能只算此刻承载的那些点**；而把一只长方体的八个角**当成一组**
    又是错的另一头——八个角永远不可能同时贴同一个平面，那样算出来的上限
    比物理上可能出现的最紧模态还紧一大截（实测本仓那只箱子：
    按面分组272000、八个角当一组480000，**紧了76%**，会把一条本来跑得好好的
    案例判成不稳定）。

    **所以形制是：调用方声明若干组，每组是一批能同时承载的点，本函数取最紧的那组。**
    一只长方体就是六个面六组。
    """

    groups = tuple(tuple(group) for group in support_point_groups)
    if not groups:
        raise ContactDynamicsError("support_point_groups是空的——没有声明任何支承组")
    bounds = tuple(
        contact_dynamics_step_bound(support_points_body_mm=group, **kwargs)  # type: ignore[arg-type]
        for group in groups
    )
    tightest = min(range(len(bounds)), key=lambda index: bounds[index].step_bound_s)
    return GroupStepBound(
        step_bound_s=bounds[tightest].step_bound_s,
        governing_group_index=tightest,
        bounds=bounds,
    )


@dataclass(frozen=True)
class GoverningAssemblyStepBound:
    """力矩装配这一档的两条独立上限里更紧的那条，**连同"是哪一条在管"**。

    只报数字不报出处，读的人会以为唯一那条上限就是全部——
    而research/17第五节抓到的正是这个错：`rigidbody`看着有一条`step_bound`，
    **它只挡了一半物理**。
    """

    step_bound_s: float
    #: `contact_assembly`／`rotational_mode`／`both`（两条恰好相等时）。
    governed_by: str
    contact_assembly_bound_s: float
    rotational_mode_bound_s: float


def governing_assembly_step_bound(
    *, contact_assembly_bound_s: float, rotational_mode_bound_s: float
) -> GoverningAssemblyStepBound:
    """取两条独立上限里更紧的那个。**这就是"取更紧的那个"这句话的执行体。**

    ``rotational_mode_bound_s``是`rigidbody`那条``h < 2.785/|ω|_max``；
    ``contact_assembly_bound_s``是本模块`contact_dynamics_step_bound`给的那条。
    两条**物理上独立**：前者是自由刚体的转动模态（没有接触也在），
    后者是接触刚度与切向阻尼（体停着不转也在）。
    """

    for name, value in (
        ("contact assembly", contact_assembly_bound_s),
        ("rotational mode", rotational_mode_bound_s),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ContactDynamicsError(f"{name} step bound必须是正的有限数：{value!r}")
    if contact_assembly_bound_s < rotational_mode_bound_s:
        governed = "contact_assembly"
    elif rotational_mode_bound_s < contact_assembly_bound_s:
        governed = "rotational_mode"
    else:
        governed = "both"
    return GoverningAssemblyStepBound(
        step_bound_s=min(contact_assembly_bound_s, rotational_mode_bound_s),
        governed_by=governed,
        contact_assembly_bound_s=contact_assembly_bound_s,
        rotational_mode_bound_s=rotational_mode_bound_s,
    )


@dataclass(frozen=True)
class TangentialCoefficientWindow:
    """`k_t`的可用窗口：显式稳定给上界、蠕滑分辨力给下界。声明原文见
    `TANGENTIAL_COEFFICIENT_BOUND`。

    **窗口可以是空的**，而那是一条真结论不是一次失败：它说的是
    "这个步长下，既想稳又想粘住是办不到的，要改的是`h`"。
    实测本仓那只箱子在`dt = 4e-4`上正是空的（上界0.02560、下界0.04627），
    案例最后的办法也正是把步长压到`2e-4`。
    """

    #: 显式稳定上界，N·s/mm。
    upper_bound_n_s_per_mm: float
    #: 蠕滑分辨力下界，N·s/mm；调用方没给参考量时是``None``。
    lower_bound_n_s_per_mm: float | None
    #: 切向迁移率`σ_max`，1/kg。
    tangential_mobility_per_kg: float
    #: 定这条上界的步长，秒。
    step_s: float
    stability_radius: float
    spectrum: ContactModeSpectrum

    @property
    def is_empty(self) -> bool:
        """下界比上界还大——**这个步长下没有可用的`k_t`**。"""

        return (
            self.lower_bound_n_s_per_mm is not None
            and self.lower_bound_n_s_per_mm > self.upper_bound_n_s_per_mm
        )

    def admits(self, tangential_stiffness_n_s_per_mm: float) -> bool:
        """给定的`k_t`在窗口里吗。**判的是`≤`不是`<`**：上界本身是允许的那一档，
        与`contact_pipeline`那条"上限是`h < bound`"的口径差一个端点，
        因此这里写成方法而不是让调用方自己比——两处口径各写一遍就会漂。"""

        if tangential_stiffness_n_s_per_mm > self.upper_bound_n_s_per_mm:
            return False
        if self.lower_bound_n_s_per_mm is None:
            return True
        return tangential_stiffness_n_s_per_mm >= self.lower_bound_n_s_per_mm


def creep_resolution_lower_bound(
    *, required_tangential_force_n: float, allowed_creep_mm_per_s: float
) -> float:
    """速度型库仑摩擦的蠕滑分辨力下界``k_t ≥ F_req / v_allowed``。

    速度型摩擦在真正的静止粘着上有一个**残余滑移**``|v_t| = F_req/k_t``
    （`sphere_plane_contact`的docstring已经写明它不是零）。要它小到能被叫作
    "粘住"，`k_t`就必须大——**这是一条下界，方向与显式稳定那条上界相反**。

    `required_tangential_force_n`是那一步真正要传的切向力（滚球那条是
    ``(2/7)·W·sinθ``、翻倒那条是重力沿坡分量减掉已饱和那几点的锥值）；
    `allowed_creep_mm_per_s`是调用方愿意叫作"粘住"的最大蠕滑速率。
    **两个都必须显式给**：没有一个普适的"够小"。
    """

    if not math.isfinite(required_tangential_force_n) or required_tangential_force_n <= 0.0:
        raise ContactDynamicsError(
            f"required_tangential_force_n必须是正的有限数：{required_tangential_force_n!r}"
        )
    if not math.isfinite(allowed_creep_mm_per_s) or allowed_creep_mm_per_s <= 0.0:
        raise ContactDynamicsError(
            f"allowed_creep_mm_per_s必须是正的有限数：{allowed_creep_mm_per_s!r}"
        )
    return required_tangential_force_n / allowed_creep_mm_per_s


def tangential_coefficient_window(
    *,
    support_points_body_mm: tuple[Vector3, ...],
    mass_kg: float,
    inertia_body_kg_mm2: Matrix3,
    step_s: float,
    plane_normal_body: Vector3 | None = None,
    stability_radius: float = RK4_STABILITY_RADIUS,
    required_tangential_force_n: float | None = None,
    allowed_creep_mm_per_s: float | None = None,
) -> TangentialCoefficientWindow:
    """`k_t`的统一上限声明（丙2）。

    上界``k_t ≤ stability_radius/(1000·h·σ_max)``——它与
    `contact_dynamics_step_bound`那条切向族是**同一个不等式**，
    一个解`h`一个解`k_t`。

    ## 两条实测硬边界为什么是同一条判据

    [0082](../../docs/decisions/0082_翻倒进多支承点接触_底面是一组物质点_20260818.md)
    第五节记的是"翻倒那条是显式稳定率"、`cases/rolling_ball_incline`第四节第1条
    记的是"滚球那条是颤振区"，两页都写着它们**理由相反**。实测下来它们是
    **同一个`λh > 稳定区半径`**，差别只在症状：

    * 切向力**没有**被摩擦锥截住时，越界的模态直接发散（翻倒那条，
      横坡角速度从舍入量级长到0.1 rad/s）；
    * **被截住**时，力钉在`μ|F_n|`上，幅值涨不上去，于是变成绕着摩擦锥的
      极限环——那就是"颤振区"，`sliding`恒为真、平均加速度还对
      （滚球那条，实测`k_t = 5e3`时平均加速度偏差仍只有2.4e-4）。

    **哪一条更紧由`σ_max`决定，不由症状决定**：同一个`k_t`，一个球的
    `σ_max = 1/m + R²/I`，一只箱子四个角的`σ_max`要大两个量级
    （实测3500 vs 272000，1/kg）。
    """

    if not math.isfinite(step_s) or step_s <= 0.0:
        raise ContactDynamicsError(f"step_s必须是正的有限数：{step_s!r}")
    if not math.isfinite(stability_radius) or stability_radius <= 0.0:
        raise ContactDynamicsError(f"stability_radius必须是正的有限数：{stability_radius!r}")
    spectrum = contact_mode_spectrum(
        support_points_body_mm=support_points_body_mm,
        mass_kg=mass_kg,
        inertia_body_kg_mm2=inertia_body_kg_mm2,
        plane_normal_body=plane_normal_body,
    )
    mobility = spectrum.tangential_mobility_per_kg
    if mobility <= 0.0:
        raise ContactDynamicsError(
            "切向迁移率是零——支承点全在过质心的法向轴上，本上限判不了这种构型"
        )
    if (required_tangential_force_n is None) != (allowed_creep_mm_per_s is None):
        raise ContactDynamicsError(
            "蠕滑下界的两个参数要么都给要么都不给——"
            "只给一个等于让本函数替你猜另一个，而没有一个普适的『够小』"
        )
    lower: float | None = None
    if required_tangential_force_n is not None and allowed_creep_mm_per_s is not None:
        lower = creep_resolution_lower_bound(
            required_tangential_force_n=required_tangential_force_n,
            allowed_creep_mm_per_s=allowed_creep_mm_per_s,
        )
    return TangentialCoefficientWindow(
        upper_bound_n_s_per_mm=stability_radius / (MM_PER_M * step_s * mobility),
        lower_bound_n_s_per_mm=lower,
        tangential_mobility_per_kg=mobility,
        step_s=step_s,
        stability_radius=stability_radius,
        spectrum=spectrum,
    )


__all__ = [
    "CONTACT_DYNAMICS_STEP_BOUND",
    "EXPLICIT_EULER_STABILITY_RADIUS",
    "JACOBI_MAX_SWEEPS",
    "JACOBI_OFF_DIAGONAL_REL_TOL",
    "RK4_STABILITY_RADIUS",
    "TANGENTIAL_COEFFICIENT_BOUND",
    "ContactDynamicsError",
    "ContactDynamicsStepBound",
    "ContactModeSpectrum",
    "ContactResponse",
    "GoverningAssemblyStepBound",
    "GroupStepBound",
    "SupportPointContact",
    "SupportSetResponse",
    "TangentialCoefficientWindow",
    "box_corner_points_mm",
    "contact_dynamics_step_bound",
    "contact_mode_spectrum",
    "creep_resolution_lower_bound",
    "governing_assembly_step_bound",
    "sphere_plane_callbacks",
    "sphere_plane_contact",
    "sphere_plane_step_bound",
    "support_points_plane_callbacks",
    "support_points_plane_contact",
    "tangential_coefficient_window",
    "tightest_step_bound",
]
