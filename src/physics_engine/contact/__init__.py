"""接触层——决策0050的形制落地（第一片：**锚点布局**，力学域）。

本片**不算任何物理**。它只回答一个问题：粘着锚点放在状态的什么位置。
理由见0050第一节——接触的形制一旦定错后面全要返工，
而锚点的位置是形制里最先被别的东西压上的那一块。

## 0033那道题，以及为什么这里不是三条路里的任何一条

0033裁过：粘着锚点**是真历史**（"这一点现在粘着还是滑动"不能从当前位形算出来，
只能从历史知道），必须进状态并随状态被复现。
0043随后实测确认`StateLayout`装不下它，并列了三条路，**代价都很硬**：
每种活动集一份布局（跨步守恒断言全废）、可变长段（`fingerprint()`语义要重定义，
而那是0019的承重条款）、锚点留状态外（0033已否）。

**三条路的分母都取错了**：它们假定锚点数由**活动**接触集决定。
而接触对是**声明**出来的——`scene.ContactPair`是冻结、有序的，声明期就定死。

**按声明的对分槽，布局就是定长的**：指纹跨步不变、`fingerprint()`语义一个字不用改、
跨步守恒量断言继续成立。**活动与否变成向量里的一个值，不是一次布局变更。**

代价如实写在0050第一节：自由度按**声明**的对数算而不是按活动的对数算。
它是可接受的，因为`declare_contact_between`本来就是显式声明——
**声明的对数是使用者控制的量**，本仓从不做"任意两个体都可能碰"的全局接触。

## 声明从哪来：今天由调用方给，**不由本模块去问`Scene`**

`FinalizedScene.contact_pairs`是**将来**的来源，但本片不去接它。
理由是三前提第二条：把场景里的"体"映射到力学状态里的"节点"是另一个未解的问题
（`AssembledBody`是位姿+几何，不是自由度），**现在接等于替一个还不存在的
消费方预支一套映射**。

**触发条件**：第一个既走`Scene`装配又走能量求解的案例出现时，那时映射的形状才清楚。
本模块只要求调用方给出**已冻结、有序**的声明——那是0050真正依赖的性质，
而不是"声明必须来自`Scene`"。

## 本片顺带关掉的一个洞

`energies.resolve_node_count`（本轮上一片）只看得见``len(vector)``，
**看不见"布局里哪一段是节点块"**——于是一份声明了3个质量而布局只有2个节点的
上下文，重力仍会落到锚点上。那条洞当时如实登记了，触发条件写的正是
"**锚点布局构造器进仓时把边界带上来**"。

`ContactLayout`知道边界（它就是造边界的那个东西），
故`assert_matches_context`在这里把第二次比对补上。

## 本子包的文件划分（2026-08-17拆自单文件`contact.py`）

原文件2157行，接下来有三件活同时落在它里面（各向异性摩擦、多轮路由、
转动自由度进接触）。**并行的代价不在合并冲突，在读文件的开销**——
改200行要先读2157行。于是按"谁依赖谁"切开，依赖是单向的：

    errors ← layout ← {penalty, friction} ← {damping, stepper}

| 文件 | 装什么 |
|---|---|
| `errors.py` | `ContactError`。谁也不依赖，故不产生环 |
| `layout.py` | 槽宽/每对槽数/regime取值/法向来源类型＋声明与锚点布局 |
| `penalty.py` | 罚法向四族：半空间、球-球、有限长圆柱侧面、法兰内环面 |
| `friction.py` | 库仑return-map（**圆与椭圆两条并列**，决策0068）与粘着弹簧 |
| `damping.py` | 恢复系数↔阻尼比换算与线性法向dashpot |
| `stepper.py` | 准静态步进器：单槽位与多槽位 |

**拆分本身不改任何物理**：全部函数体逐字节搬运，公开名经本文件原样再导出，
`from physics_engine.contact import X`一个字不用改。
"""

from __future__ import annotations

from physics_engine.contact.damping import (
    LinearDashpotParameters,
    LinearNormalDashpot,
    damping_ratio_from_restitution,
    linear_dashpot_parameters,
    restitution_from_damping_ratio,
)
from physics_engine.contact.errors import ContactError
from physics_engine.contact.friction import (
    IN_PLANE_DIRECTION_MIN_SINE,
    TRIAL_OUT_OF_PLANE_TOLERANCE,
    FrictionEllipse,
    FrictionOutcome,
    TangentialStickSpring,
    anisotropic_return_map,
    coulomb_return_map,
)
from physics_engine.contact.layout import (
    MAX_POINTS_PER_PAIR_SPHERE_CAPSULE,
    NORMAL_UNIT_TOLERANCE,
    REGIME_SEPARATED,
    REGIME_SLIP,
    REGIME_STICK,
    SLOT_WIDTH,
    ContactDeclaration,
    ContactLayout,
    ContactSlot,
    NormalSource,
    build_contact_layout,
)
from physics_engine.contact.penalty import (
    PenaltyAnnulusLimit,
    PenaltyCylinderContact,
    PenaltyNormalContact,
    PenaltySphereContact,
)
from physics_engine.contact.stepper import (
    ContactPoint,
    ContactStep,
    MultiContactStep,
    advance_contact_quasistatic,
    advance_contacts_quasistatic,
)

__all__ = [
    "IN_PLANE_DIRECTION_MIN_SINE",
    "MAX_POINTS_PER_PAIR_SPHERE_CAPSULE",
    "NORMAL_UNIT_TOLERANCE",
    "REGIME_SEPARATED",
    "REGIME_SLIP",
    "REGIME_STICK",
    "SLOT_WIDTH",
    "TRIAL_OUT_OF_PLANE_TOLERANCE",
    "ContactDeclaration",
    "ContactError",
    "ContactLayout",
    "ContactPoint",
    "ContactSlot",
    "ContactStep",
    "FrictionEllipse",
    "FrictionOutcome",
    "LinearDashpotParameters",
    "LinearNormalDashpot",
    "MultiContactStep",
    "NormalSource",
    "PenaltyAnnulusLimit",
    "PenaltyCylinderContact",
    "PenaltyNormalContact",
    "PenaltySphereContact",
    "TangentialStickSpring",
    "advance_contact_quasistatic",
    "advance_contacts_quasistatic",
    "anisotropic_return_map",
    "build_contact_layout",
    "coulomb_return_map",
    "damping_ratio_from_restitution",
    "linear_dashpot_parameters",
    "restitution_from_damping_ratio",
]
