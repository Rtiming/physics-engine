"""罚法向接触的六个几何族：半空间、球-球、有限长圆柱侧面、法兰内环面、扫掠槽壁两档。

六个都只算**法向**，切向在`friction`。六个共享同一条纪律：
罚势的精度地板是``k·ulp(间隙表达式里被相减的那个量)``——半空间那条
"跨六个数量级一个ulp都不动"之所以成立，是因为它那个量恰好是0；
圆柱是``k·ulp(R)``、环带是``k·ulp(W/2)``、扫掠槽壁是``k·ulp(w/2)``。见plans/13。

拆分自原`contact.py`（2026-08-17）——**函数体逐字节未动**。
第五族`PenaltyGrooveSweep`是2026-08-18新增（决策0075），
**前四族一个字节未动**：退化逐位门判的正是它与`PenaltyAnnulusLimit`的逐位相等。

`PenaltyAnnulusLimit`（第四族）2026-08-18由决策0088丁2**接上扭转**：
边缘点的偏移方向可以随边扭角γ转，局部模板从3个变量开到5个``[x(3), γ_l, γ_r]``。
**不声明``edge_twists``时走的仍是原来那串代码，一个字节没动**——
逐位退化门判的正是它。

第六族`PenaltyGrooveSweepLive`是2026-08-18同日新增（决策0078）：
它是第五族的**活站点**档，把冻结帧丢掉的``A·t``补回去——**办法是改能量而不是改梯度**。
**第五族又一个字节未动**：两档并存，冻结帧仍是默认，逐位退化那条链条对两档都成立。

## 2026-08-19：本模块从一个文件拆成一个包

拆之前是**2378行、114 KB**，装着六个族。拆的判据**不是"大就该拆"**，
是2026-08-17拆`contact.py`那次自己写下的那一条：

> **并行的代价不在合并冲突，在读文件的开销**——改200行要先读2157行。

`penalty.py`今天**2378行，比触发上一次拆分的那个数还大**。而这一批（基础设施批次二）
刚实测到那条代价的第二种形态：轨丁独占本文件，于是同期要往接触里加距离场的轨甲
**只能另开`contact/field.py`**——那次是对的，但它对的理由是"新族本来就该独立成文件"，
不是"本文件不该拆"。

**拆法**：一族一个文件，共享的那一小段（边缘几何的容差与jet构造）进`_edges.py`。
本模块退成**只做再导出**，于是所有既有的
`from physics_engine.contact.penalty import X`与`from physics_engine.contact import X`
**一个字都不用改**。

**硬约束与0083的M7同一条：既有产物逐位不变。** 验收走仓里现成的两把尺子——
全套测试（含几十条`float.hex()`逐位断言）与40个案例生成器重跑逐字节复现，
**两者都覆盖真实构型，比手搓一组随机构型硬**。

**为什么不顺手接那条断边**（`contact_pipeline` → `contact_dynamics`）：0083第6.4节
裁过且理由没变——接它要先裁"球-球两个都在动时杆臂各取谁的半径"，那件事今天没裁，
且没有任何消费方同时要两条路。**本次只搬文件，不动依赖方向。**
"""

from __future__ import annotations

from physics_engine.contact.penalty._edges import (
    EDGE_FRAME_TOLERANCE,
    EDGE_WIDTH_MIN_LENGTH,
)
from physics_engine.contact.penalty.annulus import PenaltyAnnulusLimit
from physics_engine.contact.penalty.cylinder import PenaltyCylinderContact
from physics_engine.contact.penalty.groove import (
    PenaltyGrooveSweep,
    groove_sweep_walls,
)
from physics_engine.contact.penalty.groove_live import (
    ARC_SOLVE_ITERATIONS,
    ARC_SOLVE_TOL_MM,
    STRAIGHT_DARBOUX,
    PenaltyGrooveSweepLive,
    groove_sweep_live_walls,
)
from physics_engine.contact.penalty.halfspace import PenaltyNormalContact
from physics_engine.contact.penalty.sphere import PenaltySphereContact

__all__ = [
    "ARC_SOLVE_ITERATIONS",
    "ARC_SOLVE_TOL_MM",
    "EDGE_FRAME_TOLERANCE",
    "EDGE_WIDTH_MIN_LENGTH",
    "STRAIGHT_DARBOUX",
    "PenaltyAnnulusLimit",
    "PenaltyCylinderContact",
    "PenaltyGrooveSweep",
    "PenaltyGrooveSweepLive",
    "PenaltyNormalContact",
    "PenaltySphereContact",
    "groove_sweep_live_walls",
    "groove_sweep_walls",
]
