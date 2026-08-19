"""六个罚接触族共用的那一小段：边缘几何的容差与jet构造。

2026-08-19从`penalty.py`拆出来（见`__init__.py`的拆分说明）。
**这里只放真正被两个以上族用到的东西**——一个"公共模块"一旦开始收留
"看起来通用"的东西，它就会变成第二个大文件。
"""

from __future__ import annotations

from physics_engine.autodiff import Jet1, Jet2
from physics_engine.contact.errors import ContactError

#: 带宽方向平分线的最短可用长度。两条材料帧几乎反向时平分线由舍入决定，
#: 那不是方向是噪声。与`rod.WIDTH_DIRECTION_MIN_LENGTH`同值同理由，
#: **但不从那里import**——0072第3.2节裁定`contact → rod`这条import边不开。
EDGE_WIDTH_MIN_LENGTH = 1.0e-6
#: 材料帧两条轴的正交性容差（``|d1·d2|``）。与`rod.FRAME_TOLERANCE`同值。
EDGE_FRAME_TOLERANCE = 1.0e-10


def _edge_jets(vector: tuple[float, ...], indices: tuple[int, ...], order: int):
    """局部变量的jet。``order``为0时是裸float——**活动判定走这一档**。

    与`rod._local_jets`同形。**不从`rod`import**（同上一条注释的理由），
    而`tests/test_contact_annulus_twist.py`有一条门拿两边的``m̂2``逐位对拍，
    重复实现的漂移由那条门守。
    """

    if order == 0:
        return tuple(vector[index] for index in indices)
    if order == 1:
        return tuple(
            Jet1.variable(vector[index], slot, len(indices))
            for slot, index in enumerate(indices)
        )
    if order == 2:
        return tuple(
            Jet2.variable(vector[index], slot, len(indices))
            for slot, index in enumerate(indices)
        )
    raise ContactError(f"derivative order must be 0, 1 or 2: {order!r}")
