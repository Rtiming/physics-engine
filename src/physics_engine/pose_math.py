"""域中立的刚体位姿组合——只处理frame关系，不处理状态或力。

``Pose``统一表示``parent_from_child``。本模块只提供这一种组合：
``parent_from_middle ∘ middle_from_child -> parent_from_child``。它放在scene基座，
避免模型装配为了一个四元数组合反向import力学域的``rigidbody``。
"""

from __future__ import annotations

from physics_engine.motion import Pose

IDENTITY_POSE = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class PoseMathError(ValueError):
    """位姿组合的类型或frame语义错误。"""


def _rotate(rotation_xyzw: tuple[float, float, float, float], vector: tuple[float, ...]):
    x, y, z, w = rotation_xyzw
    vx, vy, vz = vector
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def _multiply(
    left_xyzw: tuple[float, float, float, float],
    right_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = left_xyzw
    x2, y2, z2, w2 = right_xyzw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def compose_pose(parent_from_middle: Pose, middle_from_child: Pose) -> Pose:
    """组合两个相邻frame变换，返回``parent_from_child``。

    平移必须先由``parent_from_middle``的旋转搬到parent轴，再加parent平移；
    旋转按Hamilton积``q_parent_middle ⊗ q_middle_child``组合。两步的次序都是
    坐标系语义，不能交换。
    """

    if not isinstance(parent_from_middle, Pose) or not isinstance(middle_from_child, Pose):
        raise PoseMathError("compose_pose expects two Pose values")
    rotated = _rotate(
        parent_from_middle.rotation_xyzw, middle_from_child.translation_mm
    )
    translation = tuple(
        parent_from_middle.translation_mm[index] + rotated[index] for index in range(3)
    )
    return Pose(
        translation_mm=translation,  # type: ignore[arg-type]
        rotation_xyzw=_multiply(
            parent_from_middle.rotation_xyzw, middle_from_child.rotation_xyzw
        ),
    )


__all__ = ["IDENTITY_POSE", "PoseMathError", "compose_pose"]
