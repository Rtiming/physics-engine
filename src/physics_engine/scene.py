"""场景文件——引擎数据层入口（research/02第1层，M-E3第一片）。

学的是谁、学了什么（decisions/0011）：

* **MuJoCo/Gazebo**：场景内容100%住在文件里，库源码零改动；
* **Gazebo**：连"要加载什么扩展"都写在世界文件里——本格式的`extensions`
  照此形制：声明模块名，加载后新形状种类进注册表，未声明的种类失败关闭；
* **FTS**：scene.json先例（JSON+严格解析+规范化哈希，正合轴3）；
* **WDS design/16 §5.1**：禁止include/extends/环境变量替换——闭包纪律
  比MJCF的include更严，这是"更规范"的部分；
* 我们自己的加强：场景文件是登记的面（轴1，出生draft）、内容寻址（轴3）、
  产物走run package（轴4/5）。

安全模型与Gazebo插件相同：`extensions`声明的模块会被import执行——
场景文件是估算内的受信输入（内容寻址+入库），不是任意来源的数据。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from physics_engine.canonical import FTS_PROFILE, canonical_sha256, strict_loads
from physics_engine.engine_facets import ENGINE_REGISTRY
from physics_engine.shapes import (
    Capsule,
    CollisionShape,
    FiniteCylinder,
    GeneratedShape,
    MeshAsset,
    PosedBody,
    RoundedBox,
    ShapeError,
    SimBody,
    Sphere,
    VisualShape,
)

#: 场景面的规范化声明：ensure_ascii、带尾换行（与FTS场景先例同参）。
SCENE_CANONICAL_PROFILE = FTS_PROFILE

SCENE_FACET = "physics_scene"
SCENE_FACET_VERSION = "1.0.0"


class SceneError(ValueError):
    """场景文件的一切失败关闭。"""


#: 形状种类注册表——内建种类+扩展声明加载的种类（Gazebo形制+WDS注册表纪律）。
SHAPE_KINDS: dict[str, Callable[..., Any]] = {}


def register_shape_kind(kind: str, constructor: Callable[..., Any]) -> None:
    """登记一个形状种类。重复登记失败关闭（新增能力必须是自觉的改动）。"""

    if kind in SHAPE_KINDS:
        raise SceneError(f"shape kind already registered: {kind}")
    SHAPE_KINDS[kind] = constructor


def _mesh(**kwargs: Any) -> MeshAsset:
    kwargs["aabb_min_mm"] = tuple(kwargs["aabb_min_mm"])
    kwargs["aabb_max_mm"] = tuple(kwargs["aabb_max_mm"])
    return MeshAsset(**kwargs)


def _generated(**kwargs: Any) -> GeneratedShape:
    inner = kwargs.pop("shape")
    kwargs["parameters"] = tuple((k, float(v)) for k, v in kwargs.get("parameters", ()))
    return GeneratedShape(shape=_build_shape(inner), **kwargs)


def _capsule(**kwargs: Any) -> Capsule:
    kwargs["point_a_mm"] = tuple(kwargs["point_a_mm"])
    kwargs["point_b_mm"] = tuple(kwargs["point_b_mm"])
    return Capsule(**kwargs)


def _rounded_box(**kwargs: Any) -> RoundedBox:
    kwargs["half_extents_mm"] = tuple(kwargs["half_extents_mm"])
    return RoundedBox(**kwargs)


for _kind, _constructor in (
    ("sphere", Sphere),
    ("capsule", _capsule),
    ("rounded_box", _rounded_box),
    ("finite_cylinder", FiniteCylinder),
    ("mesh", _mesh),
    ("generated", _generated),
):
    register_shape_kind(_kind, _constructor)


def _build_shape(declaration: Any) -> Any:
    if not isinstance(declaration, dict) or "kind" not in declaration:
        raise SceneError("shape declaration requires a 'kind' field")
    fields = dict(declaration)
    kind = fields.pop("kind")
    constructor = SHAPE_KINDS.get(kind)
    if constructor is None:
        raise SceneError(
            f"unknown shape kind {kind!r}; built-ins are {sorted(SHAPE_KINDS)} — "
            "third-party kinds must be declared in the scene's 'extensions'"
        )
    try:
        return constructor(**fields)
    except TypeError as error:
        raise SceneError(f"invalid parameters for shape kind {kind!r}: {error}") from error


_TOP_KEYS = {
    "contract_type", "contract_version", "scene_id", "description",
    "extensions", "bodies", "allowed_pairs",
}
_BODY_KEYS = {"body_id", "collision", "visual", "pose", "mass_kg"}


@dataclass(frozen=True)
class Scene:
    scene_id: str
    source_sha256: str
    posed_bodies: tuple[PosedBody, ...]
    allowed_pairs: frozenset[frozenset[str]]


def load_scene(payload: bytes) -> Scene:
    """严格加载：未知键、未知种类、重复体、坏位姿一律失败关闭。"""

    document = strict_loads(payload)
    if not isinstance(document, dict):
        raise SceneError("a scene file must be a JSON object")
    unknown = set(document) - _TOP_KEYS
    if unknown:
        raise SceneError(f"unknown top-level keys: {sorted(unknown)}")
    if document.get("contract_type") != SCENE_FACET:
        raise SceneError(f"contract_type must be {SCENE_FACET!r}")
    ENGINE_REGISTRY.assert_reader_compatible(
        SCENE_FACET, str(document.get("contract_version", ""))
    )
    scene_id = document.get("scene_id", "")
    if not isinstance(scene_id, str) or not scene_id.startswith("scene/"):
        raise SceneError("scene_id must be namespaced like 'scene/...'")

    for module_name in document.get("extensions", ()):
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            raise SceneError(f"declared extension not importable: {module_name}") from error

    bodies_field = document.get("bodies")
    if not isinstance(bodies_field, list) or not bodies_field:
        raise SceneError("a scene requires a nonempty 'bodies' list")
    posed: list[PosedBody] = []
    for entry in bodies_field:
        unknown_body = set(entry) - _BODY_KEYS
        if unknown_body:
            raise SceneError(f"unknown body keys: {sorted(unknown_body)}")
        if "collision" not in entry:
            raise SceneError(f"body {entry.get('body_id')!r} requires a collision declaration")
        collision_field = dict(entry["collision"])
        try:
            collision = CollisionShape(
                shape=_build_shape(collision_field.pop("shape", None)),
                direction=collision_field.pop("direction", None),
            )
        except TypeError as error:
            raise SceneError(f"invalid collision declaration: {error}") from error
        except ShapeError as error:
            # 缺省的direction会以None走到这里——spec/11规则5"保守方向缺省禁止"
            # 对写场景文件的人同样成立，不只对写Python的人。
            raise SceneError(f"invalid collision declaration: {error}") from error
        if collision_field:
            raise SceneError(f"unknown collision keys: {sorted(collision_field)}")
        visual = None
        if "visual" in entry:
            visual = VisualShape(shape=_build_shape(entry["visual"].get("shape")))
        pose = entry.get("pose", {})
        try:
            posed.append(
                PosedBody(
                    body=SimBody(
                        body_id=entry.get("body_id", ""),
                        collision=collision,
                        visual=visual,
                        mass_kg=entry.get("mass_kg"),
                    ),
                    translation_mm=tuple(pose.get("translation_mm", (0.0, 0.0, 0.0))),
                    rotation_xyzw=tuple(pose.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))),
                )
            )
        except ShapeError as error:
            raise SceneError(str(error)) from error

    # 装配期统一校验（spec/10第3条：finalize统一校验、配错当场炸）。
    # 这两条此前推迟到BroadPhaseCollisionQuery构造期才炸，后果是
    # `pe-scene validate`对非法场景报valid并以0退出。
    identifiers = [entry.body.body_id for entry in posed]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates:
        raise SceneError(f"duplicate body_id in scene: {duplicates}")

    pairs_field = document.get("allowed_pairs", [])
    allowed = frozenset(
        frozenset(pair) for pair in pairs_field
    )
    known = set(identifiers)
    for pair in allowed:
        if len(pair) != 2:
            raise SceneError(f"allowed pair must name two distinct bodies: {sorted(pair)}")
        unknown_members = sorted(pair - known)
        if unknown_members:
            raise SceneError(f"allowed pair references unknown bodies: {unknown_members}")

    return Scene(
        scene_id=scene_id,
        source_sha256=canonical_sha256(document, SCENE_CANONICAL_PROFILE),
        posed_bodies=tuple(posed),
        allowed_pairs=allowed,
    )


__all__ = [
    "SCENE_CANONICAL_PROFILE",
    "SCENE_FACET",
    "SCENE_FACET_VERSION",
    "SHAPE_KINDS",
    "Scene",
    "SceneError",
    "load_scene",
    "register_shape_kind",
]
