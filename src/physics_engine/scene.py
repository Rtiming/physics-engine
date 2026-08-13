"""场景：文件加载器（数据层入口）**加上**装配容器（spec/10第一节）。

本模块有两个入口，**互不替代**：

* ``load_scene(payload)``——从场景文件字节建一个``Scene``。数据层入口
  （research/02第1层，M-E3第一片），消费方是``pe-scene``与``examples/``；
* ``SceneAssembly``/``finalize()``——**在Python里把多个体装成一个场景**
  （spec/10第一节，决策0045）。它是**加上去的一层，不是对加载器的替换**：
  场景文件表达不了运动源与驱动器（那两样是对象不是字节），而
  ``MotionSource``一进来，"场景"就不再是一份静态位姿表了。

两条路各自的位置写在这里，免得下一个人以为其中一条该被另一条吃掉：
**文件那条是给"场景内容住在文件里"服务的**（MuJoCo/Gazebo形制，见下），
**装配那条是给"场景内容由调用方在进程内声明"服务的**。今天没有从文件建装配的
通道，也**不该有**——那要先给运动源与驱动器定一份字节形制，而那是一个面
（轴1规则1），要先进``engine_facets.py``登记。真实需求到来之前不预支。

## 装配容器（第二部分）学的是谁

* **spec/10第一节**（本模块第二部分的正本）：组件/端口用稳定ID声明、
  交互**按对声明**不设全局表、**延迟实例化**（组装期只登记，``finalize()``
  统一校验）、数量与拓扑全部来自声明；
* **PyElastica**：整个场景在跑第一步之前被完整校验，配错当场炸；
* **WII相邻连杆忽略**：``allowed_pairs``的既有语义（**允许重叠、不报事件**）
  由``collision.BroadPhaseCollisionQuery``定义，本模块**沿用不另造**。

## 学的是谁、学了什么（decisions/0011，文件加载器那半）

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

from physics_engine.actuators import ActuatorDeclaration
from physics_engine.canonical import (
    FTS_PROFILE,
    CanonicalProfile,
    canonical_bytes,
    canonical_sha256,
    strict_loads,
)
from physics_engine.engine_facets import ENGINE_REGISTRY
from physics_engine.motion import MotionSource
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
    """场景模块的一切失败关闭——文件加载与装配容器共用一个。

    共用而不是分成两个，是因为两条路失败时调用方要做的事是同一件：拒绝这个场景。
    ``pe-scene``已经在按这个类捕获，分裂它会让一个装配期错误从CLI的
    退出码2掉成栈回溯。
    """


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


# ==========================================================================
# 装配容器——spec/10第一节（决策0045）
# ==========================================================================
#
# 以上是文件加载器，以下是装配容器。两者共用``SceneError``与``PosedBody``，
# 除此之外没有耦合：加载器一个字节都没改。


#: 装配清单的规范化参数。**这不是一个已登记的面**（``engine_facets.py``里没有
#: 它），与``modelgen.MODELGEN_PROFILE``同一条理由：这份字节**不跨边界**——
#: 不落盘、不进场景文件、不进run package，只作"同一份装配声明是否产出同一份
#: 清单"的**进程内比较基准**（``assembly_manifest_bytes``）。
#: 哪天它要落盘或要被消费方读，必须先进面清册——那是AGENTS.md的面清册义务，
#: 本仓已经因为忘了这条违反轴1规则1近两个版本（decisions/0017）。
ASSEMBLY_MANIFEST_PROFILE = CanonicalProfile(ensure_ascii=True, file_trailing_newline=False)


@dataclass(frozen=True)
class ContactPair:
    """一对被显式声明要算接触的体。**次序即声明次序**。

    为什么留住次序：接触对清单是要交给求解器按序处理的，而集合的遍历次序
    随``PYTHONHASHSEED``变。一个"同一份声明两次跑出不同次序"的场景，
    它的接触求解结果就不可复现——那是轴3规则5要挡的事，在这里花不了什么代价。

    ``members``是判"是不是同一对"的口径：``(A, B)``与``(B, A)``是同一对。
    """

    body_a: str
    body_b: str

    @property
    def members(self) -> frozenset[str]:
        return frozenset((self.body_a, self.body_b))


@dataclass(frozen=True)
class AssembledBody:
    """装配里的一个体：几何+位姿（``PosedBody``）、可选运动源、可选驱动器。

    **位姿只有一个来源。** 一个体要么用静态位姿（``posed``里带的那个），
    要么由运动源给位姿，**不许两样都给**——本层不发明"挂载位姿∘运动源位姿"
    的复合约定。那个约定要定，得先回答"运动源给的是世界位姿还是挂载系里的
    位姿"，而spec/10第二节一个字都没写。默认挑一个的后果与``motion.py``
    那五条插值语义是同一种：两个调用方算出不同的物理，两边都以为自己是对的。
    """

    posed: PosedBody
    motion_source: MotionSource | None
    actuator: ActuatorDeclaration | None

    @property
    def body_id(self) -> str:
        return self.posed.body.body_id

    def posed_at(self, t_s: float) -> PosedBody:
        """``t_s``时刻的位姿。没有运动源的体逐字节返回它被声明时那个对象。"""

        if self.motion_source is None:
            return self.posed
        pose = self.motion_source.pose_at(t_s)
        return PosedBody(
            body=self.posed.body,
            translation_mm=pose.translation_mm,
            rotation_xyzw=pose.rotation_xyzw,
        )


@dataclass(frozen=True)
class FinalizedScene:
    """``finalize()``的产出：**冻结、已校验、可以开跑**的场景。

    ``allowed_pairs``的类型与语义**逐字与``Scene``相同**，为的是这两个字段能
    直接喂给``collision.BroadPhaseCollisionQuery(bodies, allowed_pairs=...)``——
    那是今天唯一在用它的消费方，本层不为它另造一套。
    """

    scene_id: str
    bodies: tuple[AssembledBody, ...]
    contact_pairs: tuple[ContactPair, ...]
    allowed_pairs: frozenset[frozenset[str]]

    @property
    def body_ids(self) -> tuple[str, ...]:
        return tuple(body.body_id for body in self.bodies)

    def posed_bodies_at(self, t_s: float) -> tuple[PosedBody, ...]:
        """按声明次序给出``t_s``时刻的全部位姿——``BroadPhaseCollisionQuery``的入参。"""

        return tuple(body.posed_at(t_s) for body in self.bodies)

    def assembly_manifest_bytes(self) -> bytes:
        """装配清单的规范字节——**确定性的比较基准，不是产物**。

        体与接触对按**声明次序**出，``allowed_pairs``排序后出（它是集合，
        没有声明次序可留）。见``ASSEMBLY_MANIFEST_PROFILE``：这不是一个面。
        """

        document = {
            "scene_id": self.scene_id,
            "bodies": list(self.body_ids),
            "contact_pairs": [[pair.body_a, pair.body_b] for pair in self.contact_pairs],
            "allowed_pairs": sorted(sorted(pair) for pair in self.allowed_pairs),
        }
        return canonical_bytes(document, ASSEMBLY_MANIFEST_PROFILE)


def _require_body_id(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise SceneError(f"{what} must be a nonempty string: {value!r}")
    return value


class SceneAssembly:
    """装配容器：**组装期只登记，``finalize()``统一校验**（spec/10第一节第3条）。

    五个``declare_*``一律只记账不校验交叉引用——这不是偷懒，是spec点名的形制：
    声明的次序不该决定装配成不成立（先声明接触对、后声明体，与反过来，
    必须是同一个场景）。于是**一切交叉引用在``finalize()``一次性校验**，
    而``finalize()``之后容器冻结，再声明什么都会被拒。

    **它不做什么**（写在明处，免得被当成"暂未支持"）：

    * **没有父子/运动学树**。一个盘的筒与两片法兰是三个体，各自带各自的
      运动源。要"一个刚体组同转"，那是约束或运动学树，spec/10没有它；
    * **本层不做接触物理**。``declare_contact_between``只记账——它记的是
      *要算哪些对*；球-球候选已由力学域``contact_pipeline``消费并逐帧检测，
      非球形、动态平面与摩擦历史仍由各自接触内核负责，不反塞进装配层；
    * **没有``Actuator.apply``**。驱动器只以声明进来（决策0038的边界）；
    * **没有从场景文件建装配的通道**。理由见模块文档。
    """

    def __init__(self, scene_id: str) -> None:
        if not isinstance(scene_id, str) or not scene_id.startswith("scene/"):
            raise SceneError(
                f"scene_id must be namespaced like 'scene/...': {scene_id!r}"
            )
        self._scene_id = scene_id
        self._bodies: list[PosedBody] = []
        self._motion_sources: list[tuple[str, MotionSource]] = []
        self._actuators: list[tuple[str, ActuatorDeclaration]] = []
        self._contacts: list[ContactPair] = []
        self._allowed: list[tuple[str, str]] = []
        self._finalized = False

    @property
    def scene_id(self) -> str:
        return self._scene_id

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    # -- 组装期：只登记 ---------------------------------------------------

    def _assert_open(self, what: str) -> None:
        if self._finalized:
            raise SceneError(
                f"{self._scene_id}: the scene is finalized; {what} is no longer allowed — "
                "finalize()之后场景不可变（spec/10第一节第3条：整个场景在跑第一步之前"
                "被完整校验）。改装配就重新装一个，不许在已校验的场景上追加声明"
            )

    def declare_body(self, posed: PosedBody) -> None:
        """登记一个体（几何+静态位姿）。重复的``body_id``留到``finalize()``炸。"""

        self._assert_open("declare_body")
        if not isinstance(posed, PosedBody):
            raise SceneError(f"declare_body expects a PosedBody, got {posed!r}")
        self._bodies.append(posed)

    def declare_motion_source(self, body_id: str, source: MotionSource) -> None:
        """把一个运动源绑到一个体上。体存不存在留到``finalize()``查。"""

        self._assert_open("declare_motion_source")
        name = _require_body_id(body_id, "body_id")
        if not isinstance(source, MotionSource):
            raise SceneError(
                f"{name}: motion source must implement MotionSource "
                "(pose_at/horizon_s/is_replayable are all required), "
                f"got {source!r}"
            )
        self._motion_sources.append((name, source))

    def declare_actuator(self, body_id: str, actuator: ActuatorDeclaration) -> None:
        """把一个驱动器声明绑到一个体上。"""

        self._assert_open("declare_actuator")
        name = _require_body_id(body_id, "body_id")
        if not isinstance(actuator, ActuatorDeclaration):
            raise SceneError(
                f"{name}: actuator must be an ActuatorDeclaration, got {actuator!r}"
            )
        self._actuators.append((name, actuator))

    def declare_contact_between(self, body_a: str, body_b: str) -> None:
        """**按对声明**要算接触的两个体（spec/10第一节第2条：不设全局表）。

        为什么不是"全体两两自动检测"：那样接触对数随体数平方增长，而且
        **没有人声明过那些对的意图**——一个自动生成的接触对既说不出它为什么
        存在，也说不出它什么时候该消失。本仓在``allowed_pairs``上已经采过
        按对声明的形制（WII相邻连杆忽略），这里是同一条。
        """

        self._assert_open("declare_contact_between")
        self._contacts.append(
            ContactPair(
                body_a=_require_body_id(body_a, "body_a"),
                body_b=_require_body_id(body_b, "body_b"),
            )
        )

    def declare_allowed_pair(self, body_a: str, body_b: str) -> None:
        """声明一对**允许重叠、不报事件**的体——``allowed_pairs``的既有语义。

        语义是``collision.BroadPhaseCollisionQuery``定的（其``check_state``对
        白名单里的对直接``continue``），本层沿用，不另造一套。典型用法就是
        本仓一直在用的那个：同一个装配里按构造就贴在一起的件
        （盘的筒与法兰、相邻连杆），它们的"重叠"是建模产物不是物理事件。
        """

        self._assert_open("declare_allowed_pair")
        self._allowed.append(
            (
                _require_body_id(body_a, "body_a"),
                _require_body_id(body_b, "body_b"),
            )
        )

    # -- 统一校验 ---------------------------------------------------------

    def finalize(self) -> FinalizedScene:
        """一次性校验整份装配，产出冻结的``FinalizedScene``，并把容器封死。

        **失败关闭，且炸在这里**——不推迟到查询期。本仓修过一次同类缺陷：
        重复``body_id``与未知``allowed_pairs``成员此前推迟到
        ``BroadPhaseCollisionQuery``构造期才炸，后果是``pe-scene validate``
        对非法场景报valid并以0退出（CHANGELOG T0）。

        ``finalize()``可以重复调用，每次产出**等价且清单字节相同**的场景——
        它是纯的读取，封死的是``declare_*``那一侧。
        """

        if not self._bodies:
            raise SceneError(f"{self._scene_id}: a scene needs at least one body")

        identifiers = [posed.body.body_id for posed in self._bodies]
        duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
        if duplicates:
            raise SceneError(
                f"{self._scene_id}: duplicate body_id in assembly: {duplicates} — "
                "体的身份是接触对、运动源、驱动器三样东西的指向目标，"
                "重名意味着那三样指向谁都说不清"
            )
        known = set(identifiers)

        motion_of: dict[str, MotionSource] = {}
        for name, source in self._motion_sources:
            if name not in known:
                raise SceneError(
                    f"{self._scene_id}: motion source bound to unknown body {name!r}; "
                    f"declared bodies are {sorted(known)}"
                )
            if name in motion_of:
                raise SceneError(
                    f"{self._scene_id}: body {name!r} already has a motion source — "
                    "两个运动源就是两个不同的位姿，本层不替你挑一个"
                )
            motion_of[name] = source

        actuator_of: dict[str, ActuatorDeclaration] = {}
        for name, actuator in self._actuators:
            if name not in known:
                raise SceneError(
                    f"{self._scene_id}: actuator {actuator.actuator_id!r} bound to "
                    f"unknown body {name!r}; declared bodies are {sorted(known)}"
                )
            if name in actuator_of:
                raise SceneError(
                    f"{self._scene_id}: body {name!r} already has actuator "
                    f"{actuator_of[name].actuator_id!r}"
                )
            actuator_of[name] = actuator

        contact_pairs = self._finalize_pairs(known)
        allowed = self._finalize_allowed(known, contact_pairs)

        bodies = tuple(
            AssembledBody(
                posed=posed,
                motion_source=motion_of.get(posed.body.body_id),
                actuator=actuator_of.get(posed.body.body_id),
            )
            for posed in self._bodies
        )
        for body in bodies:
            _assert_single_pose_source(self._scene_id, body)

        self._finalized = True
        return FinalizedScene(
            scene_id=self._scene_id,
            bodies=bodies,
            contact_pairs=contact_pairs,
            allowed_pairs=allowed,
        )

    def _finalize_pairs(self, known: set[str]) -> tuple[ContactPair, ...]:
        seen: set[frozenset[str]] = set()
        for pair in self._contacts:
            if pair.body_a == pair.body_b:
                raise SceneError(
                    f"{self._scene_id}: contact declared between {pair.body_a!r} and "
                    "itself — 自接触在今天的引擎里没有表示（plans/04场景⑤登记的墙），"
                    "一个体与自己的对不是自接触，是笔误"
                )
            unknown = sorted(pair.members - known)
            if unknown:
                raise SceneError(
                    f"{self._scene_id}: contact pair "
                    f"({pair.body_a!r}, {pair.body_b!r}) references unknown bodies: "
                    f"{unknown}; declared bodies are {sorted(known)}"
                )
            if pair.members in seen:
                raise SceneError(
                    f"{self._scene_id}: contact between {pair.body_a!r} and "
                    f"{pair.body_b!r} is declared twice — 一对接触声明两次，"
                    "要么是同一件事说了两遍（多余），要么是有人以为自己在声明别的对"
                    "（错）。本层不去猜是哪一种，也不去重"
                )
            seen.add(pair.members)
        return tuple(self._contacts)

    def _finalize_allowed(
        self, known: set[str], contact_pairs: tuple[ContactPair, ...]
    ) -> frozenset[frozenset[str]]:
        declared_contacts = {pair.members for pair in contact_pairs}
        allowed: set[frozenset[str]] = set()
        for body_a, body_b in self._allowed:
            members = frozenset((body_a, body_b))
            if len(members) != 2:
                raise SceneError(
                    f"{self._scene_id}: allowed pair must name two distinct bodies: "
                    f"{body_a!r}"
                )
            unknown = sorted(members - known)
            if unknown:
                raise SceneError(
                    f"{self._scene_id}: allowed pair references unknown bodies: "
                    f"{unknown}; declared bodies are {sorted(known)}"
                )
            if members in declared_contacts:
                raise SceneError(
                    f"{self._scene_id}: pair ({body_a!r}, {body_b!r}) is declared both "
                    "as a contact pair and as an allowed pair — 这两条声明是矛盾的："
                    "allowed_pairs的语义是'允许重叠、不报事件'"
                    "（collision.BroadPhaseCollisionQuery.check_state直接continue），"
                    "而declare_contact_between的意思正是'这一对的接触要算'。"
                    "留着它，接触对清单与碰撞查询会对同一对给出相反的处置"
                )
            allowed.add(members)
        return frozenset(allowed)


def _assert_single_pose_source(scene_id: str, body: AssembledBody) -> None:
    """一个体的位姿只能有一个来源——见``AssembledBody``的类文档。"""

    if body.motion_source is None:
        return
    posed = body.posed
    # 用``tuple()``取一遍：``PosedBody``不强制平移是元组，而``[0.0, 0.0, 0.0]``
    # 与``(0.0, 0.0, 0.0)``在``!=``下是不等的——那会让一条合法装配假红。
    if tuple(posed.translation_mm) != (0.0, 0.0, 0.0) or tuple(
        posed.rotation_xyzw
    ) != (0.0, 0.0, 0.0, 1.0):
        raise SceneError(
            f"{scene_id}: body {body.body_id!r} carries both a nonidentity static pose "
            f"(translation {posed.translation_mm}, rotation {posed.rotation_xyzw}) and a "
            "motion source — 位姿只能有一个来源。本层不发明"
            "'挂载位姿∘运动源位姿'的复合约定（spec/10第二节没有定义运动源给的是"
            "世界位姿还是挂载系位姿）。把挂载位姿写进运动源，或去掉运动源"
        )


__all__ = [
    "ASSEMBLY_MANIFEST_PROFILE",
    "SCENE_CANONICAL_PROFILE",
    "SCENE_FACET",
    "SCENE_FACET_VERSION",
    "SHAPE_KINDS",
    "AssembledBody",
    "ContactPair",
    "FinalizedScene",
    "Scene",
    "SceneAssembly",
    "SceneError",
    "load_scene",
    "register_shape_kind",
]
