"""本仓自己的面清册——引擎吃自己的药。

本仓每新增一个跨边界字节形制，先来这里登记再落盘——和我们要求消费方做的
一模一样。清册在此**一次登记齐**（plans/02的T0a闸门）：并行开发期间各轨道
只`from physics_engine.engine_facets import ...`，不再改本文件；要加第五个面的
轨道走一次闸门提交。

登记时机的教训（decisions/0017）：`cli.py`从0.3.0起就在落盘
`physics_scene_collision_events`这个形制，却一直没有登记——本仓自己违反了
轴1规则1近两个版本。面清册的价值不在于它记了什么，在于**落盘前必须先来登记**
这条纪律本身。
"""

from __future__ import annotations

from physics_engine.facets import Facet, FacetRegistry, FacetStatus

#: 验收回执面：内部消费（工具与governance测试），不作对外兼容承诺。
ACCEPTANCE_RECEIPT_FACET = "engine_acceptance_receipt"
ACCEPTANCE_RECEIPT_VERSION = "0.1"

#: 场景文件面：数据层入口的格式（scene.py）。出生draft——第一个消费方
#: 拿它过完自己的门禁前不作兼容承诺（升档须决策记录）。
PHYSICS_SCENE_FACET = "physics_scene"
PHYSICS_SCENE_VERSION = "1.0.0"

#: 碰撞事件产物面：`pe-scene check-collisions --out-dir`落盘的判定结果
#: （cli.py）。出生draft，与场景面同理。
COLLISION_EVENTS_FACET = "physics_scene_collision_events"
COLLISION_EVENTS_VERSION = "0.1"

#: 性能基线面：spec/13第零节的延迟预算与wheel预算的落盘形制（plans/02 T1）。
#: 内部消费——它是**开发计时裁决**的输入，永不进功能路径（0014争用韧性条款）。
PERF_BASELINE_FACET = "engine_perf_baseline"
PERF_BASELINE_VERSION = "0.1"

#: oracle清单面：轴7规则2要求的形制（生成器身份+逐条expected+逐条tolerances
#: +数组双哈希+清单自指哈希）。出生draft——两个消费方（WDS的B1—B6/C1/X2/
#: free_rod、FTS的m0清单）评审后才谈升档。
ORACLE_MANIFEST_FACET = "engine_oracle_manifest"
ORACLE_MANIFEST_VERSION = "0.1"

#: 行为基线面：确定性整数计数器的金标（plans/02第一批案例7）。
#: **它不是轴7 oracle**——spec/08规则1明写实测数不作金标；这一面记的是
#: "本实现当前的确定性行为"，用途是回退网，不是物理真值。分开命名以免稀释轴7。
BEHAVIOR_BASELINE_FACET = "engine_behavior_baseline"
BEHAVIOR_BASELINE_VERSION = "0.1"

#: 材料记录面：一份记录聚合力学/光学/外观字段，各域各取所需（spec/01第44行的
#: "形制即规范种子"）。出生draft——形制取自WDS材料记录（SHA锁定+单位后缀+
#: 适用域声明三件套），两个域各自用过之前不作兼容承诺。
MATERIAL_RECORD_FACET = "engine_material_record"
MATERIAL_RECORD_VERSION = "0.1"

#: 运行轨迹面：一次run的时间序列落盘形制（`tools/view/`的两个工具之间那份JSON）。
#: 出生draft。
#:
#: **它2026-08-18之前一直没有登记，而它确实跨边界**——`trace_from_closed_loop.py`
#: 写、`replay.py`读，两个工具住在**互不认识对方依赖的两个环境**里（0076），
#: 中间只有这份字节。当时不登记的理由写在`tools/view/README.md`第五节：
#: 那一轨的卖点是`src/`零字节改动，而本文件在`src/`里。
#: **那是一条轨道范围内的约束，不是一条原则**，而0084已按所有者裁决把源码上限抬到4 MiB，
#: 约束不复存在。0076第五节把这件事登记成"由后续批次裁"，本批就是那个批次：**裁升**。
#:
#: 读的那一侧`replay.py`**仍然不import本模块**——它住在rerun那个venv里，
#: 让它认识`physics_engine`会破掉0076那条"两个环境互不认识对方的依赖"。
#: 把它与本清册钉在一起的是一道门（`tests/governance/test_run_trace_facet.py`），
#: 不是一条import。**门在这里正是因为不能靠import。**
ENGINE_RUN_TRACE_FACET = "engine_run_trace"
ENGINE_RUN_TRACE_VERSION = "0.1"

#: 张力测量样点面：从两侧带材张力/方向一路保留到测力轮合力、敏感轴、
#: tare、支承、电气量化与显示张力的分层结果（决策0097、plans/19 P0）。
#: 出生draft——真实LTS标定与WDS消费均未完成，不作兼容承诺。
TENSION_MEASUREMENT_SAMPLE_FACET = "tension_measurement_sample"
TENSION_MEASUREMENT_SAMPLE_VERSION = "0.1"

ENGINE_REGISTRY = FacetRegistry(
    Facet(
        name=ACCEPTANCE_RECEIPT_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.INTERNAL,
    ),
    Facet(
        name=PHYSICS_SCENE_FACET,
        major=1,
        max_tested_minor=0,
        status=FacetStatus.DRAFT,
    ),
    Facet(
        name=COLLISION_EVENTS_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.DRAFT,
    ),
    Facet(
        name=PERF_BASELINE_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.INTERNAL,
    ),
    Facet(
        name=ORACLE_MANIFEST_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.DRAFT,
    ),
    Facet(
        name=BEHAVIOR_BASELINE_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.INTERNAL,
    ),
    Facet(
        name=MATERIAL_RECORD_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.DRAFT,
    ),
    Facet(
        name=ENGINE_RUN_TRACE_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.DRAFT,
    ),
    Facet(
        name=TENSION_MEASUREMENT_SAMPLE_FACET,
        major=0,
        max_tested_minor=1,
        status=FacetStatus.DRAFT,
    ),
)

__all__ = [
    "ACCEPTANCE_RECEIPT_FACET",
    "ACCEPTANCE_RECEIPT_VERSION",
    "BEHAVIOR_BASELINE_FACET",
    "BEHAVIOR_BASELINE_VERSION",
    "COLLISION_EVENTS_FACET",
    "COLLISION_EVENTS_VERSION",
    "ENGINE_REGISTRY",
    "ENGINE_RUN_TRACE_FACET",
    "ENGINE_RUN_TRACE_VERSION",
    "MATERIAL_RECORD_FACET",
    "MATERIAL_RECORD_VERSION",
    "ORACLE_MANIFEST_FACET",
    "ORACLE_MANIFEST_VERSION",
    "PERF_BASELINE_FACET",
    "PERF_BASELINE_VERSION",
    "PHYSICS_SCENE_FACET",
    "PHYSICS_SCENE_VERSION",
    "TENSION_MEASUREMENT_SAMPLE_FACET",
    "TENSION_MEASUREMENT_SAMPLE_VERSION",
]
