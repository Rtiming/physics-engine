"""本仓自己的面清册——引擎吃自己的药。

physics-engine目前只有一个序列化面：验收器的回执。它在这里登记，
验收器写回执时盖它的名与版本，governance测试用同一清册验"回执声明的
版本能过失败关闭的读取端"。以后本仓每新增一个跨边界字节形制，先来这里
登记再落盘——和我们要求消费方做的一模一样。
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
)

__all__ = [
    "ACCEPTANCE_RECEIPT_FACET",
    "ACCEPTANCE_RECEIPT_VERSION",
    "ENGINE_REGISTRY",
]
