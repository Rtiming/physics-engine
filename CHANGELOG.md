# 变更日志

形制按Keep a Changelog；版本语义见decisions/0010与0012（0.x期：
minor可破坏兼容，patch只含兼容修复，**不回port**——修复只进新版本）。
发版脚本强制：版本必须先在本页有条目才准发。

## Unreleased

并行开发期的分槽（plans/02第三节规避规则3）：每条轨道只在自己的小节内append，
合并冲突面为零。发版时由一次收口提交合并成版本条目。

### T0 收口（缺陷清零+面清册+文档对账）

- 面清册一次登记齐四个新面（`physics_scene_collision_events`、
  `engine_perf_baseline`、`engine_oracle_manifest`、`engine_behavior_baseline`），
  其中`physics_scene_collision_events`补登记了自0.3.0起就在落盘却未登记的形制
  （本仓自己违反轴1规则1近两个版本）
- `pyproject.toml`登记与负载三级对应的pytest marker（`batch`/`serverclass`）
- **失败关闭补齐（破坏性：此前被放行的非法场景现在会被拒）**：
  `MeshAsset`的`units`/`usage`/`convexity`与`CollisionShape.direction`
  运行时校验取值；场景文件缺`direction`不再放行；重复`body_id`与
  未知`allowed_pairs`成员在**加载期**失败关闭（此前推迟到查询构造期才炸）
- `pe-scene`退出码修复：查询构造异常纳入捕获，非法场景两个子命令一律退出码2
  （此前`check-collisions`抛栈回溯并以1退出，而1的语义是"有候选"）
- `collision_events.json`补`facet_version`字段（**破坏字节**，实验档minor内声明）
- `examples/collision_preview_cell.scene.json`的网格AABB改为资产真值
  （此前声明的包络在z轴上与资产完全不相交，`direction: "envelope"`不成立）

### T1 性能门

### T2 几何

### T3 oracle与案例

### T4 力学地基

### T6 资产（materials）

## 0.4.0 — 2026-08-05

新增：

- **narrow phase第一片（球/胶囊族）**：线段-线段闭式最近距离；同族对给
  `confidence="narrow_phase"`与精确`penetration_mm`；**broad命中但narrow
  判分离的对不再报事件**（假阳性消除）。圆柱/盒/网格对诚实保留
  `broad_phase`（下一片再做，不冒充）
- `PosedBody.transform_point_mm`/`rotate_local_mm`位姿变换
- `segment_segment_distance_mm`进公开面

## 0.3.0 — 2026-08-04

新增：

- `physics_engine.scene`：场景文件格式（`physics_scene`面1.0.0，draft）——
  严格加载、未知键失败关闭、`extensions`声明式扩展加载（Gazebo形制）
- `pe-scene`命令行入口：`validate`、`check-collisions`（linter式退出码；
  `--out-dir`走run package原子发布+同边界复读）
- `SHAPE_KINDS`注册表与`register_shape_kind`（重复登记拒收）
- 示例场景`examples/collision_preview_cell.scene.json`（真实参数）

## 0.2.0 — 2026-08-04

首个发布版。新增：

- `facets`：面清册+失败关闭读取端（轴1参考实现）
- `canonical`：声明式规范化JSON读写对（轴3规则2）
- `identity`：四段身份、命名空间ID、单位后缀校验（轴2）
- `provenance`：耐久独占写、no-replace目录改名、四点签名保护读、
  目录快照与机械严格复读（轴5机械半边）
- `run_package`：装配（manifest最后写）、语义闭包复读、生命周期法则（轴4/5）
- `shapes`+`collision`（实验档）：形状声明层与broad-phase碰撞查询
- `tools/accept.py`自吃药验收器（30/120双档、三轴正交）与
  `engine_acceptance_receipt`面

## 0.1.0 — 2026-08-04

骨架占位（未发wheel）：包结构、章程、七条规范轴、操作单。
