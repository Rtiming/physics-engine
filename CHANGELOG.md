# 变更日志

形制按Keep a Changelog；版本语义见decisions/0010与0012（0.x期：
minor可破坏兼容，patch只含兼容修复，**不回port**——修复只进新版本）。
发版脚本强制：版本必须先在本页有条目才准发。

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
