# case/segment_distance：球/胶囊解析距离与线段退化分支

`segment_segment_distance_mm`是全仓唯一在算实数值的内核。**先给唯一的内核
建judge，再谈加内核**——这条案例就是那个judge。原先判据写死在
`tests/test_narrow_phase.py`里，按轴7规则3搬进`oracle.json`，同批补上
五条退化分支与一条无退化的一般路径。

- 清单：`oracle.json`（11条oracle，`load_tier=interactive`）
- 生成器：`generate_oracle.py`（SHA钉在清单`generator.sha256`）
- conformance：`tests/cases/test_segment_distance.py`

## 一、物理/几何设定

单位一律mm。两组输入。

**A组：线段对（7条，直接喂`segment_segment_distance_mm`）**

| oracle | p1 | q1 | p2 | q2 | 走到的内核分支 |
|---|---|---|---|---|---|
| `branch_point_point` | (0,0,0) | (0,0,0) | (3,4,0) | (3,4,0) | `a≤1e-12 且 e≤1e-12` |
| `branch_segment1_degenerate` | (0,12,5) | (0,12,5) | (−10,0,0) | (10,0,0) | `a≤1e-12`（段1退化） |
| `branch_segment2_degenerate` | (−10,0,0) | (10,0,0) | (0,0,7) | (0,0,7) | `e≤1e-12`（段2退化） |
| `branch_parallel` | (0,0,0) | (10,0,0) | (0,3,0) | (10,3,0) | `denominator≤1e-12` |
| `branch_t_below_zero` | (0,0,0) | (10,0,0) | (5,3,4) | (5,11,4) | `t<0`钳制 |
| `branch_t_above_one` | (0,0,0) | (10,0,0) | (5,−16,6) | (5,−8,6) | `t>1`钳制 |
| `interior_crossing` | (−5,0,1) | (5,0,1) | (0,−5,−1) | (0,5,−1) | 无退化 |

**B组：体对（4条，喂`BroadPhaseCollisionQuery`）**

| oracle | 体A | 体B | 位姿 |
|---|---|---|---|
| `two_spheres_overlap` | 球r=10 | 球r=10 | B平移(15,0,0) |
| `sphere_capsule` | 胶囊(0,0,0)→(20,0,0)、r=3 | 球r=4 | B平移(10,5,0) |
| `rotated_capsule` | 胶囊(0,0,0)→(20,0,0)、r=2，绕z转90°（xyzw=(0,0,√½,√½)） | 球r=1 | B平移(0,10,2.5) |
| `broad_hit_narrow_clear` | 球r=5 | 球r=5 | B平移(8,8,0) |

## 二、参考解出处

**闭式解，手推**。算法出处：Ericson, *Real-Time Collision Detection*, 2005,
§5.1.9 `ClosestPtSegmentSegment`（被验内核实现的就是它）；判据侧不抄它的实现。

球/胶囊族的距离与穿透（research/05 §2.1）：

    d = segdist − (r1 + r2)          penetration_mm = −d

逐条推导（值全部是可精确表示的整数，不是"跑一遍记下来"）：

1. `branch_point_point`：两段皆退化，`|r| = |(−3,−4,0)| = 5`；
2. `branch_segment1_degenerate`：点(0,12,5)到x轴段，`f=200`、`e=400`→`t=0.5`→
   最近点(0,0,0)，`√(12²+5²)=13`；
3. `branch_segment2_degenerate`：`c=−200`、`a=400`→`s=0.5`→最近点(0,0,0)，
   到(0,0,7)距离7；
4. `branch_parallel`：`a=e=b=100`→`denominator=0`→`s=0`；`f=0`→`t=0`，间隙3；
5. `branch_t_below_zero`：`b=0`、`f=−24`、`e=64`→`t=−0.375<0`→钳到0，
   `s=clamp(−c/a)=0.5`，最近点对(5,0,0)与(5,3,4)，`√(3²+4²)=5`；
6. `branch_t_above_one`：`f=128`、`e=64`→`t=2>1`→钳到1，`s=0.5`，
   最近点对(5,0,0)与(5,−8,6)，`√(8²+6²)=10`；
7. `interior_crossing`：`s=t=0.5`，两段z向相距2；
8. B组四条：球心距15−半径和20→侵入5；球心到胶囊轴距5−半径和7→侵入2；
   转90°后胶囊轴沿y，球心到轴距2.5−半径和3→侵入0.5；
   第四条球心距`8√2≈11.3137`>半径和10→**不报事件**（broad假阳性被吃掉）。

**独立交叉验证**：`generate_oracle.py`里的`_independent_distance`把二维最小化
化成"点到线段距离的一维凸最小化"，用黄金分割搜索求解——与Ericson的
钳制-重算不是同一条路径。它不产生金标，只在写盘前对每条手推值做1e-9的
交叉验证，防的是手算笔误（轴7规则4的独立性在这里是"防笔误"而不是"当金标"，
因为搜索法的精度不足以当1e-12判据的源）。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---|---|---|
| `distance_mm` | 0 | 1e-12 mm | 闭式解与内核都只做加减乘与一次开方，量级10mm下双精度舍入~1e-15mm，留三个量级余量。**不写==0**：逐位相等会把求和次序也冻结成契约，那是实现细节不是物理。rel=0因为判据是绝对几何间隙，不随量级缩放。 |
| `penetration_mm` | 0 | 1e-12 mm | 同上。`rotated_capsule`那条经过四元数→矩阵，世界端点误差实测~4.4e-15mm，仍在余量内。 |
| `segment_distance_mm` | 0 | 1e-12 mm | 同上。 |
| `event_count` | 0 | 0 | 事件条数是确定性整数，零容差；多报是假阳性没吃干净，漏报是破了碰撞查询对外唯一的硬承诺。 |
| `confidence` | 0 | 0 | 可信度是枚举字符串，逐位相等。降级冒充`narrow_phase`是诚实性缺陷（AGENTS.md本仓纪律第四条）。 |

## 四、已知失效清单

- **不覆盖临界接触**（`separation`恰为0）：钳制分支在`t`恰好等于0或1时走哪一条
  由浮点比较决定，本案例的输入都离边界至少3mm。要覆盖需要专门的
  病态输入族，属B档。解封条件：narrow phase第二片进来时一并做。
- **不覆盖极端量级**：所有输入在1—50mm量级。1e-6mm与1e6mm量级下
  `1e-12`的绝对容差与`1e-12`的退化阈值都会失去意义（阈值是绝对量，
  对缩放不变性有破坏）。这条是内核的已知性质，不是案例的疏漏——
  写在这里是为了让人别误以为它被验过。
- **无静默skip**：本案例的conformance测试没有任何`skip`/`xfail`。
- 分支覆盖是**一次性实测**（见第五节），不是持续断言：若日后有人重构
  `segment_segment_distance_mm`把分支合并，本案例的11条值仍会全绿而分支
  覆盖声明失效。持续断言需要行/字节码级插桩进门，代价与收益不成比例，
  故留在这里明说而不假装。

## 五、档位与负载级

- 判据强度：**A档**（解析闭式解，research/05第一节最强一档）；
- 负载级：**interactive**（交互级，无pytest marker）；
- 实测：`pytest tests/cases/test_segment_distance.py -q` = 12 passed，0.03秒
  （墙钟仅供落级参考，**不进门**——性能走`tests/perf/`的确定性量与`tools/bench.py`）。

**分支覆盖的实测方式**（2026-08-05本机）：用`sys.settrace`开
`f_trace_opcodes`跟踪`collision.py`的执行，行级+字节码偏移两级判定——
第60行`s = _clamp(…) if denominator > 1e-12 else 0.0`两侧同在一行，
行级分不开，故按`dis`给出的偏移区分（`CALL`在偏移376=非平行侧，
`LOAD_CONST 0.0`在偏移386=平行侧）。七条输入各自命中且仅命中声明的那一条。
未改动`collision.py`一个字节。

## 六、本案例不是什么

- **不是broad phase的判据**。它只验narrow phase的数值与事件分类；
  "AABB是不是保守"归`case/broadphase_superset`。
- **不是圆柱/盒/网格族的判据**。那些族本仓没有narrow phase实现，
  `penetration_mm`按设计是`None`——不要因为本案例全绿就以为所有形状对
  都有精确穿透值。
- **不是接触力学**。`penetration_mm`是几何侵入深度，不是接触力、不是
  接触点、不是法向。把它当罚接触的输入需要另一层判据（C档，随WDS内核搬迁）。
- **不是性能门**。11条oracle的耗时不裁决任何性能结论；性能走
  `tests/perf/`的确定性量与`tools/bench.py`的报告。
