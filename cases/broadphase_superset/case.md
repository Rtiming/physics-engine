# case/broadphase_superset：broad ⊇ narrow 超集不变量

守的是碰撞查询对外**唯一的硬承诺**：broad phase不许漏报。

    separation_mm < 0  ⟹  两个世界AABB相交

反例数必须严格为0。换SAP/BVH/网格哈希那天，这是唯一不能破的门——
其余的都可以协商，这条不能。

- 清单：`oracle.json`（1条oracle，`load_tier=interactive`）
- 语料：`samples.json`（120对、3600个float64，双哈希钉在清单`arrays.samples`）
- 生成器：`generate_oracle.py`（SHA钉在清单`generator.sha256`）
- conformance：`tests/cases/test_broadphase_superset.py`

## 一、物理/几何设定

单位一律mm。120对随机球/胶囊，种子`20260805`，RNG=`python:random.Random`
（CPython的MT19937，同种子跨平台跨版本同序列）。每个体15个数、每对30个数、
共3600个数，逐个进`samples.json`并被双哈希覆盖。

| 参数 | 取值 |
|---|---|
| 形状族 | 球或胶囊各50%（`kind`字段0/1） |
| 半径 | 均匀[2, 12] mm |
| 胶囊端点（局部系） | 每轴均匀[−15, 15] mm |
| 体A平移 | 每轴均匀[−6, 6] mm |
| 体B平移 | 每轴均匀[−22, 22] mm |
| 姿态 | 四维高斯归一化的均匀随机单位四元数 |
| 判定边界排斥 | `|separation| ≤ 1e-6 mm` 或 任一轴AABB间隙`≤ 1e-6 mm` 即重采样 |

实测采样统计（冻结在清单`expected`里）：120对中侵入26对、AABB相交93对、
**broad假阳性67对**、反例0对；边界排斥实际触发0次。

`broad_only_pairs=67 > 0`是刻意的判据：如果语料里一个假阳性都没有，
超集命题就退化成平凡真，门看着绿其实什么都没守。

## 二、参考解出处

**可证命题**，不是拟合数也不是实测数：世界AABB按定义是形状的保守外包
（`world_aabb_mm`枚举局部盒八角点），两个形状若有交点，该交点同时落在
两个世界AABB内，故两盒必相交。逆否即得本判据。research/05 §2.1把它列为
"换SAP/BVH那天唯一不能破的承诺"。

分类计数由`generate_oracle.py`独立算出（轴7规则4）：Arvo中心-半边长包盒 +
凸一维黄金分割距离，与`cases/rotated_aabb/`、`cases/segment_distance/`
同源的两份独立实现，都不import`shapes.py`/`collision.py`。

**计数为什么敢钉死**：采样期已拒绝一切离判定边界1e-6mm以内的配置，
而两条实现路径的浮点差实测在1e-14mm量级——差八个量级，没有任何一对
会因为浮点差异翻面。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---|---|---|
| `superset_counterexamples` | 0 | 0 | 可证命题，反例只能是0。任何非零都意味着broad漏报，而漏报是碰撞查询对外唯一的硬承诺。零容差不是"选了个很严的数"，是这里没有连续量。 |
| `pair_count` | 0 | 0 | 语料规模是确定性整数；少一对说明语料被截断。 |
| `negative_separation_pairs` | 0 | 0 | 确定性整数。边界排斥保证浮点差不会让任何一对翻面，故可零容差。 |
| `aabb_overlapping_pairs` | 0 | 0 | 同上。它同时是broad phase工作量的确定性代理。 |
| `broad_only_pairs` | 0 | 0 | 必须>0才证明语料里真有假阳性；否则超集判据平凡成立。零容差同上。 |
| `narrow_phase_events` | 0 | 0 | 查询报出的事件条数必须**恰好**等于`separation<0`的对数——多一条是假阳性没吃干净，少一条是漏报。 |

语料完整性另有两条零容差判据（`arrays.samples`）：raw级SHA-256盖文件字节、
语义级SHA-256盖3600个float64的小端C序字节流。前者抓"文件被动过"，
后者抓"换了存储形式但值该不变"。

## 四、已知失效清单

- **只对球/胶囊族成立**。不是因为AABB对别的族不保守（它对所有族都保守），
  而是因为别的族本仓**算不出**`separation_mm`——narrow phase第一片只覆盖
  球/胶囊。含`finite_cylinder`/`rounded_box`/`mesh`的对只有broad结论，
  进不了这条判据。解封条件：narrow phase第二片落地。
- **不覆盖临界接触**：采样期主动排斥了离判定边界1e-6mm以内的配置。
  正切接触（`separation`恰为0）的行为归B档。
- **不覆盖多体场景**：语料是两两独立的对，不验`allowed_pairs`白名单、
  不验O(n²)遍历次序、不验重复body_id。那些在
  `tests/test_shapes_and_collision.py`与`tests/test_scene_validation_gaps.py`。
- **`superset_counterexamples`单独看是粗筛**：2026-08-05实测（把世界包盒的
  半边长按比例缩小来模拟"更紧的AABB"这类优化），缩到0.95时反例数仍为0、
  缩到0.60仍为0，直到**0.40**才出现13个反例。深度侵入的对包盒重叠余量很大，
  轻度缩窄不会立刻产生漏报。真正在0.95就把门顶红的是
  `aabb_overlapping_pairs`（93→88）——**冻结的计数比不变量本身灵敏**。
  所以这条案例的六个量要一起看，不能只盯反例数。
- **不是随机化测试**：种子固定写进清单，每次跑的是同一批120对。
  换种子=换金标=走轴7规则5（决策记录+重生成）。这是刻意的：
  随机种子每次不同的门会偶发红，而偶发红的门半年后一定会被人加`retry`。
- **无静默skip**：本案例的conformance测试没有任何`skip`/`xfail`。

## 五、档位与负载级

- 判据强度：**A档**（可证命题，强于拟合数）；
- 负载级：**interactive**（交互级，无pytest marker）；
- 实测：`pytest tests/cases/test_broadphase_superset.py -q` = 2 passed，0.04秒
  （含120对的AABB+narrow+查询三遍计算；墙钟仅供落级参考，**不进门**）。

## 六、本案例不是什么

- **不是"世界AABB是SE(3)不变量"的判据**。世界AABB**不是**SE(3)不变量——
  把整个场景刚体旋转一下，每个体的世界AABB都会变，broad phase的候选集合
  因此也会变。"整场景旋转后事件集合相同"是**错的**判据：含
  `finite_cylinder`/`rounded_box`/`mesh`的对必然红，含球/胶囊的对也会在
  AABB擦边处翻面。本案例不写这种判据，将来也不要写。
- **不是紧致性判据**。超集成立不代表包盒紧；假阳性67/93说明它相当松。
  "松得离谱"是性能问题（broad phase工作量），归确定性计数器基线
  （plans/02第一批案例7），不归这里。
- **不是narrow phase数值判据**。`separation_mm`的**值**对不对归
  `case/segment_distance`；这里只用它的符号。
- **不是对所有实现的证明**。它是对**当前实现在这120对上**的检查。
  可证命题的证明在第二节的文字里，门只能抽样——所以换broad phase实现时
  除了跑绿这条门，还要重读那段证明是否仍然成立。
