# peer_fcl_distance 同行库对拍：球/胶囊解析距离对FCL

承接decisions/0015第二条（"把别人的库下载下来进行对比实验"）与plans/02第四节第一批案例2。
这是0015第二条的第一次真正兑现：**同行库真的装上了，对比数字真的跑出来了**。
身份边界与许可证结论见[decisions/0025](../../docs/decisions/0025_同行库对比实验的身份与合规_20260805.md)
与[validation/README.md](../../validation/README.md)。

判据值的正本是同目录的`criteria.json`（本页只解释，不重复声明；两处冲突以`criteria.json`为准）。

## 一、物理/几何设定

纯几何、无时间演化。三类形状对 × 三个构型带，每格300组，共**2700组**输入：

| 项 | 取值 |
|---|---|
| 形状对 | 球-球、球-胶囊、胶囊-胶囊（narrow phase今天唯一覆盖的一族） |
| 半径 | 对数均匀0.5—50mm |
| 胶囊中轴半长 | 对数均匀1—200mm（中轴线段沿本体z轴，关于本体原点对称） |
| 位姿 | 平移各轴均匀−500—500mm；姿态取S^3均匀单位四元数（Shoemake法） |
| 构型带`separated` | 目标分离量对数均匀1e-2—3e2 mm |
| 构型带`grazing` | 目标分离量对数均匀1e-7—1e-2 mm（**两侧接触判据唯一可能分歧的地方**） |
| 构型带`penetrating` | 目标分离量−1e-3 mm 到 −0.9×min(r₁,r₂)，对数均匀 |
| 种子 | `random.Random(20260805)`，**stdlib**——重生成这批输入不需要任何第三方包 |

摆位方式：沿随机单位方向平移B体，用二分逼近目标分离量。
**摆位用了本仓的实现，但摆位不是被判的数**——摆完之后FCL独立地对同一批坐标算它那一列。
即便本仓实现有错，摆位也只是偏离目标带，对拍本身照样成立。

被验对象：`physics_engine.collision.segment_segment_distance_mm`（减两半径后即分离量）
与`physics_engine.collision.BroadPhaseCollisionQuery`（报不报事件）。

## 二、参考解出处

| 项 | 值 |
|---|---|
| 同行库 | python-fcl（Flexible Collision Library的Python绑定） |
| 版本 | 0.7.0.11（`criteria.json`钉死；测试会核对，漂了就红） |
| 安装方式 | `pip install python-fcl`，PyPI预编译wheel，**未本地编译** |
| wheel | `python_fcl-0.7.0.11-cp313-cp313-macosx_11_0_arm64.whl` |
| wheel SHA-256 | `fb10b3037d82ef754ba236a94a7f12da1864e13d4bc2f0d1ae9078c9a814915f` |
| 随轮二进制 | `libfcl.0.7.0` / `libccd.2.0` / `liboctomap.1.9.8` / `liboctomath.1.9.8`，逐个SHA-256进manifest |
| 上游 | https://github.com/berkeleyautomation/python-fcl |
| 许可证 | BSD-3-Clause（详见`validation/LICENSES.md`） |
| 平台 | macOS 26.5.2 / arm64 / CPython 3.13.12 / numpy 2.5.1 / BLAS=Accelerate |

**第三个证人**：本页第四节里三条"FCL错了"的结论，都另用`fractions.Fraction`精确算过真值
（段-段最小距离按"内部驻点 + 四条边界"的枚举式精确求解，与本仓的Ericson分支写法是两种写法，
只有最后开方落到float）。**同行的数不是真值，是另一个证人**——这一节就是这句话的执行。

## 三、语义差异清单（先查清楚，再定判据）

### 3.1 两侧的对象语义

| 项 | 本仓 | FCL |
|---|---|---|
| 胶囊 | `Capsule(point_a_mm, point_b_mm, radius_mm)`，中轴线段给任意两端点 | `fcl.Capsule(radius, lz)`，中轴固定为z轴上`[−lz/2, +lz/2]`，**总长lz+2r** |
| 四元数 | `rotation_xyzw`（标量在**后**） | `Transform(q, t)`的q是**wxyz**（标量在**前**） |
| 接触判据 | `separation < 0`才报事件（**恰好相切不报**） | `collide`把恰好相切判为碰撞（`d ≤ 0`闭区间） |
| 单位 | mm | 无量纲（调用方自负）——本对拍全程按mm喂 |

映射的正确性由数据反证：若胶囊长度或四元数次序映射错了，分离构型的偏差会到1e+1 mm量级
（见第六节注错结果），而实测是2.3e-13 mm。

### 3.2 FCL的三条路**不是同一个算子**（本次对拍最贵的一条发现）

| FCL调用 | 走哪条实现 | 实测行为 |
|---|---|---|
| `distance(enable_signed_distance=False)` | 解析特化 | 分离构型精确；球-球/球-胶囊相交时返回**哨兵`−1.0`**（不是深度）；**胶囊-胶囊相交时返回真的负距离** |
| `distance(enable_signed_distance=True)` | libccd | 两种构型都给数，但**连分离构型的精度也一起降到绝对约1e-6**，且会大错（见第四节） |
| `collide(...).contacts[].penetration_depth` | 解析接触生成 | 球-球/球-胶囊**精确**；**胶囊-胶囊会大错**（见第四节） |

因此判据的算子路由是：`distance.unsigned`不是哨兵就用它，是哨兵才退到`collide`的接触深度。
路由**只看FCL自己的返回值**，不看我们的数（否则对拍是循环的）。九个格子实际落位：

| 格子 | 用的FCL算子 |
|---|---|
| 球-球 / 球-胶囊 / 胶囊-胶囊 的`separated`与`grazing` | `distance.unsigned` |
| 胶囊-胶囊 的`penetrating` | `distance.unsigned`（该对有带符号的解析实现） |
| 球-球 与 球-胶囊 的`penetrating` | `collide.penetration_depth` |

这张表被`criteria.json`的`expected_peer_operator`钉死、被J3判。它变了，说明同行换了实现，
上一轮结论必须重新审——这是轴7规则5"金标不许为让改动通过而漂移"的对偶：**同行漂了也要停下来看**。

## 四、判据表

| 判据 | 量 | 容差 | 理由（摘要，全文见`criteria.json`） |
|---|---|---|---|
| J1 | 分离量（本仓闭式解 对 FCL同构型算子） | `abs ≤ 1e-9 mm` **或** `rel ≤ 1e-11` | 坐标量级上界约1000mm，float64在该量级ulp约1.1e-13mm；实测2700组上界2.27e-13mm，取1e-9留约4400倍余量。判据取"二者其一"是因为`grazing`带的分离量本身可小到1e-7mm，那里rel被放大到5.6e-7而abs仍在1e-13 |
| J2 | 接触判据（本仓报事件 对 FCL的`collide`布尔） | 分歧数=0，零容忍 | 漏报为零是碰撞查询对外唯一的硬承诺，布尔量没有"差不多" |
| J3 | 同行算子路由 | 九格逐条相等，零容忍 | 三条路精度差六个量级，路由一变结论必须重审 |
| J4 | 样本规模 | `count==2700`且`missing==0` | 防"悄悄缩小扫描然后宣布一致" |

### 实测结果（2026-08-05，本机，seed=20260805，2700组）

| 格子 | FCL算子 | max abs (mm) | max rel | 判据分歧 |
|---|---|---|---|---|
| 球-球 / separated | `distance.unsigned` | 1.42e-14 | 7.12e-14 | 0 |
| 球-球 / grazing | `distance.unsigned` | **0.0** | 0.0 | 0 |
| 球-球 / penetrating | `collide.penetration_depth` | **0.0** | 0.0 | 0 |
| 球-胶囊 / separated | `distance.unsigned` | 2.27e-13 | 5.61e-12 | 0 |
| 球-胶囊 / grazing | `distance.unsigned` | 1.35e-13 | 5.59e-07 | 0 |
| 球-胶囊 / penetrating | `collide.penetration_depth` | 1.09e-13 | 6.63e-11 | 0 |
| 胶囊-胶囊 / separated | `distance.unsigned` | 5.74e-14 | 2.49e-12 | 0 |
| 胶囊-胶囊 / grazing | `distance.unsigned` | 6.22e-14 | 1.61e-07 | 0 |
| 胶囊-胶囊 / penetrating | `distance.unsigned` | 7.24e-14 | 3.17e-11 | 0 |
| **全体** | — | **2.27e-13** | 5.59e-07 | **0** |

球-球的两个格子`max abs = 0.0`是逐位相同，不是"小于容差"。

## 五、已知失效清单

每条一行理由，禁止静默skip。

1. **FCL的`distance(enable_signed_distance=True)`会大错，本案例因此不用它作判据。**
   2700组里109组偏差>1e-5mm、663组>1e-6mm，最大3.42e-3mm。最坏的一条（index 1356，
   球-胶囊`grazing`，真值0.007508269363576403mm）FCL给0.010930570785204657mm，
   **相对误差45.6%，而且这还是一个分离构型**。真值用Fraction精确算过，本仓的数与它逐位相同。
   `GST_LIBCCD`与`GST_INDEP`给出逐位相同的错值，换solver无效。
2. **FCL的`collide().penetration_depth`在胶囊-胶囊上会大错**，因此该对改用`distance.unsigned`。
   900组胶囊-胶囊里128组偏差>1e-5mm，最大2.11mm（index 2597：真值−9.026294739619544mm，
   FCL接触深度给−11.139532930858906mm，而FCL自己的`distance.unsigned`给−9.026294739619537mm）。
   **FCL在这里自相矛盾**，本仓的数与它自己那条解析路一致。球-球与球-胶囊上该算子精确（0.0与1.09e-13）。
3. **恰好相切（分离量精确为0）的边界不在本案例的可达集内。** 随机采样打不中精确零。
   两侧在该点判据相反：FCL的`collide`判碰撞（`d ≤ 0`），本仓不报事件（`separation < 0`）。
   这是**已知且刻意**的差异，不是缺陷——但它意味着J2的"零分歧"结论只覆盖`|分离量| ≥ 1e-7mm`。
   构造性的相切用例属`cases/segment_distance`（五条退化分支那条），不在本案例。
4. **本案例只跑了macOS/arm64一个平台。** FCL的解析路是纯double运算，跨平台差异预期在ulp量级，
   但**没测过就是没测过**。Windows与Linux的复核是待办（见第七节）。
5. **未覆盖圆柱、盒、网格。** 本仓narrow phase今天只有球/胶囊族；FCL支持`Box`/`Cylinder`/`Convex`，
   对拍面等`narrow.py`落地后再扩（plans/02 T2）。
6. **`grazing`带的极小值区依赖摆位二分的收敛。** 二分停在`hi−lo ≤ 1e-15·max(1,hi)`，
   因此实际分离量可能与目标差若干ulp。这只影响"样本落在哪个带"，不影响对拍本身。

## 六、必须红：三条注错

轴7规则6要求每道门有"它必须红"的输入。三条注错的共同点是
**仓内自洽测试全都抓不到、只有另一个证人能指出来**：

| 注错 | 内容 | 实测max abs偏差 | 越界样本 | 判据分歧 | J1 |
|---|---|---|---|---|---|
| `quaternion_order` | 把本仓的xyzw当wxyz用 | 1.508e+2 mm | 360/540 | 153 | 红 |
| `radius_sum` | 分离量只减一个半径 | 4.992e+1 mm | 540/540 | 0 | 红 |
| `capsule_half_length` | 把胶囊半长当全长 | 3.938e+1 mm | 213/540 | 92 | 红 |

`radius_sum`那条"判据分歧=0"值得单看：漏减一个半径把分离量整体抬高，
`BroadPhaseCollisionQuery`的**布尔判据在多数样本上仍然正确**——**这正是布尔门抓不到数值错的证据**，
也正是J1（数值）与J2（布尔）必须并列、不能互相顶替的原因。

注错**只改本仓这一侧看到的输入**，FCL那侧始终拿原始输入——否则两边一起错，门就哑了。
三条注错各跑540组（`fault_per_cell=60`），由`tests/cases/test_peer_fcl_distance.py`断言必红。

## 七、档位与负载级

- **档位**：A档（0015第三节：只用现有shapes/collision，零新增`src/`代码）；
- **负载级**：本机批级（pytest marker `batch`，与`accept.py full`的120秒预算同源）。
  实测：单次2700组对拍端到端**0.85秒**（含进程启动、numpy与fcl导入、run package落盘）；
  三条注错各540组约0.29秒；`pytest tests/cases -q`全部5条**1.42秒**。
  申报批级而非交互级的理由：它派生子进程且依赖外部环境，把它放进交互档会让
  "交互级<1秒"的口径受同行环境的启动开销支配——**分级只回答"在哪跑"，不减轻优化要求**（spec/13零之二）。
- **同行库缺席时skip**：`validation/.venv`不存在或`import fcl`失败，测试skip并给出具体是哪一步缺。
  本仓的accept绝不因为一台机器没装同行库而红。

## 八、本案例不是什么（Drake形制）

1. **不是"FCL是对的"的证明。** FCL在本案例里被抓到两处大错（第五节1、2）。
   两侧一致只说明两个独立实现在同一批输入上给了同一个数；不一致时先怀疑双方。
2. **不是精度基准。** 它比的是两个实现的一致性，不是任一方对解析真值的绝对精度。
   绝对精度归`cases/segment_distance`（闭式解+五条退化分支）。
3. **不是性能对比。** 本案例一次FCL调用要过Python绑定与三次不同的查询，
   拿它的墙钟去比"我们快还是FCL快"是错的口径。性能门归T1轨道。
4. **不是broad phase的对拍。** 只比了`BroadPhaseCollisionQuery`最终报不报事件，
   没比AABB本身、没比FCL的`DynamicAABBTreeCollisionManager`。broad⊇narrow超集不变量
   归`cases/broadphase_superset`。
5. **不是把FCL引进本仓的第一步。** 见decisions/0025：同行库永远不进`dependencies`，
   永远不被`src/physics_engine` import，永远只住在`validation/`。
6. **不覆盖圆柱/盒/网格/凸包**，见第五节第5条。

## 九、怎么复跑

```bash
# 一次性：建同行环境（详见validation/README.md）
bash validation/setup_peer_env.sh

# 跑一次对拍，产物落work/（已被.gitignore排除）
validation/.venv/bin/python validation/peer_fcl/run_comparison.py \
    --out-dir work/peer_fcl --name run-001

# 跑门（同行库缺席则skip）
.venv/bin/python -m pytest tests/cases -q -m batch
```
