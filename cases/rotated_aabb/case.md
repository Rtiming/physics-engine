# case/rotated_aabb：旋转AABB对Arvo闭式解

`world_aabb_mm()`是本仓第二个在算实数值的内核。它用八角点枚举求包盒，
判据侧用Arvo的中心-半边长闭式解——两条独立路径对拍。

- 清单：`oracle.json`（5条oracle，`load_tier=interactive`）
- 生成器：`generate_oracle.py`（SHA钉在清单`generator.sha256`）
- conformance：`tests/cases/test_rotated_aabb.py`

## 一、物理/几何设定

单位一律mm。五条，每条=形状+四元数(xyzw)+平移。局部AABB是**手写字面量**，
不从`shapes.py`取（那样连"局部盒怎么算"都会与被验内核共享路径）。

| oracle | 形状 | 手推局部盒 | 旋转 | 平移 |
|---|---|---|---|---|
| `capsule_offcentre_general_rotation` | 胶囊(0,0,0)→(20,0,0)、r=3 | (−3,−3,−3)..(23,3,3) | 轴(1,2,3)/√14、0.7 rad | (10,−20,30) |
| `rounded_box_fillet_counted` | 圆角盒半边长(30,10,5)、圆角2 | (−32,−12,−7)..(32,12,7) | 绕z、π/6 | (5,6,7) |
| `sphere_translation_only` | 球r=7 | (−7,−7,−7)..(7,7,7) | 单位四元数 | (1,2,3) |
| `finite_cylinder_flange_counted` | 有限圆柱r=45、半宽9、法兰外径50 | (−50,−50,−9)..(50,50,9) | 轴(2,−1,0.5)、1.1 rad | (−40,15,0) |
| `capsule_quaternion_order_trap` | 胶囊(0,−4,0)→(30,−4,0)、r=2.5 | (−2.5,−6.5,−2.5)..(32.5,−1.5,2.5) | xyzw=(0.6,0,0,0.8) | (0,0,0) |

局部盒的推导：胶囊=两端点逐轴min/max再±r；圆角盒=半边长+圆角半径；
球=±r；有限圆柱=法兰外径**取代**基圆半径进x/y，z取±半宽。

## 二、参考解出处

**闭式解**：Arvo, J., *Graphics Gems*, 1990, "Transforming Axis-Aligned
Bounding Boxes"（research/05 §2.1 转述）。中心-半边长分解：

    c = (lo + hi)/2        h = (hi − lo)/2
    c' = R·c + t           h'_i = Σ_j |R_ij|·h_j
    世界盒 = (c' − h', c' + h')

**易错点（本案例第一条专治）**：局部AABB不居中时**不能**直接对`hi`套这式子。
胶囊沿+x放时局部盒是(−3,−3,−3)..(23,3,3)，中心在x=10而不是原点；
对`hi`直接套会把包盒算歪10mm量级。

四条典型错各由谁抓：

| 错法 | 抓它的oracle | 为什么 |
|---|---|---|
| 四元数xyzw/wxyz次序颠倒 | `capsule_quaternion_order_trap` | (0.6,0,0,0.8)按wxyz读会变成绕z转，细长胶囊的包盒差十几mm |
| 旋转矩阵转置 | `capsule_offcentre_general_rotation` | 一般轴下`|R_ij|`不对称，且非居中盒的`R·c`对转置敏感（单轴旋转时`|R|`恰好对称，抓不到——所以必须用一般轴） |
| 平移漏加 | `sphere_translation_only` | 球的旋转项恒等，平移是唯一变量，漏加当场红 |
| 圆角/法兰半径未计入 | `rounded_box_fillet_counted`、`finite_cylinder_flange_counted` | 局部盒判据先红，不用等世界盒 |

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---|---|---|
| `local_aabb_min_mm` | 0 | 1e-12 mm | 局部盒由形状参数的加减与min/max得到，量级50mm下双精度精确可表示；1e-12是对『没有额外误差源』的声明。不写==0是不想把求值次序冻进契约。 |
| `local_aabb_max_mm` | 0 | 1e-12 mm | 同上。 |
| `world_aabb_min_mm` | 0 | 1e-9 mm | 内核枚举八角点取min/max，Arvo走中心-半边长求和——**加法次序不同**，量级50mm下双精度差~1e-14mm；1e-9留五个量级余量。**不写==0**：逐位相等等于把求和次序冻成契约，换成向量化实现就会红，而那不是物理回退。 |
| `world_aabb_max_mm` | 0 | 1e-9 mm | 同上。 |

rel一律为0：判据是绝对包盒坐标，某些轴的坐标接近0（如
`capsule_quaternion_order_trap`的x下界−2.5），相对容差在那里会退化成==0。

## 四、已知失效清单

- **不覆盖网格资产**（`MeshAsset`）：它的局部盒是**声明**的不是算出来的，
  保守性归`case/mesh_asset_integrity`。本案例的五条形状都是参数化形。
- **不覆盖非单位四元数**：位姿层已在`PosedBody.__post_init__`拒收
  （abs_tol=1e-9），本案例不重复验那道门。
- **不覆盖`GeneratedShape`包裹层**：它的`local_aabb_mm`直接转发内层形状，
  没有独立几何。要验的是转发不丢参数，属装配层测试不属oracle。
- **无静默skip**：本案例的conformance测试没有任何`skip`/`xfail`。

## 五、档位与负载级

- 判据强度：**A档**（解析闭式解）；
- 负载级：**interactive**（交互级，无pytest marker）；
- 实测：`pytest tests/cases/test_rotated_aabb.py -q` = 6 passed，0.04秒
  （墙钟仅供落级参考，**不进门**）。

## 六、本案例不是什么

- **不是贴合性判据**。世界AABB是**保守外包**，它比真实形状大是正确行为，
  不是误差。谁要"包盒有多紧"，那是另一个量（体积比），本案例不给。
- **不是SE(3)不变量的判据**。世界AABB随姿态变化——同一个物体转个角度
  包盒就变，这是定义决定的。"整场景旋转后包盒集合不变"是**错的**判据，
  本案例不写（同`case/broadphase_superset`第六节）。
- **不是碰撞判据**。包盒相交不等于形状相交；narrow phase归
  `case/segment_distance`，两者的关系归`case/broadphase_superset`。
- **不覆盖数值极端**：所有量级在1—50mm。远离原点（如坐标1e6mm）时
  `R·c`的相消误差会远超1e-9mm的绝对容差——那时判据要改成相对量，
  而改判据要走决策记录（轴7规则5）。
