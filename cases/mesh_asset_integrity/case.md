# case/mesh_asset_integrity：网格资产完整性与包盒保守性

**本批唯一今天就必红的案例**。decisions/0017第四条记的是一条真实缺陷：
`examples/collision_preview_cell.scene.json`声明的STL SHA-256与WII资产
逐字节吻合，但同一条声明的AABB与真实包盒
`[52.500,−273.236,1133.918]..[453.013,138.239,1559.650]` mm在z轴上**完全不相交**，
`direction:"envelope"`根本不成立。T0已修那条声明，但**引擎至今没有任何门
能发现这类错误**——本案例建的就是那道门。

- 清单：`oracle.json`（3条oracle，`load_tier=interactive`）
- 资产：`assets/tetra.stl`（284字节二进制STL）+ `assets/generate_tetra.py`（生成脚本入库，Chrono形制）
- 语料：`tetra_envelope.scene.json`（合格）、`red_wrong_aabb.scene.json`、`red_wrong_sha.scene.json`（两条必红）、`fitted_not_judged.scene.json`（验拒判行为，非oracle）
- 生成器：`generate_oracle.py`（SHA钉在清单`generator.sha256`）
- conformance：`tests/cases/test_mesh_asset_integrity.py`

## 一、物理/几何设定

单位一律mm。仓内自带的合成资产（WII那个资产不在本仓，包盒保守性判据
没有字节可查，所以必须自带）：一个四面体，二进制STL，284字节
（80头部+4计数+4×50三角形），SHA-256
`f0b168923850a72248c783a1a797f29c71c16d2964a4b7f5cdc26481a70b4b71`。

顶点（刻意不对称、不居中、含负坐标；全部是±0.25的整数倍，float32存储无舍入）：

| 顶点 | x | y | z |
|---|---|---|---|
| v0 | −3.5 | 1.25 | −2.0 |
| v1 | 26.5 | 1.25 | −2.0 |
| v2 | 1.5 | 25.25 | −2.0 |
| v3 | 4.5 | 7.25 | 16.0 |

真实包盒 = (−3.5, 1.25, −2.0) .. (26.5, 25.25, 16.0)。

三条场景语料，都是`units="mm"`、`usage="collision"`、`convexity="exact_convex"`、
`direction="envelope"`，只在声明的SHA与AABB上分岔：

| 场景 | 声明SHA | 声明包盒 | 应有判决 |
|---|---|---|---|
| `tetra_envelope` | 真值 | (−4, 1, −2.5)..(27, 25.5, 16.5) | 两条判据皆过 |
| `red_wrong_aabb` | 真值 | (0, 1, 100)..(27, 25.5, 120) | 包络判据红，违规轴`min_x`、`min_z` |
| `red_wrong_sha` | 末位改一字符 | 同`tetra_envelope` | SHA判据红 |

`red_wrong_aabb`的z轴[100,120]与真值[−2,16]完全不相交——正是那条真实缺陷的
形状；x下界0砍进资产里（真值−3.5），是为了验门**逐轴**报违规而不是只报第一条。

## 二、参考解出处

**无闭式解，生成脚本入库**（Chrono形制：生成金标的输入卡与脚本一起进版本控制）。

- 资产生成脚本：`assets/generate_tetra.py`，纯标准库、确定性输出（固定ASCII
  头部、无时间戳、无路径、无随机数），同一份脚本在任何机器上产出逐字节相同的
  284字节。其SHA-256钉在清单每条oracle的`inputs.asset_generator_sha256`，
  conformance测试逐条校验。
- 真实包盒由上表顶点手推（逐轴min/max）。测试侧**从STL字节解析**出包盒
  与这份手推值对拍——两条路径不共享代码：顶点表是生成侧的真值，
  `struct`解析是读侧的独立测量。
- 判据出处：轴3规则1（内容寻址）与spec/11规则5（保守方向必填）。
  research/05把这条列为"唯一一条今天就必红的案例"。

**为什么解析器放案例侧不放`src/`**：引擎当前的设计是"不解析网格字节、
包盒由声明携带"（`shapes.py`的`MeshAsset`文档）。把25行`struct`塞进产品面
等于顺手改了产品的能力边界，那是B档的裁决不是这条案例的副产品。见第六节。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---|---|---|
| `sha256_matches` | 0 | 0 | 内容寻址是逐字节判据（轴3规则1），容差概念不适用——零容差不是『选了个很严的数』，是『这里没有连续量』。SHA对不上说明字节变了，没有"变得不多"这回事。 |
| `envelope_encloses_asset` | 0 | 0 | 包络保守性是不等式判据（逐轴`declared_min ≤ true_min` 且 `declared_max ≥ true_max`），零容差=不给不等式任何松弛。给了松弛就等于声明"漏出去一点点没关系"，而漏出去的那一点点正是broad phase会漏掉的接触。 |
| `violated_axes` | 0 | 0 | 违规轴名是字符串列表，逐位相等。它把"红"的**理由**也钉住：红在哪个轴、是下界还是上界。 |
| `true_aabb_min_mm` | 0 | 1e-12 mm | 顶点全是±0.25的整数倍，float32存储无舍入、解析回float64精确；1e-12是对『这里没有误差源』的声明，不写==0是不把解析实现的求值次序冻进契约。 |
| `true_aabb_max_mm` | 0 | 1e-12 mm | 同上。 |
| `triangle_count` | 0 | 0 | 确定性整数。 |
| `asset_bytes` | 0 | 0 | 确定性整数（284=84+50×4）。 |

场景语料本身也钉SHA（清单`inputs.scene_sha256`）：改了语料而不重生成清单，
conformance当场红——金标与它的输入必须同批变（轴7规则5）。

## 四、已知失效清单

- **只判`units="mm"`**：本仓与FTS的米制之间零换算代码（plans/02第一节
  identity缺口："共享形状面那天就是静默1000倍"）。门遇到`units="m"`
  **拒判**（抛`GateNotApplicable`）而不是按mm硬算。解封条件：量纲代数落地。
- **只判`direction="envelope"`**：见第六节。
- **不判凸性声明**：`convexity="exact_convex"`说这个四面体是精确凸体——
  它确实是，但门不验这件事（验凸性需要真的做凸包，属B档）。
  声明与事实不符时本案例不会红。
- **不判ASCII STL**：解析器只认二进制形，遇到以`solid`开头的字节直接拒
  （失败关闭，不猜格式）。
- **不判`examples/`里的KUKA连杆**：那个资产在WII仓、不在本仓，无字节可查。
  这正是要自带合成资产的原因。跨仓资产的校验需要资产随run package入库，
  属轴4的事。
- **无静默skip**：本案例的conformance测试没有任何`skip`/`xfail`；
  两条必红语料是**断言红**（`expected`里写着`False`），不是被跳过。

## 五、档位与负载级

- 判据强度：**A档**（零容差的逐字节与不等式判据）；
- 负载级：**interactive**（交互级，无pytest marker）；
- 实测：`pytest tests/cases/test_mesh_asset_integrity.py -q` = 6 passed，0.02秒
  （墙钟仅供落级参考，**不进门**）。

## 六、本案例不是什么

- **不是`fitted`的判据**。`envelope`（包络）承诺"包住资产"，`fitted`（贴合）
  承诺的是"贴着物体的近似形"——它**可以比资产小**，这是它的用途不是它的缺陷。
  把包络保守性套到`fitted`上会得出一堆假红，然后有人会去放宽容差，
  最后包络判据也一起废掉。所以门遇到`fitted`**拒判**（抛`GateNotApplicable`），
  由`fitted_not_judged.scene.json`那条测试钉住这个行为。
  `fitted`该有的判据（贴合误差上界）是另一条案例，本批不做。
- **不是网格碰撞判据**。本案例一个碰撞查询都不跑。资产的包盒对不对，
  与用它做narrow phase是两件事——后者本仓还没有实现。
- **不是资产可信度判据**。SHA对得上只说明字节没变，不说明这个网格是对的
  （模型建错了、单位标错了、坐标系搞反了，SHA照样对）。
  0017那条真实缺陷恰恰是"SHA对得上但包盒是编的"——**内容寻址防篡改，
  不防一开始就写错**。这两层要分开看。
- **不是网格解析器的规范**。案例侧那25行`struct`只够读这一个合成资产：
  不处理ASCII形、不处理大端、不处理属性字节、不做退化三角形检查、
  不做流式读取。它不是产品面的候选实现，别照抄进`src/`。
