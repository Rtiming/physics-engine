# 案例套件（中央索引）

案例是一等交付物（decisions/0015第二条），不是测试的附属品。
本目录的形制按plans/02第四节"案例套件的仓内布局"：

- **布局抄Chrono**：案例与参考数据同库，**生成金标的输入卡/脚本一起入库**；
- **测试组织抄MuJoCo**：conformance测试在`tests/cases/`，与案例目录一一对应；
- **金标形制抄FEBio**：只比确定性量、零容差、判据表逐条带理由；
- **oracle清单抄FTS**的`manifest.json`（轴7规则2，参考实现在`src/physics_engine/oracles.py`）；
- **容差声明抄GROMACS**：成对rel/abs**并写理由**。

**本仓不做"不炸即过"**——PyElastica那个`examples/`只判脚本退出码为0、
25个案例README全是0字节，是明确的反面教材（research/05第一节）。

## 一、现有案例

| 案例 | 判据 | 档 | 负载级 | 清单 | conformance |
|---|---|---|---|---|---|
| [`segment_distance`](segment_distance/case.md) | `d = segdist − (r1+r2)`，abs 1e-12mm；五条退化分支+一条一般路径各一条手算用例 | A | interactive | 11条 | `tests/cases/test_segment_distance.py` |
| [`rotated_aabb`](rotated_aabb/case.md) | 八角点枚举 对 Arvo中心-半边长闭式解，abs 1e-9mm | A | interactive | 5条 | `tests/cases/test_rotated_aabb.py` |
| [`broadphase_superset`](broadphase_superset/case.md) | `separation_mm < 0 ⟹ AABB相交`，反例数严格为0（仅球/胶囊族） | A | interactive | 1条+120对语料 | `tests/cases/test_broadphase_superset.py` |
| [`mesh_asset_integrity`](mesh_asset_integrity/case.md) | `sha256(资产)==声明` 且 逐轴`declared_min≤true_min`、`declared_max≥true_max`，均零容差 | A | interactive | 3条 | `tests/cases/test_mesh_asset_integrity.py` |
| [`ballistic_free_flight`](ballistic_free_flight/case.md) | 半隐式Euler误差恰为`+a·T·h/2`、显式恰为`−a·T·h/2`（**同幅反号，判据必须带符号**）；velocity Verlet对常加速度精确，rel<1e-12 | B | interactive | 9条 | `tests/cases/test_ballistic_free_flight.py` |
| [`harmonic_oscillator`](harmonic_oscillator/case.md) | Verlet对`cos(ωT)`收敛比落在`[3.9,4.1]`（**不写死为4**）；漂移排序`explicit > symplectic > verlet`且三者先各自断非零 | B | interactive | 3条 | `tests/cases/test_harmonic_oscillator.py` |
| [`two_body_spring`](two_body_spring/case.md) | 拉伸能与重力能的闭式值；两体振动角频率`ω=sqrt(k/μ·1000)`（**1000倍单位bug的捕手**）；质心不动 | B | interactive | 3条 | `tests/cases/test_two_body_spring.py` |
| [`cantilever_self_weight`](cantilever_self_weight/case.md) | 自重悬臂端点挠度对教科书闭式，**二阶收敛实测比恰为4.000**；牛顿一步收敛（二次能量） | B | interactive | 1条 | `tests/cases/test_cantilever_self_weight.py` |
| [`axial_stretch_hessian`](axial_stretch_hessian/case.md) | 拉伸项梯度与Hessian对**精确有理算术**金标（`Fraction`二阶前向jet + 手推闭式，两条独立路径逐位相等）；**闭合决策0024第六节登记的缺口** | B | interactive | 6条 | `tests/cases/test_axial_stretch_hessian.py` |
| [`generator_determinism`](generator_determinism/case.md) | 参数化生成器：同参数逐字节相同（**含换`PYTHONHASHSEED`的子进程**）；22条一位扰动全改变声明；特征长度乘2时40个长度**逐位**翻倍；产出的形对圆柱/胶囊教科书闭式，rel 1e-15 | A | interactive | 5条 | `tests/cases/test_generator_determinism.py` |
| [`scalar_diffraction_airy`](scalar_diffraction_airy/case.md) | 艾里斑`E(x)=2·J1(x)/x`对贝塞尔积分的独立求值，abs 1e-12（**判绝对不判相对：J1有零点**）；首零`3.8317059702`；角度↔空间频率↔半径的**单位往返** | B | interactive | 2条 | `tests/cases/test_scalar_diffraction_airy.py` |
| [`fts_instrument_line_shape`](fts_instrument_line_shape/case.md) | 无切趾ILS半宽`1.2067091288/(2L)`对独立求根，半高点上ILS恰为0.5；Norton-Beer三组`Σ Ci = 1`（实测残差**恰为0**）；通量代价闭式对Simpson，排序弱>中>强 | B | interactive | 2条 | `tests/cases/test_fts_instrument_line_shape.py` |
| [`large_deflection_cantilever`](large_deflection_cantilever/case.md) | 大挠度悬臂端点位置对Bisshopp-Drucker 1945椭圆积分闭式，二阶收敛实测比3.9612/4.0512/4.0600；几何精确项比小挠度理论准**1519倍**；固支Voronoi退回`h`必掉到一阶（正向必须红） | B | local_batch | 1条 | `tests/cases/test_large_deflection_cantilever.py` |

**在建**：`peer_fcl_distance`（同行库对拍：球/胶囊距离对FCL，plans/02第四节
第一批第2条）由同行对比轨道交付，落地后在上表补一行。

## 二、案例页六必填字段（缺一即红）

`tools/check_case_pages.py`逐条校验，进`accept.py full`档：

1. `## 一、物理/几何设定`——全部参数与单位；
2. `## 二、参考解出处`——闭式解引文献作者/年/式号；无闭式解则生成脚本入库并给SHA；
3. `## 三、判据表`——量→rel/abs→**理由**（必须是表格，表头含"理由"列，至少一行数据）；
4. `## 四、已知失效清单`——每条一行理由，**禁止静默skip**；
5. `## 五、档位与负载级`——A/B/C档 + 交互级/本机批级/服务器级，与清单`load_tier`一致；
6. `## 六、本案例不是什么`——Drake形制的负空间声明。

外加三条结构校验：案例目录必须出现在本页（新案例不许悄悄进仓）；
有`oracle.json`就必须过`physics_engine.oracles`的严格加载器；
清单的`case_id`与`load_tier`必须在案例页里出现。

新建案例：`cp -r cases/_template cases/<新案例>`，照着填。
下划线开头的目录不被当作案例。

## 三、一个案例目录长什么样

    cases/<case_id>/
      case.md              六必填字段的案例页
      oracle.json          oracle清单（engine_oracle_manifest面，自指哈希+生成器SHA）
      generate_oracle.py   金标生成器（脚本SHA钉在清单里，改了不重跑读侧当场红）
      …                    语料（场景文件、数组文件、资产与其生成脚本）

`oracle.json`按`indent=2`落盘：清单的**身份**是规范字节的SHA-256
（`manifest_self_sha256`），与排版无关，所以可以排成人能读的样子——
金标改一个数时`git diff`只显示那一行，而规则5要审的正是这种改动。
压成一行的话整份清单一起变，审无可审。

读清单：`.venv/bin/python -c "import json;print(json.load(open('cases/<case>/oracle.json'))['oracles'][0])"`。

## 四、改金标的规矩（轴7规则5）

**不得为让改动通过而放宽容差、删除测试或重生成金标。**
重生成必须走决策记录，并把决策记录路径写进清单的`regenerated_by`字段——
加载器校验它以`docs/decisions/`开头且文件真的存在，否则拒收。
FEBio的`acceptChanges.py`是这条闭环的出处，本仓的加强是"决策记录必须存在"。

跑一遍全部生成器（改了生成器之后必须做）：

    for case in segment_distance rotated_aabb broadphase_superset mesh_asset_integrity \
                scalar_diffraction_airy fts_instrument_line_shape; do
      PYTHONPATH=src .venv/bin/python cases/$case/generate_oracle.py
    done

## 计入成功标准的是哪些（决策0040）

本仓的成功标准是**案例阶梯**，分母取 research/05 第2.3节的
**C档13条同行标准案例**（杆梁族6、接触族3、光学族4）——它们是
MuJoCo/Bullet/Chrono/Drake/PyElastica/SOFA/FEBio 等十余家共同承认的标准案例。
**用别人的题当分母，比自己出题自己打分诚实。**

**今天13条过6条**：`cantilever_self_weight`、`large_deflection_cantilever`、
`scalar_diffraction_airy`、`fts_instrument_line_shape`（含Norton-Beer两条）。
缺的七条与各自缺的能力见决策0040第二、三节。

**上表其余案例不计入分母**——它们验的是"我们自己的实现有没有内部错误"
（协议门、逐字节门、失败关闭），而C档验的是"这个引擎算不算得对同行公认的物理"。
两者都必要，但**只有后者能回答"我们成了没有"**。
`peer_fcl_distance`是特例：同行库对拍、判据强度高，但它验的是几何查询不是物理求解，
所以归门不归分母。
