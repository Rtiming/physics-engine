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
| [`two_beam_interference`](two_beam_interference/case.md) | 双光束`I=I1+I2+2√(I1I2)|γ|cosΔφ`对60位Decimal参考；杨氏`Δx=λL/d`对**精确两点源几何**（傍轴偏差3.209e-6可算可断言）；**能量守恒**：条纹平均恒为`I1+I2`（四构型实测残差**恰为0**）；迈克尔逊级次15802.78量出"相位精度随级次线性退化"；**与FTS的桥**：ILS首零=整扫描程滑过一个整条纹 | B | interactive | 4条 | `tests/cases/test_two_beam_interference.py` |
| [`mutual_inductance_coaxial`](mutual_inductance_coaxial/case.md) | 同轴圆环互感Maxwell闭式`M = μ0·√(r1r2)·[(2/k−k)K−(2/k)E]`对Neumann双回路积分，rel 1e-12（**7组构型，含d=0与k=0.002的极远场**）；椭圆积分对Carlson（**模k与参数m两种约定各一对函数并互钉**）；互易零容差；远场退化到偶极子**收敛阶实测1.9806→1.9999**；匝数`N1·N2`逐位精确；mm↔m往返零容差 | B | interactive | 6条 | `tests/cases/test_mutual_inductance_coaxial.py` |
| [`rigid_body_free_flight`](rigid_body_free_flight/case.md) | 四层：无力矩下**惯性系**角动量与转动动能漂移随步长降16倍（区间`[15,17]`，实测15.68—16.12）；轴对称进动率`λ=ω3(Ia−It)/It`**带符号**（扁体正长体负，rel 1e-8）；中间轴增长率`σ=Ω√((I3−I2)(I2−I1)/(I1I3))`与稳定轴振幅上界闭式、翻转数2/0/0；四元数范数**归一化前**偏离≤1e-11。**七门×五注错的必红矩阵，含一条主动找出的盲区（无力矩判据对`I→c·I`全盲）** | B | local_batch | 11条 | `tests/cases/test_rigid_body_free_flight.py` |
| [`large_deflection_cantilever`](large_deflection_cantilever/case.md) | 大挠度悬臂端点位置对Bisshopp-Drucker 1945椭圆积分闭式，二阶收敛实测比3.9612/4.0512/4.0600；几何精确项比小挠度理论准**1519倍**；固支Voronoi退回`h`必掉到一阶（正向必须红） | B | local_batch | 1条 | `tests/cases/test_large_deflection_cantilever.py` |
| [`euler_buckling`](euler_buckling/case.md) | 临界载荷对`Fc = π²EI/(bL)²`，**b=1/2/0.5三种边界条件由特征方程求根而非抄表**；对铰接刚性链的**精确特征值**命中到2e-6（比连续解紧100倍）；二阶收敛3.9916/4.0033/4.0226；保长扰动的能量在0.7Fc升、1.3Fc降（**唯一验`energy()`符号的门**）；正弦半波在四个试验形状里给出最低临界载荷，间隙5.9e-7 | C | local_batch | 1条 | `tests/cases/test_euler_buckling.py` |
| [`norris_thin_strip`](norris_thin_strip/case.md) | Norris 1970薄带临界态：片电流分布对**50位十进制**参考，rel 1e-12；电流守恒`∫K dx = I`两条求积（代换式1e-9 对 直接式1e-6，**同一恒等式差三个数量级**）；`b`的两个零容差极限；损耗渐近`i⁴/3`与`i³/3`（**幂次4与3是这两条式的招牌**）。**验公式不验引擎，不进0040分母** | B | interactive | 4条 | `tests/cases/test_norris_thin_strip.py` |
| [`peer_fcl_distance`](peer_fcl_distance/case.md) | 同行库对拍：球/胶囊解析距离对python-fcl 0.7.0.11（版本钉死，漂了就红），3类形状对×3构型带×300组=2700组；**判据正本是`criteria.json`不是本页**。抓到FCL两处大错（`enable_signed_distance`最坏相对误差45.6%、胶囊-胶囊接触深度最大偏2.11mm且**FCL自相矛盾**），本仓的数与Fraction精确值逐位相同 | A | local_batch | `criteria.json` | `tests/cases/test_peer_fcl_distance.py` |

## 一之二、每个案例穿过引擎的哪几层（决策0048第三节通则）

**"验公式"与"验引擎"是两类，混在一个计数里计数就不再有意义。**
本节是那条通则的执行面：19个案例按**穿过引擎哪几层**分类，**不按案例数报成绩**。

| 穿过的层 | 条数 | 案例 |
|---|---|---|
| **`state`→`energies`→`solve`（整条路）** | **3** | `cantilever_self_weight`、`large_deflection_cantilever`、`euler_buckling` |
| `state`→`energies`→`integrate` | 1 | `two_body_spring` |
| `state`→`integrate`（不碰能量装配与求解器） | 3 | `ballistic_free_flight`、`harmonic_oscillator`、`rigid_body_free_flight` |
| `energies`协议层（梯度与Hessian，不求解） | 1 | `axial_stretch_hessian` |
| **闭式计算器（不碰引擎的任何一层）** | **5** | `scalar_diffraction_airy`、`fts_instrument_line_shape`、`two_beam_interference`、`mutual_inductance_coaxial`、`norris_thin_strip` |
| 几何查询／资产治理／模型生成（基座，不碰物理求解） | 6 | `segment_distance`、`rotated_aabb`、`broadphase_superset`、`mesh_asset_integrity`、`generator_determinism`、`peer_fcl_distance` |

**怎么读这张表**：闭式计算器那5条**判据强度很高**（第1档解析闭式、容差是算出来的、
必红门逐条实测），而且它们**抓到过两个真缺陷**（`materials.py`的长度制漏项、
我们自己0029那条只对一半）。**但它们证明的是"这个公式我们抄对了"，
不是"这个引擎算得对"。**

真正锻炼引擎机械的是第一行那3条。**新增案例时先问它落在哪一行**——
若又是一条闭式计算器，它可以进仓，但**不许被算进"引擎能力"那本账**。

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
                scalar_diffraction_airy fts_instrument_line_shape two_beam_interference; do
      PYTHONPATH=src .venv/bin/python cases/$case/generate_oracle.py
    done

## 计入成功标准的是哪些（决策0040，经0048第二节修订为两条并列）

**两条分母，报的时候必须并排**：

| 分母 | 验什么 | 今天 |
|---|---|---|
| **主｜用户六场景端到端**（[plans/04](../docs/plans/04_真实使用场景与能力差距_20260805.md)） | 算不算得了**我们要算的** | **0/6** |
| 从｜同行C档13条标准案例（research/05第2.3节） | 算得**对不对** | **6/13**（逐条重数，见决策0049第十节；此前长期报7是多算了一条） |

**只报后者是恭维自己。** 0040当初选同行案例当分母的理由
（"用别人的题当分母，比自己出题自己打分诚实"）**当时是对的**——
但那是在**还不知道引擎要用来算什么**的时候定的。用户随后给出六个真实场景，
于是主从关系倒过来了：**同行阶梯验的是手艺，用户场景验的是有没有用。**

从分母的构成：C档13条（杆梁族6、接触族3、光学族4），
是MuJoCo/Bullet/Chrono/Drake/PyElastica/SOFA/FEBio等十余家共同承认的标准案例。

**今天13条过6条**（逐条重数，见决策0049第十节——**此前长期报7是多算了一条**）：
第1条自重悬臂（`cantilever_self_weight`）、第2条大挠度悬臂
（`large_deflection_cantilever`）、第3条Euler屈曲（`euler_buckling`，0046打钩）、
第10条艾里斑（`scalar_diffraction_airy`）、第11条FTS仪器线型与第12条Norton-Beer
（同在`fts_instrument_line_shape`内，**一个案例文件顶两条C档**）。

**缺7条**：第4条Timoshenko悬臂（缺剪切刚度）、第5条Michell失稳与第6条局部螺旋屈曲
（缺扭转）、第7—9条接触族（缺接触与摩擦，且要先解0033）、
第13条变换层自洽三件套（缺复数场与FFT）。各自缺的能力见决策0040第二、三节。

**上表其余案例不计入分母**——它们验的是"我们自己的实现有没有内部错误"
（协议门、逐字节门、失败关闭），而C档验的是"这个引擎算不算得对同行公认的物理"。
两者都必要，但**只有后者能回答"我们成了没有"**。
`peer_fcl_distance`是特例：同行库对拍、判据强度高，但它验的是几何查询不是物理求解，
所以归门不归分母。

`rigid_body_free_flight`同样**不计入分母**（决策0043第九节）：它的判据来自
research/05第**2.2**节的B档，而分母划的是第2.3节的C档。
**但它的判据强度是第一节表里的第1档（解析闭式，Drake `free_body` + 四家独立实现的
Dzhanibekov）**——分母要不要按"判据强度"划而不是按"哪一节"划，是一次待裁的口径决定。
