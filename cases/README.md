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
| [`bouncing_ball_restitution`](bouncing_ball_restitution/case.md) | **引擎第一条被声明的瞬态接触**。无阻尼三条闭式守`t_c = π/ω`、`δ_max = v_in/ω`、`e = 1`；阶段2再以独立三段闭式跨`ζ=1`守欠阻尼`e=0.8`与过阻尼`e=0.05`，同时验合力归零时长、快根、物理耗散与能量账残差。**必红专防**把瞬态穿透写成准静态`N/k` | A | interactive | 7条 | `tests/cases/test_bouncing_ball_restitution.py` |
| [`ten_ball_funnel`](ten_ball_funnel/case.md) | **10球最小漏斗组合**：45个场景候选逐帧走球体broad/narrow，只有活动对进入罚势与dashpot；三个解析平面、重力与耗散账同跑。只判质心/速度/穿透/动态裁剪/两类接触/耗散/能量残差，不伪造十球轨迹oracle。候选仍全量预声明、平面未进流水线，故场景③仍为partial | C | local_batch | `criteria.json` | `tests/cases/test_ten_ball_funnel.py` |
| [`two_body_spring`](two_body_spring/case.md) | 拉伸能与重力能的闭式值；两体振动角频率`ω=sqrt(k/μ·1000)`（**1000倍单位bug的捕手**）；质心不动 | B | interactive | 3条 | `tests/cases/test_two_body_spring.py` |
| [`cantilever_self_weight`](cantilever_self_weight/case.md) | 自重悬臂端点挠度对教科书闭式，**二阶收敛实测比恰为4.000**；牛顿一步收敛（二次能量） | B | interactive | 1条 | `tests/cases/test_cantilever_self_weight.py` |
| [`axial_stretch_hessian`](axial_stretch_hessian/case.md) | 拉伸项梯度与Hessian对**精确有理算术**金标（`Fraction`二阶前向jet + 手推闭式，两条独立路径逐位相等）；**闭合决策0024第六节登记的缺口** | B | interactive | 6条 | `tests/cases/test_axial_stretch_hessian.py` |
| [`rectangular_section_springback`](rectangular_section_springback/case.md) | **阶段4第一条截面非线性**：矩形中点纤维逐点理想弹塑性、显式历史、N/M求积与自由回弹局部平衡；连续闭式`M(κ)`+离散`Fraction`金标、8→64点二阶收敛、六格必红。只完成局部截面本构切片，不冒充独立截面场自由度 | B | interactive | 2条 | `tests/cases/test_rectangular_section_springback.py` |
| [`kirchhoff_section_vertex_springback`](kirchhoff_section_vertex_springback/case.md) | **阶段4第一条全局截面站点**：WDS式3节点+2边扭角生成easy-axis曲率，弹塑性纤维M与一致Hessian进入`EnergyRegistry`/全局Newton，失败不提交历史；独立`Fraction`验全局力/切线/回弹，固定WDS源SHA夹具验曲率/能量/11维梯度。只证明引擎侧单站源级兼容，不冒充WDS已采纳或整杆 | B | interactive | 2条 | `tests/cases/test_kirchhoff_section_vertex_springback.py` |
| [`generator_determinism`](generator_determinism/case.md) | 参数化生成器：同参数逐字节相同（**含换`PYTHONHASHSEED`的子进程**）；22条一位扰动全改变声明；特征长度乘2时40个长度**逐位**翻倍；产出的形对圆柱/胶囊教科书闭式，rel 1e-15 | A | interactive | 5条 | `tests/cases/test_generator_determinism.py` |
| [`scalar_diffraction_airy`](scalar_diffraction_airy/case.md) | 艾里斑`E(x)=2·J1(x)/x`对贝塞尔积分的独立求值，abs 1e-12（**判绝对不判相对：J1有零点**）；首零`3.8317059702`；角度↔空间频率↔半径的**单位往返** | B | interactive | 2条 | `tests/cases/test_scalar_diffraction_airy.py` |
| [`fts_instrument_line_shape`](fts_instrument_line_shape/case.md) | 无切趾ILS半宽`1.2067091288/(2L)`对独立求根，半高点上ILS恰为0.5；Norton-Beer三组`Σ Ci = 1`（实测残差**恰为0**）；通量代价闭式对Simpson，排序弱>中>强 | B | interactive | 2条 | `tests/cases/test_fts_instrument_line_shape.py` |
| [`two_beam_interference`](two_beam_interference/case.md) | 双光束`I=I1+I2+2√(I1I2)|γ|cosΔφ`对60位Decimal参考；杨氏`Δx=λL/d`对**精确两点源几何**（傍轴偏差3.209e-6可算可断言）；**能量守恒**：条纹平均恒为`I1+I2`（四构型实测残差**恰为0**）；迈克尔逊级次15802.78量出"相位精度随级次线性退化"；**与FTS的桥**：ILS首零=整扫描程滑过一个整条纹 | B | interactive | 4条 | `tests/cases/test_two_beam_interference.py` |
| [`mutual_inductance_coaxial`](mutual_inductance_coaxial/case.md) | 同轴圆环互感Maxwell闭式`M = μ0·√(r1r2)·[(2/k−k)K−(2/k)E]`对Neumann双回路积分，rel 1e-12（**7组构型，含d=0与k=0.002的极远场**）；椭圆积分对Carlson（**模k与参数m两种约定各一对函数并互钉**）；互易零容差；远场退化到偶极子**收敛阶实测1.9806→1.9999**；匝数`N1·N2`逐位精确；mm↔m往返零容差 | B | interactive | 6条 | `tests/cases/test_mutual_inductance_coaxial.py` |
| [`rigid_body_free_flight`](rigid_body_free_flight/case.md) | 四层：无力矩下**惯性系**角动量与转动动能漂移随步长降16倍（区间`[15,17]`，实测15.68—16.12）；轴对称进动率`λ=ω3(Ia−It)/It`**带符号**（扁体正长体负，rel 1e-8）；中间轴增长率`σ=Ω√((I3−I2)(I2−I1)/(I1I3))`与稳定轴振幅上界闭式、翻转数2/0/0；四元数范数**归一化前**偏离≤1e-11。**七门×五注错的必红矩阵，含一条主动找出的盲区（无力矩判据对`I→c·I`全盲）** | B | local_batch | 11条 | `tests/cases/test_rigid_body_free_flight.py` |
| [`large_deflection_cantilever`](large_deflection_cantilever/case.md) | 大挠度悬臂端点位置对Bisshopp-Drucker 1945椭圆积分闭式，二阶收敛实测比3.9612/4.0512/4.0600；几何精确项比小挠度理论准**1519倍**；固支Voronoi退回`h`必掉到一阶（正向必须红） | B | local_batch | 1条 | `tests/cases/test_large_deflection_cantilever.py` |
| [`euler_buckling`](euler_buckling/case.md) | 临界载荷对`Fc = π²EI/(bL)²`，**b=1/2/0.5三种边界条件由特征方程求根而非抄表**；对铰接刚性链的**精确特征值**命中到2e-6（比连续解紧100倍）；二阶收敛3.9916/4.0033/4.0226；保长扰动的能量在0.7Fc升、1.3Fc降（**唯一验`energy()`符号的门**）；正弦半波在四个试验形状里给出最低临界载荷，间隙5.9e-7 | C | local_batch | 1条 | `tests/cases/test_euler_buckling.py` |
| [`norris_thin_strip`](norris_thin_strip/case.md) | Norris 1970薄带临界态：片电流分布对**50位十进制**参考，rel 1e-12；电流守恒`∫K dx = I`两条求积（代换式1e-9 对 直接式1e-6，**同一恒等式差三个数量级**）；`b`的两个零容差极限；损耗渐近`i⁴/3`与`i³/3`（**幂次4与3是这两条式的招牌**）。**验公式不验引擎，不进0040分母** | B | interactive | 4条 | `tests/cases/test_norris_thin_strip.py` |
| [`incline_slide_threshold`](incline_slide_threshold/case.md) | **引擎第一个带接触的案例**：斜面滑动阈值`θc = arctan(μs)`，**阈值两侧行为定性相反**（θc−1e-7°粘、θc+1e-7°滑，零容差）；重力分解`N=Wcosθ`/`T=Wsinθ`五档倾角实测偏差**≤2.2e-16（1 ulp）**；**阈值与两个罚刚度、与质量全部无关**各自一道门（换一个数量级判据必须照样过）；return-map投影后落在锥面上的自洽门。**判力与阈值，不判位置**——穿透是模型自带的O(1/k) | C | interactive | 2条 | `tests/cases/test_incline_slide_threshold.py` |
| [`friction_hysteresis_loop`](friction_hysteresis_loop/case.md) | **引擎第一次记住历史**：位移控制拖拽的弹塑性滑块。力饱和`|T|=μN`**零容差**（构造保证的等式）；整循环耗散对闭式`4μN(u_max−u_y)`三个幅值**实测偏差0**（分段线性+拐点落在步长边界，**不对齐时1e-5，机械是真的**）；**路径相关：同一位置两条路径力之比恰为1/2、判别相反、锚点各差0.5u_y**——这一条验的是形制不是公式，**锚点若可从位形导出则前一个接触案例照样全绿**；反向门：纯弹性往返不许挪锚点、不许产生耗散 | C | local_batch | 3条 | `tests/cases/test_friction_hysteresis_loop.py` |
| [`three_sphere_pyramid`](three_sphere_pyramid/case.md) | **引擎第一个多体接触**（球-球两端都是自由度，Hessian第一次有跨节点耦合块）。临界摩擦`μc = 1/(3√3)`**纯几何、与质量重力刚度全无关**；Chrono形制两侧判据：`μ/μc=1.00001`撑住、`0.99999`塌（零容差）；静力分解`F=W/√3`、`N=3W/2`、`T=W/(2√3)`。**罚柔度让T/N带O(1/k)偏差**（穿透改变接触几何本身，与斜面案例的关键差别），故多一条**收敛阶门**：刚度涨10倍偏差降10.00/10.50倍，区间`[8,12]`不写死为10 | C | interactive | 3条 | `tests/cases/test_three_sphere_pyramid.py` |
| [`peer_fcl_distance`](peer_fcl_distance/case.md) | 同行库对拍：球/胶囊解析距离对python-fcl 0.7.0.11（版本钉死，漂了就红），3类形状对×3构型带×300组=2700组；**判据正本是`criteria.json`不是本页**。抓到FCL两处大错（`enable_signed_distance`最坏相对误差45.6%、胶囊-胶囊接触深度最大偏2.11mm且**FCL自相矛盾**），本仓的数与Fraction精确值逐位相同 | A | local_batch | `criteria.json` | `tests/cases/test_peer_fcl_distance.py` |
| [`capstan_tension_ratio`](capstan_tension_ratio/case.md) | **第一条多槽位同时滑移**：带材绕真机导轮（R=50mm、半宽8.5mm，现场实测尺寸）90°，8个接触同时走`advance_contacts_quasistatic`。两条**恒等式**（全滑移下`\|T\|/(μN)`实测**2.22e-16**落在锥面上；法向力＝两侧张力的法向合量，用**实际段方向**不用名义`sin(Δφ/2)`）＋两条**收敛结果**（逐节点比对精确离散式`(1+μtan(Δφ/2))/(1−μtan(Δφ/2))`，载荷步一阶收敛实测比2.10/2.06）。**不判端到端比**——两端是半节点，实测随载荷步非单调。永久`hypothesis_only` | B | local_batch | 4条 | `tests/cases/test_capstan_tension_ratio.py` |

| [`winding_line_endtoend`](winding_line_endtoend/case.md) | **第一条把四件事装在一条链路上跑的**：放线端张力（力边界条件）＋导向轮罚接触与多槽位库仑摩擦＋收线端位移控制＋排线横动把带材边缘顶上法兰内环面。四条闭式：放线端轴力＝外载（rel 1e-9）；落位点/放线端＝`exp(μθ)`（480步实测1.604693，偏差1.695e-3）；半间隙6.5mm阈值两侧**零容差**；蹭边力＝`T·δ/L`（两档实测差0.11%）。**混合控制照真机来**——力在放线端、位移在收线端；装反了张力会沿走向递减。**跑出了`PenaltyAnnulusLimit`一个真bug**（法兰朝向从限位符号推出来，横动过原点即失效）。**不让主分母动**：场景⑥四位仍partial | C | local_batch | 4条 | `tests/cases/test_winding_line_endtoend.py` |

| [`roller_skew_lateral_drift`](roller_skew_lateral_drift/case.md) | **导轮轴偏斜引起的稳态横漂，而且它不需要材料输运**——Shelton正规入轮条件`y'(L)=θ_r`本身就是输运的结果，写成边界条件后稳态是静力边值问题。引擎DER弯曲＋轴向张力对闭式`y_ss = θ_r·L·f(KL)`，`f(u)=(sinh u−u cosh u)/(u(1−cosh u))`，**二阶收敛**（实测比3.739/3.861/3.928）。与WDS `research/04`引的**五个独立数字**逐条对拍。**反直觉结论：张力越大横漂越大**（`f`单调增，10→40N给+11.3%），闭式与引擎各判一次。振幅二次律两档一致（9.97e-3/9.93e-3 per mm²），1mm处+0.9%由张力自升解释、验算对上。**装配公差直接输出**：半间隙6.5mm下，跨长100/200/300/500mm的临界偏斜为5.381°/2.489°/1.546°/**0.852°** | B | local_batch | 4条 | `tests/cases/test_roller_skew_lateral_drift.py` |
| [`helix_laydown_closure`](helix_laydown_closure/case.md) | **第一条把位姿时间线与送带账放在一起判的**：落位点的弧长坐标、世界系槽三标架、入射角、所需送带率与**闭合残差**。金标全闭式（解析螺旋线＋"姿态随便选、平移反解"的解析位姿，故闭合恒成立）。**核心判据是残差按方向拆开**——送带账偏0.7mm与位姿举偏1.5mm的残差模长同量级，而前者弧长差**恰为0.7**、几乎全在沿槽（多放带材就好），后者弧长差**恰为零**、全在横向且不可约（**多放少放都修不掉**）。收敛阶实测：位置语义`chord_linear`二阶／`hermite_tangent`四阶（比值4.00/4.30/3.87与16.02/18.47/14.97，**区间不相交**），帧语义`hold_station`一阶／`reorthonormalised_linear`二阶。**平面圆退化档**（`p`=0、闭合、走1.273匝）挠率恒零、三标架恒定、入射角恒零（实测2.96e-08 rad）。**订正了"跨段随臂变长"那条错**：两个端点都是世界系常量，跨段长度由类型保证与`t`无关 | B | local_batch | 5条 | `tests/cases/test_helix_laydown_closure.py` |

| [`free_span_tension_step`](free_span_tension_step/case.md) | **第一条让张力由速度差生成的**：`T = M/R`被换成"输运账＋带材弹性＋放线盘力矩平衡"，于是**收线端速度阶跃第一次成为一个真扰动**。稳态`T = M/R + c·v/R²`，`c=0`时与`drives.SpoolTension`**逐位相同**（旧模型是本模型的零阻尼稳态特例）。阶跃闭式带一个零点（张力连续而速度差当场跳）：`t_p=(π−acos ζ)/ω_d`、超调`exp(−ζ(π−acos ζ)/√(1−ζ²))/(2ζ)`，两档ζ实测9.03e-6/7.35e-5。**开环几乎不衰减**：真实量级轴承给`ζ=0.0132`，`c=0`时跑满12.08个周期幅值比**0.99999998**——一次10%线速度阶跃摆1.05 N（20 N的5.3%）且自己不会停。半隐式Euler**一阶**（比1.99906/1.99911/1.99868）。**方向门跑出了反面**：两端对调后张力单调掉、7.595 ms转速穿零被`ω≤0`接住。**盲区实测**：`exp(−ζΦ/√(1−ζ²))`在本仓的第三支，`ζ=0.5`处与恢复系数式差一个ulp | B | local_batch | 7条 | `tests/cases/test_free_span_tension_step.py` |

| [`anisotropic_friction_ellipse`](anisotropic_friction_ellipse/case.md) | **各向异性摩擦椭圆的η-return，以及"径向缩"为什么是错的**：圆上"沿径向缩"、"欧氏最近点投影"、"最大耗散原理的外法向流动"三件事恰好重合，椭圆上互不相同而**只有第三件是物理**。四条独立闭式：支撑函数`h(ψ)`、椭圆半径`ρ(ψ)`、最高短缺`(a−b)²/(a²+b²)`、径向返回违反外法向的最大正弦`(a²−b²)/(a²+b²)`。`μ_∥:μ_⊥=5:1`下实测**混合角耗散最高短缺61.538%（在45°）**、径向返回的滑移方向最坏偏**67.38°**、**横向摩擦力被高估恰好5倍**；关联流动侧`\|sin\|≤4.9e-15`。**退化是构造出来的**：两系数逐位相等时原样转交`coulomb_return_map`（连报错一起），另有一条门强走通用路径防转交掩盖错误。混合角回线无闭式，改判**外功与塑性功两条独立记账相等**（O(h²)，细化比3.98/3.99/3.99/4.00）＋**稳态圈逐位闭合**，顺手补上`friction_hysteresis_loop`第六条"多圈稳态性没验" | B | local_batch | 6条 | `tests/cases/test_anisotropic_friction_ellipse.py` |

| [`anisotropic_rod_twist`](anisotropic_rod_twist/case.md) | **第一条有材料帧的整杆，也是全仓第一条扭转金标**。四条闭式各自独立于内核：螺旋线`κ=R/(R²+p²)`、`τ=p/(R²+p²)`二阶收敛（比值恒4.00），**`κ2`是结构零**（1.07—8.83e-15，**不随h下降**，另有一条门专判它没在下降）；**易/难轴互换**——参考`d1`转90°挠度比实测999.99972 vs 1000（偏差2.79e-7，随载荷平方下降＝几何非线性），**同行点名了这个失效模式却没有门守着**；端扭矩`θ=M·L/GJ`偏差4.885e-15、牛顿一步收敛、扭率极差6.85e-17；**Gauss-Bonnet holonomy**——切向`x̂→ŷ→ẑ→x̂`围出三直角球面三角形，面积`4π/8=π/2`，重输运后`γ₃−γ₀`与`−π/2`**逐位相同**。**最后一条判据期望的是错误答案**（不重输运时扭转能量恒为`0.0`，零容差）——那是「抄了公式不抄retransport外层循环」的分辨力本身。顺带量到固支半格柔度的**第三例**（不订正只有一阶，`3h/2`订正后二阶3.876/3.939/3.968） | B | local_batch | 4条 | `tests/cases/test_anisotropic_rod_twist.py` |

| [`span_disturbance_channels`](span_disturbance_channels/case.md) | **第一条建"扰动从哪儿进来"的**：臂动经落位点几何变成收线端速度、人手横向触碰让直线段变折线。**正面否掉一条错误因果**——0066说"真机上跨长逐样点变"，而plans/14第3.3节订正后的场景里自由跨两端都是世界系固定的，`L_geo`是常数；臂动改的是落位点几何不是跨长（0071第二节）。机制`σ' = Ω_切向/κ`，于是**平面圆退化档必然为零**（κ恒定⟹送带率恒定），实测`c=0`档20000步2.2752e-11 N、`c=50`档1.1376e-11 N——**这一条最初写的是零容差，实测当场否掉**（病根是`T→L_mat→T`往返的几个ulp，不是扰动）。单调判据在同一条槽上扫Ω=1/2/4/8 rad/s，幅值比2.0295/2.1123/2.3785，对正弦响应闭式`K·a·√(4ζ²ω_n²+ω²)/√((ω_n²−ω²)²+4ζ²ω_n²ω²)`偏差−4.95e-6→−2.11e-4；**零频退回稳态关系`c/R²`逐位相同**。触碰那条：路径增量是几何恒等式（小角度式在δ=4 mm处差1.777e-4）、**中点是最软的地方**（比值0.75，直觉是反的）、尖峰`EA·p/L_mat`**在四档dt上给同一个偏差1.70e-13**（恒等式的指纹）、**横向冲量与材料长度账之间有一条恒等式**（残差8.29e-16 N·s，`−c·dt`那一项是半隐式的指纹）。**反向那条**：撤掉控制器时尖峰**不回落**——四个阻尼周期后还剩71.8%，对闭式偏差1e-8量级。**闭环那一半没做**（归轨道E）。永久`hypothesis_only` | B | local_batch | 7条 | `tests/cases/test_span_disturbance_channels.py` |

## 一之二、每个案例穿过引擎的哪几层（决策0048第三节通则）

**"验公式"与"验引擎"是两类，混在一个计数里计数就不再有意义。**
本节是那条通则的执行面：33个案例按**穿过引擎哪几层**分类，**不按案例数报成绩**。

| 穿过的层 | 条数 | 案例 |
|---|---|---|
| **`state`→`energies`→`solve`（整条路）** | **9** | `cantilever_self_weight`、`large_deflection_cantilever`、`euler_buckling`、**`incline_slide_threshold`（第一条带接触的）**、**`friction_hysteresis_loop`（第一条改写历史的）**、**`three_sphere_pyramid`（第一条多体接触的）**、**`capstan_tension_ratio`（第一条多槽位同时滑移的）**、**`winding_line_endtoend`（第一条把张力、接触、摩擦、蹭边装在一条链路上的）**、**`roller_skew_lateral_drift`（第一条弯曲＋张力的梁-弦边值问题）** |
| **`state`→`sections`→`section_beam`→`energies`→`solve`** | **1** | **`kirchhoff_section_vertex_springback`（第一条WDS运动学全局截面站点与收敛后历史提交）** |
| **`state`→`rod`→`energies`→`solve`＋外层重输运** | **1** | **`anisotropic_rod_twist`（第一条有材料帧的整杆：双轴弯曲＋扭转，且`solve_equilibrium`之上还有一层retransport循环——**它是唯一一条求解入口被包了一层的**）** |
| **`state`→`sections`→局部平衡** | **1** | **`rectangular_section_springback`（第一条截面非线性与逐点材料历史）** |
| `state`→`energies`→`integrate` | **3** | `two_body_spring`、**`bouncing_ball_restitution`（第一条瞬态阻尼接触）**、**`ten_ball_funnel`（第一条10球耗散组合）** |
| `state`→`integrate`（不碰能量装配与求解器） | 3 | `ballistic_free_flight`、`harmonic_oscillator`、`rigid_body_free_flight` |
| `energies`协议层（梯度与Hessian，不求解） | 1 | `axial_stretch_hessian` |
| **闭式计算器（不碰引擎的任何一层）** | **5** | `scalar_diffraction_airy`、`fts_instrument_line_shape`、`two_beam_interference`、`mutual_inductance_coaxial`、`norris_thin_strip` |
| 几何查询／资产治理／模型生成（基座，不碰物理求解） | 6 | `segment_distance`、`rotated_aabb`、`broadphase_superset`、`mesh_asset_integrity`、`generator_determinism`、`peer_fcl_distance` |
| **`transport`自带的输运推进（不碰`state`／`energies`／`solve`）** | **1** | **`free_span_tension_step`（第一条张力由速度差生成的；状态是"材料长度＋放线盘转速"两个标量，不进`State`——**这一行是新开的**，本仓此前没有任何案例落在它上面）** |
| **`motion`→`laydown`的纯运动学（不碰`state`／`energies`／`solve`）** | **1** | **`helix_laydown_closure`（第一条把位姿时间线与送带账放在一起判的）**——它**不是**力学案例：`laydown`只回答"这一瞬落位点在槽的哪里、槽标架朝哪、要放多快"，一个自由度都不解。**这一行是波次二收口时补的**：轨道C把它写进了上面的案例索引表却漏了本表，而**两张表漏一张，分类计数就与目录数对不上**（实测31对32） |
| **`laydown`→`transport`的扰动通道（不碰`state`／`energies`／`solve`）** | **1** | **`span_disturbance_channels`（第一条把落位点几何接到跨段张力上的）**——它是本表里**唯一跨两个模块**的一行：`laydown`给"这一瞬要放多少带材"、`transport`把它变成张力，中间那层`disturbance`只做接线与人手触碰的外扰。**它仍然一个自由度都不解**，所以不进上面任何一行 |
| **`contact`切向本构层（驱动return-map，不装配能量也不求解）** | **1** | **`anisotropic_friction_ellipse`（第一条各向异性摩擦椭圆）**——**它不在上面任何一行，是因为步进器今天接不上椭圆映射**（决策0068裁定不改`advance_contact_quasistatic`的既有行为）。这一行是**欠账的位置**，不是一个新的分类成绩 |

**怎么读这张表**：闭式计算器那5条**判据强度很高**（第1档解析闭式、容差是算出来的、
必红门逐条实测），而且它们**抓到过两个真缺陷**（`materials.py`的长度制漏项、
我们自己0029那条只对一半）。**但它们证明的是"这个公式我们抄对了"，
不是"这个引擎算得对"。**

真正锻炼引擎机械的是前四行那14条，其中**七条带接触**（静置阈值、历史迟滞、
多体金字塔、单次弹跳、十球漏斗、绞盘张力比、绕线链路端到端）。**新增案例时先问它落在哪一行**——
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

主分母的逐位机械计数当前为**12/42**，每场景位数`7/5/10/7/6/7`已由0057冻结；
它不是加权完成度，正本只在`docs/capability_ledger.json`。

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

**缺7条**（其中**第7、8、9条各做了一部分**）：第4条Timoshenko悬臂（缺剪切刚度）、
第5条Michell失稳与第6条局部螺旋屈曲（缺扭转）、
**第8条斜面滑动阈值`incline_slide_threshold`已过，但该条还含无滑滚球**——
要转动自由度参与接触，故**按半条记、分母仍算6**（0050后续片才补齐）；
**第7条三球金字塔`three_sphere_pyramid`已过定性判据**，但Chrono原版**所有接触都有摩擦**（静不定），
本仓做的是球-球无摩擦的**静定变体**（才有闭式μc）——**同样按半条记**；
第9条恢复系数的**单次阻尼碰撞已跨欠阻尼/过阻尼跑通**，但同行题还要求
带重力的连续弹跳总时长，故仍为partial、不进分子；
第13条变换层自洽三件套（缺复数场与FFT）。各自缺的能力见决策0040第二、三节。

**上表其余案例不计入分母**——它们验的是"我们自己的实现有没有内部错误"
（协议门、逐字节门、失败关闭），而C档验的是"这个引擎算不算得对同行公认的物理"。
两者都必要，但**只有后者能回答"我们成了没有"**。
`peer_fcl_distance`是特例：同行库对拍、判据强度高，但它验的是几何查询不是物理求解，
所以归门不归分母。

`rigid_body_free_flight`同样**不计入C档分母**（决策0057）：它的判据来自
research/05第**2.2**节B档，而C档13条固定来自第2.3节；但它仍计入主分母S3.1，
因为自由刚体飞行确实是用户场景③的前置能力。判据强不等于可以自行扩写外部分母。
