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
| [`three_sphere_pyramid_rotational`](three_sphere_pyramid_rotational/case.md) | **引擎第一条转动自由度参与接触的**（准静态路径，SO(3)指数映射的局部图；接触力对质心取矩）。含`ΣM = 0`的全粘着支：`μc = 2−√3`、`F = W/2`、`f = |T| = (1−√3/2)W`、`N = 3W/2`，金标是仓内`Q(√3)`**精确有理算术**重算的（9方程8未知、精确秩8），与research/15逐位相同。顶球自旋钉住是**规范固定**——不钉时整座金字塔沿地面**整体侧滚**是零模（实测条件数7.86e17），而「这不改问题」由约束反力矩**逐位为0.0**证明。收敛阶9.998/10.017，区间`[8,12]`。另带斜面组：约束反力矩`W·R·sinα`五个角度对拍 | C | interactive | 5条 | `tests/cases/test_three_sphere_pyramid_rotational.py` |
| [`peer_fcl_distance`](peer_fcl_distance/case.md) | 同行库对拍：球/胶囊解析距离对python-fcl 0.7.0.11（版本钉死，漂了就红），3类形状对×3构型带×300组=2700组；**判据正本是`criteria.json`不是本页**。抓到FCL两处大错（`enable_signed_distance`最坏相对误差45.6%、胶囊-胶囊接触深度最大偏2.11mm且**FCL自相矛盾**），本仓的数与Fraction精确值逐位相同 | A | local_batch | `criteria.json` | `tests/cases/test_peer_fcl_distance.py` |
| [`capstan_tension_ratio`](capstan_tension_ratio/case.md) | **第一条多槽位同时滑移**：带材绕真机导轮（R=50mm、半宽8.5mm，现场实测尺寸）90°，8个接触同时走`advance_contacts_quasistatic`。两条**恒等式**（全滑移下`\|T\|/(μN)`实测**2.22e-16**落在锥面上；法向力＝两侧张力的法向合量，用**实际段方向**不用名义`sin(Δφ/2)`）＋两条**收敛结果**（逐节点比对精确离散式`(1+μtan(Δφ/2))/(1−μtan(Δφ/2))`，载荷步一阶收敛实测比2.10/2.06）。**不判端到端比**——两端是半节点，实测随载荷步非单调。永久`hypothesis_only` | B | local_batch | 4条 | `tests/cases/test_capstan_tension_ratio.py` |
| [`tension_measuring_roll_resultant`](tension_measuring_roll_resultant/case.md) | **T-M0：第一条验证“传感器读数之前那一层”的案例**。两侧张力按材料行进方向做矢量和，0°/60°/90°/180°四档同时对分量与独立闭式`2T sin(beta/2)`；镜像路线只翻横向力符号。它不接LTS，不让ADC台阶掩盖一个几十牛顿的方向错误 | A | interactive | 1条 | `tests/cases/test_tension_measuring_roll_resultant.py` |
| [`tension_measuring_roll_installation`](tension_measuring_roll_installation/case.md) | **T-M1：敏感轴、tare与支承分层**。同一`(-10,10,0)N` web合力跑单支承/双对称支承/显式0.3—0.7非对称支承三档，轴偏30°时读数按投影下降；gross、tare、net与每支承六组量各自对独立刚体静力。**不从“两个支承”猜份额**，现场支承未确认前永久`hypothesis_only` | A | interactive | 3条 | `tests/cases/test_tension_measuring_roll_installation.py` |
| [`tension_readout_calibration`](tension_readout_calibration/case.md) | **T-M2：原始桥路、tare位置、ADC与五级正反程标定**。0/10/20/30/40N五档每档正反程共同拟合，25N holdout不进拟合；`analog_pre_adc`与`digital_post_adc`分别对手算，110N gross即使tare后net只有90N仍必须报物理过载 | A | interactive | 4条 | `tests/cases/test_tension_readout_calibration.py` |
| [`tension_wind_unwind_dynamic`](tension_wind_unwind_dynamic/case.md) | **T-M3：小型放线—测力轮—收线动态链**。自由跨/放线盘/磁粉离合器/PID接90°测力轮、16位ADC、100μs采样与0.5ms测量时延；好/坏/开环三档30000步对独立离散oracle，另判60→80mm半径下同扭矩张力下降。永久`hypothesis_only`，不进端到端分子 | B | local_batch | 2条 | `tests/cases/test_tension_wind_unwind_dynamic.py` |
| [`model_motion_physics_binding`](model_motion_physics_binding/case.md) | **P3-M0：模型、运动与虚拟物理所有权**。三个模型组件中张力机进static体、工件进kinematic体、机器人显示模型及显示运动track明确排除于虚拟物理；process frame由独立track驱动但不冒充刚体。时间计划中点对独立线性/SLERP金标，A1/E1源状态与累计送带原样保留。接口不含GCW专用字段，WII以后只需adapter | A | interactive | 1条 | `tests/cases/test_model_motion_physics_binding.py` |
| [`model_scene_assembly`](model_scene_assembly/case.md) | **P3-M1：模型输入装配为Scene**。两份合成碰撞资产从包根复读SHA，张力机进static体、工件进kinematic体，组件位姿后乘资产安装偏置；process frame进虚拟运行面，一对显式候选直接进既有CollisionQuery。0s分离、1s仅broad-phase重叠，不冒充网格接触 | A | interactive | 1条 | `tests/cases/test_model_scene_assembly_case.py` |
| [`dynamic_model_scene_free_flight`](dynamic_model_scene_free_flight/case.md) | **P3-M2：dynamic质心状态—惯量—几何位姿闭环**。geometry原点x=10mm、geometry frame内质心x=2mm生成世界系COM x=12mm；主轴角速度πrad/s自由转0.5s后，COM不动、姿态转90°、geometry原点到(12,-2,0)mm。验的是几何确实绕COM转，不是dynamic接触 | A | interactive | 1条 | `tests/cases/test_dynamic_model_scene_free_flight.py` |
| [`dynamic_two_body_contact`](dynamic_two_body_contact/case.md) | **P3-M3：两dynamic刚体显式候选的检测—力矩—26维耦合推进**。偏心COM档用CollisionEvent法向/见证点独立手算两侧作用反作用与`r×F`；对齐COM档对两等质量1mm初压缩的四分之一接触周期闭式，同时判总动量、初始罚簧能与RK4求值次数。不顺序跑两个单体积分器 | A | interactive | 2条 | `tests/cases/test_dynamic_two_body_contact.py` |

| [`rolling_ball_incline`](rolling_ball_incline/case.md) | **本仓第一条让转动自由度参与动态接触的案例**（plans/16的M2＋M3）。两支各判一次：`mu=0.30 > mu_c=0.104` 时无滑滚，`a` 对 `(5/7)g sin(theta)` 偏 **+1.1e-04**；`mu=0.05 < mu_c` 时滑着滚，`a` 对 `g(sin-mu cos)` 偏 **+1.3e-05**、`alpha` 对 `5 mu g cos/(2R)` 偏 **-8.4e-05**。**要害是第三个量**：`a/(alpha R)` 滚动支 **1.0004**、滑动支 **2.512**（闭式 `k(tan/mu - 1)`）——**这个比值就是「滑了」的可观测定义**：一个「永远按滚动公式算」的实现会让两支都等于1，而第四条门专门关那扇门。三条系数（5/7、2/7、7/2）在 `Fraction` 上精确算、零容差。**`k_t` 取50是被实测逼出来的**：`k_t>=5e3` 时进颤振区、`sliding` 恒为真、判据失去分辨力；`k_t=50` 时 `k_t*|v_t|=0.959 N` 恰是闭式所需摩擦 0.9585 N（第三条独立对拍）。**收敛阶未做**、**不覆盖滚动阻力**——两条写在案例页第四节 | A | local_batch | 3条 | `tests/cases/test_rolling_ball_incline.py` |
| [`box_tipping_threshold`](box_tipping_threshold/case.md) | **本仓第一条多接触点对同一平面的动态案例**（plans/16的M4）。几何阈值 `tan(theta) <= w/h` 两侧各一组：`20deg` 撑住——四角全承载、**压心 `x_N = h*tan(theta)`** 实测偏 **+1.8e-05**（它是力矩平衡的**恒等式**，比单点法向力那两条紧40倍，柔度在分子分母同阶相消），倾角摆幅 **3.2e-07 rad/0.6s**、蠕滑对**含饱和分支**的闭式偏 +2.0e-03（朴素式差10.5%，是容差的21倍，于是「哪几个点饱和」也被判到）；`32deg` **真的翻过去**——承载点下标从 `[0,1,2,3]`（整个底面）经 `[1,3]`（抬上坡边）到 `[1,3,5,7]`（`x=+w` 那**一整个侧面**，与起手一个都不重合），末态倾角对 `pi/2` 偏 +6.5e-05。**临界角二分**：软档 `k=50` 夹到 1.56e-04 rad、硬档 `k=200` 夹到 3.91e-05 rad；对刚体闭式 `arctan(w/h)` 的偏差 **-8.59e-04 / -2.15e-04**，**比值 4.00 == 刚度之比**——偏差随 `1/k` 趋零，所以它是罚柔度不是缺陷（闭式 `beta_c = W cos(psi_c)/(2 n_d w k)`，软档实测/预言 1.002）。临界角对 `dt`、时长、起手蠕滑初速**逐位不变**。**必红专防抄金字塔那条**：不许拿「积分器炸了」冒充翻倒（范数偏离 < 护栏的1/10，外加一条 `dt=5e-3` 必抛）。**软罚刚度端一阶修正不准**（`k=10` 时比值1.32）、**收敛阶未做**、**底面形状由调用方声明**——三条写在案例页第四节 | B | local_batch | 3条 | `tests/cases/test_box_tipping_threshold.py` |
| [`winding_line_endtoend`](winding_line_endtoend/case.md) | **第一条把四件事装在一条链路上跑的**：放线端张力（力边界条件）＋导向轮罚接触与多槽位库仑摩擦＋收线端位移控制＋排线横动把带材边缘顶上法兰内环面。四条闭式：放线端轴力＝外载（rel 1e-9）；落位点/放线端＝`exp(μθ)`（480步实测1.604693，偏差1.695e-3）；半间隙6.5mm阈值两侧**零容差**；蹭边力＝`T·δ/L`（两档实测差0.11%）。**混合控制照真机来**——力在放线端、位移在收线端；装反了张力会沿走向递减。**跑出了`PenaltyAnnulusLimit`一个真bug**（法兰朝向从限位符号推出来，横动过原点即失效）。**不让主分母动**：场景⑥四位仍partial。**2026-08-18（决策0088）扩了两处**：① 放线端张力不再是一个手写的浮点数，而是`drives`那条回路（离合器＋卷径换算＋纯积分PID）跑到稳态的输出，经`PointLoad`真的进`solve_equilibrium`——求解器解出的落位点轴力与回路自算的被控点张力相对差**1.7e-3**，而两条腿不共享任何一行代码；**分辨力是逐位的**：拿掉`drives`直接给同一个浮点数，整条状态向量`float.hex()`全同，多加万分之一载荷当场红；② 从一只导轮扩成**R4→R3→R2→R1四轮路由**，每个接触落在**它那只轮**的摩擦锥上（偏差2.2e-16），**逐只轮的比连乘＝总比**（偏差2.40e-13），中间放一只`μ=0`的轮时切向力恰为`0.0`。**订正一条会改变别人判断的话**：本页原先那个"实测偏差2.5e-4"**不是模型精度，是构型选出来的**——同一装配单只轮`μ=0.30`，与理想离散绞盘式的偏差随段数**单调**（1段1.84e-2、4段9.50e-3、6段4.34e-3、**8段2.54e-4**），而第一版恰好用了8段 | C | local_batch | 6条 | `tests/cases/test_winding_line_endtoend.py` |

| [`roller_skew_lateral_drift`](roller_skew_lateral_drift/case.md) | **导轮轴偏斜引起的稳态横漂，而且它不需要材料输运**——Shelton正规入轮条件`y'(L)=θ_r`本身就是输运的结果，写成边界条件后稳态是静力边值问题。引擎DER弯曲＋轴向张力对闭式`y_ss = θ_r·L·f(KL)`，`f(u)=(sinh u−u cosh u)/(u(1−cosh u))`，**二阶收敛**（实测比3.739/3.861/3.928）。与WDS `research/04`引的**五个独立数字**逐条对拍。**反直觉结论：张力越大横漂越大**（`f`单调增，10→40N给+11.3%），闭式与引擎各判一次。振幅二次律两档一致（9.97e-3/9.93e-3 per mm²），1mm处+0.9%由张力自升解释、验算对上。**装配公差直接输出**：半间隙6.5mm下，跨长100/200/300/500mm的临界偏斜为5.381°/2.489°/1.546°/**0.852°** | B | local_batch | 4条 | `tests/cases/test_roller_skew_lateral_drift.py` |
| [`helix_laydown_closure`](helix_laydown_closure/case.md) | **第一条把位姿时间线与送带账放在一起判的**：落位点的弧长坐标、世界系槽三标架、入射角、所需送带率与**闭合残差**。金标全闭式（解析螺旋线＋"姿态随便选、平移反解"的解析位姿，故闭合恒成立）。**核心判据是残差按方向拆开**——送带账偏0.7mm与位姿举偏1.5mm的残差模长同量级，而前者弧长差**恰为0.7**、几乎全在沿槽（多放带材就好），后者弧长差**恰为零**、全在横向且不可约（**多放少放都修不掉**）。收敛阶实测：位置语义`chord_linear`二阶／`hermite_tangent`四阶（比值4.00/4.30/3.87与16.02/18.47/14.97，**区间不相交**），帧语义`hold_station`一阶／`reorthonormalised_linear`二阶。**平面圆退化档**（`p`=0、闭合、走1.273匝）挠率恒零、三标架恒定、入射角恒零（实测2.96e-08 rad）。**订正了"跨段随臂变长"那条错**：两个端点都是世界系常量，跨段长度由类型保证与`t`无关 | B | local_batch | 5条 | `tests/cases/test_helix_laydown_closure.py` |

| [`free_span_tension_step`](free_span_tension_step/case.md) | **第一条让张力由速度差生成的**：`T = M/R`被换成"输运账＋带材弹性＋放线盘力矩平衡"，于是**收线端速度阶跃第一次成为一个真扰动**。稳态`T = M/R + c·v/R²`，`c=0`时与`drives.SpoolTension`**逐位相同**（旧模型是本模型的零阻尼稳态特例）。阶跃闭式带一个零点（张力连续而速度差当场跳）：`t_p=(π−acos ζ)/ω_d`、超调`exp(−ζ(π−acos ζ)/√(1−ζ²))/(2ζ)`，两档ζ实测9.03e-6/7.35e-5。**开环几乎不衰减**：真实量级轴承给`ζ=0.0132`，`c=0`时跑满12.08个周期幅值比**0.99999998**——一次10%线速度阶跃摆1.05 N（20 N的5.3%）且自己不会停。半隐式Euler**一阶**（比1.99906/1.99911/1.99868）。**方向门跑出了反面**：两端对调后张力单调掉、7.595 ms转速穿零被`ω≤0`接住。**盲区实测**：`exp(−ζΦ/√(1−ζ²))`在本仓的第三支，`ζ=0.5`处与恢复系数式差一个ulp | B | local_batch | 7条 | `tests/cases/test_free_span_tension_step.py` |

| [`anisotropic_friction_ellipse`](anisotropic_friction_ellipse/case.md) | **各向异性摩擦椭圆的η-return，以及"径向缩"为什么是错的**：圆上"沿径向缩"、"欧氏最近点投影"、"最大耗散原理的外法向流动"三件事恰好重合，椭圆上互不相同而**只有第三件是物理**。四条独立闭式：支撑函数`h(ψ)`、椭圆半径`ρ(ψ)`、最高短缺`(a−b)²/(a²+b²)`、径向返回违反外法向的最大正弦`(a²−b²)/(a²+b²)`。`μ_∥:μ_⊥=5:1`下实测**混合角耗散最高短缺61.538%（在45°）**、径向返回的滑移方向最坏偏**67.38°**、**横向摩擦力被高估恰好5倍**；关联流动侧`\|sin\|≤4.9e-15`。**退化是构造出来的**：两系数逐位相等时原样转交`coulomb_return_map`（连报错一起），另有一条门强走通用路径防转交掩盖错误。混合角回线无闭式，改判**外功与塑性功两条独立记账相等**（O(h²)，细化比3.98/3.99/3.99/4.00）＋**稳态圈逐位闭合**，顺手补上`friction_hysteresis_loop`第六条"多圈稳态性没验" | B | local_batch | 6条 | `tests/cases/test_anisotropic_friction_ellipse.py` |

| [`closed_loop_tension_step`](closed_loop_tension_step/case.md) | **第一条把控制器接到对象上的**：`transport`的跨段-放线盘＋`drives`的离合器-PID-时延，多速率零阶保持串成一个真闭环。金标是**四阶闭环的精确留数解**（特征多项式手推、根用Durand-Kerner到机器精度），三档并判峰值/ISE/稳态偏移。**`fast`档（τ=0.5ms，一次数值实验不是一台机器）压制比峰值×0.31825、ISE×6.0709e-3即165倍**；**`nominal`档（τ=50ms，真机假设值）峰值×1.07266、ISE×1.38817——闭环把扰动响应变**坏**了**，而闭式早就说了为什么（积分项只在常数项上，把Routh裕度吃掉；振荡实部−5.000→−4.579）。病根是一个无量纲数`ω_n·τ=18.98`——**执行器比对象慢19倍**。裁决A（控制器接制动力矩）的量：两通道稳态权限比**恰为72**，且`c=0`时收线通道权限**恰为零**（零容差）。裁决B（整数倍抽取＋ZOH）：时延线`dt_s`不等于控制周期即失败关闭。1.5两条：闭环稳态读数对设定差1.16e-9而落位点仍高**60.1978%**，方向取反**2.5663324=exp(2μθ)差一个ulp，是平方**。可插拔验收：同一算例换纯P极小增益，峰值×3.11、稳态偏移×1.7e8，**两条都变坏**。**已知失效四条写在案例页**，含"收敛阶比值稳定在2.1—2.3而不是2、本轮没解释清楚故不设门" | B | local_batch | 8条 | `tests/cases/test_closed_loop_tension_step.py` |

| [`anisotropic_rod_twist`](anisotropic_rod_twist/case.md) | **第一条有材料帧的整杆，也是全仓第一条扭转金标**。四条闭式各自独立于内核：螺旋线`κ=R/(R²+p²)`、`τ=p/(R²+p²)`二阶收敛（比值恒4.00），**`κ2`是结构零**（1.07—8.83e-15，**不随h下降**，另有一条门专判它没在下降）；**易/难轴互换**——参考`d1`转90°挠度比实测999.99972 vs 1000（偏差2.79e-7，随载荷平方下降＝几何非线性），**同行点名了这个失效模式却没有门守着**；端扭矩`θ=M·L/GJ`偏差4.885e-15、牛顿一步收敛、扭率极差6.85e-17；**Gauss-Bonnet holonomy**——切向`x̂→ŷ→ẑ→x̂`围出三直角球面三角形，面积`4π/8=π/2`，重输运后`γ₃−γ₀`与`−π/2`**逐位相同**。**最后一条判据期望的是错误答案**（不重输运时扭转能量恒为`0.0`，零容差）——那是「抄了公式不抄retransport外层循环」的分辨力本身。顺带量到固支半格柔度的**第三例**（不订正只有一阶，`3h/2`订正后二阶3.876/3.939/3.968） | B | local_batch | 4条 | `tests/cases/test_anisotropic_rod_twist.py` |

| [`groove_sweep_wall`](groove_sweep_wall/case.md) | **0074第六节阶段一（轨A）：带材真正的接触伙伴是槽，不是三角网格**。沿真实中心线的两段外倾锥面，逐壁单边。三条判据方向各不相同：① **逐位退化**——直中心线＋`tanα=0`时与`PenaltyAnnulusLimit`的能量/梯度/Hessian`float.hex()`**逐位相同**（承重的是`halfwidth − (lateral + radius)`那个括号位置，挪走当场红）；② **锥面≠平面，而plans/15第2.2条那句话缺一个前提**——深度钉住时两者横向力`k|g|`**逐位相同**，差别全在锥面多出的举升分量`k|g|tanα`上（合力之比恒为`secα=1.0154266`，三档横移同一个数）；深度自由＋压紧3N时锥面横向力**饱和在`F_hold/tanα`=17.0138455 N、与横移量无关**，平面按`k`线性上升，五档横移比值**14.69→293.88**，逃逸阈值`3.0613646 mm`对闭式（60步二分夹到1e-6）；③ **FD一致性没有收敛阶可报**——活动集内`U`是`x`的二次多项式，中心差分截断项**恒为零**，误差是纯舍入且随`h`变小而**变大**（h=0.1时相对5.6e-14）。**冻结帧的代价被量成一个数**：`∇g_精确 = ∇g_冻结 − A·t`，代入plans/14实测`τ=2.550°/mm`得**丢失5.3707%、力方向偏3.0743°**，闭式与数值中心差分两条腿互不引用、实测二阶收敛比4.0000。**注错抓到一道空门**（`max(v,0)`那条改法36门全绿，已补`test_the_wall_does_not_hinge_at_the_groove_floor`）。`α=10°`是声明参数不是实测、本页未读真实CSV、摩擦未进——四条写在案例页第四节 | A | interactive | 10条 | `tests/cases/test_groove_sweep_wall.py` |

| [`span_disturbance_channels`](span_disturbance_channels/case.md) | **第一条建"扰动从哪儿进来"的**：臂动经落位点几何变成收线端速度、人手横向触碰让直线段变折线。**正面否掉一条错误因果**——0066说"真机上跨长逐样点变"，而plans/14第3.3节订正后的场景里自由跨两端都是世界系固定的，`L_geo`是常数；臂动改的是落位点几何不是跨长（0071第二节）。机制`σ' = Ω_切向/κ`，于是**平面圆退化档必然为零**（κ恒定⟹送带率恒定），实测`c=0`档20000步2.2752e-11 N、`c=50`档1.1376e-11 N——**这一条最初写的是零容差，实测当场否掉**（病根是`T→L_mat→T`往返的几个ulp，不是扰动）。单调判据在同一条槽上扫Ω=1/2/4/8 rad/s，幅值比2.0295/2.1123/2.3785，对正弦响应闭式`K·a·√(4ζ²ω_n²+ω²)/√((ω_n²−ω²)²+4ζ²ω_n²ω²)`偏差−4.95e-6→−2.11e-4；**零频退回稳态关系`c/R²`逐位相同**。触碰那条：路径增量是几何恒等式（小角度式在δ=4 mm处差1.777e-4）、**中点是最软的地方**（比值0.75，直觉是反的）、尖峰`EA·p/L_mat`**在四档dt上给同一个偏差1.70e-13**（恒等式的指纹）、**横向冲量与材料长度账之间有一条恒等式**（残差8.29e-16 N·s，`−c·dt`那一项是半隐式的指纹）。**反向那条**：撤掉控制器时尖峰**不回落**——四个阻尼周期后还剩71.8%，对闭式偏差1e-8量级。**闭环那一半没做**（归轨道E）。永久`hypothesis_only` | B | local_batch | 7条 | `tests/cases/test_span_disturbance_channels.py` |

| [`real_centerline_invariants`](real_centerline_invariants/case.md) | **本仓第一条读真实工件的案例**——36个案例此前全是解析构造的螺旋线、平面圆与直链。兑现plans/15阶段二2.1与2.3的**前半句**。三档方向各不相同：① **合成圆弧＋自转的帧**：`κ_s = sin(τa)/R`、`κ_n = −cos(τa)/R`、`τ_frame = −τ`四条闭式被站点差分逐条取回（二阶，比值4.000），另判恒等式`κ_s²+κ_n²=κ_total²`（**它对'两根轴被对调'有分辨力而单点没有**）与`τ`的**符号**（符号错了接触力往反方向偏，而三根轴仍然正交——那是个安静的错）；② **一段之内的扭转尖峰**：`forward`给`sinΔ/h`、`central`给`sinΔ/(2h)`，**峰值比恰为1/2（零容差）而`∫|τ|ds`两者逐位相同**——峰值统计依赖差分取法、积分统计不依赖，`sinΔ`在比值里整个约掉；③ **真语料（`PE_REAL_CENTERLINE_CSV`选择进入，0073裁决真实资产永不进仓）**：5份GCW导出（426—984站、850—1966 mm）与plans/14第2.2节那张表逐档对拍。**第一次读就读出一条对不上的**——`v1-coil-1`实测`τ_max` 6.5686而表里6.648、`v2-coil-02`实测6.6467而表里6.568，**互换之后两行都对到四位有效数字**；其余三行与这两行的`R_min`/`ε_edge` p100/两个占比全部原位对上。本页把它钉成两条判据（对不上的恰为2行、且两行互换后对上）而不是一句散文——**算错不会让两个错数恰好互为对方**。另两条实测：两种差分取法在真语料上`τ_max`差19%—25%而`∫|τ|ds`差<1%（**那些峰在2 mm采样下没收敛**），以及那批工件自己的槽宽是8/10 mm两档、不是表里假设的4 mm（换过去超标占比从0%—5.9%变成13.4%—23.3%）。**九条已知失效写在案例页第四节**，含两条自己的错（求值点没落在每个细化步长的站点上、把`τ`的截断项当成了恒等式）与**"扭转吃掉了多少硬弯"一个数都没算**（plans/15第2.3条只兑现了前半句） | A | interactive | 3条 | `tests/cases/test_real_centerline_invariants.py` |

| [`sdf_contact_convergence`](sdf_contact_convergence/case.md) | **0074第六节阶段四（轨B）：把接触的`g`从解析式换成可查询的场**。窄带块稀疏（8³块＋块坐标哈希）＋**三次B样条（C²）**，接触项形制与既有五族一字不改（`U = ½k·g²`，要的只有`g`／`∇g`／`∇²g`三样）。三条判据方向各不相同：① **场对解析SDF的逼近是二阶**（采样值直接当系数、不做预滤波，0085第三节）——球h=1.0/0.5/0.25实测**收敛比**`g` 4.0012/4.0002、`∇g` 4.0008/3.9998、`∇²g` 4.0784/3.9263（**比值4≈二阶，不是四阶**；二阶导带胞内相位抖动，窗口按实测放宽到`[3.8,4.2]`并另配必红）；**半空间是仿射的于是被精确重构**（值误差<1e-14、`∇g`到末位、`∇²g`全零），它与球两档一起才说明"阶来自插值不是来自巧合"；② **C²不是精度偏好而是适用域的硬要求**（0074第二节第4条）——胞界处B样条`|Δ∇g|`随ε线性趋零（2.03e-4→2.03e-7），而**三线性`|Δ∂g/∂x|`恒为5.082e-02、四档一位不动**，这是那句话在本仓的第一个出处；③ **与解析接触项并排**：半空间三档全部差**1 ulp**（2.220e-16 mm），**阶在这条上量不出来**（误差已经是舍入），如实写在门里、量阶去球那条。**本页最要紧的一个数**：场引入的位置误差是罚穿透的**6.3746倍**——这个刚度档上决定位置精度的是分辨率，把`k`提十倍只缩分母；配套地`normal_force_n`那条"力精确、与k无关"在场这一档**不再完整**，实测力误差7.64e-2/1.90e-2/4.73e-3 N、比4.024/4.006，即"与k无关、**与h二阶相关**"。闭式主项`Δz = −h²/(3z*)`与实测最差档差**0.35%**、阶4.0091/4.0023；解析项那一支与闭式**逐位相等**（活动段能量严格二次，牛顿一步，零容差是代数保证的）。**窄带外裁失败关闭**——远在体外与深在体内两种点的信号集合**都是`{None}`**，存下来的字节里没有一个带符号；不做预滤波（插值二阶不是四阶）与"只验过一维轴上平衡、`∇²g`那块没驱动过位置"两条写在案例页第四节 | A | interactive | 11条 | `tests/cases/test_sdf_contact_convergence.py` |

| [`double_slit_propagated`](double_slit_propagated/case.md) | **全仓第一条由传播算出来、而不是把闭式代进去的衍射**（能力位S4.6的定义原文）。走`optics/field`（二维复数场＋基2 Cooley-Tukey）→`optics/propagation`（角谱／菲涅耳／夫琅禾费），**零依赖**。判据三条：① 复振幅对**定义式直接求和**的独立参照（生成器不共用旋转因子表、不import`physics_engine.optics`）偏差**2.7756e-16**；② 条纹极大落在`sinθ = mλ/d`上偏差≤1.7e-18，**第8级缺级实测恰为0.0**（缝宽/缝距取2的幂⟹蝶形末级相位差恰为π，IEEE精确取负）；③ 通量守恒与往返可逆各一条（6.0e-16／1.4016e-16）。**判据自己的参照可能比被验对象更差**：平移定理的预测相位直接写`cmath.exp(−2πi·k·m/N)`时N=1024上残差比值**120.6**，把相位按整数`(k·m) mod N`约化后降到**1.85**——**差65倍**，那120.6里绝大部分是libm的大幅角约化误差，即那条门测的其实不是FFT。**抓到一个自己代码里的真缺陷**：孔径边界落在采样点上时"这个点归谁"由浮点末位定，实测同一段代码在两个`dx`上给出8个与**31个**（期望32）采样，**唯一的迹象是收敛阶变成−4.222**而图样仍是漂亮的sinc²；处理是失败关闭（那个点在几何上确实没有答案），修好后阶回到2.474→2.106→2.026。**圆孔阶梯边的误差是振荡的不是单调的**（Gauss圆问题那一类，"阶"实测0.040与2.892），所以那条门只断"都在上界内"与"加密真的更好"，**假装它是O(h²)会得到一条随机红的门，比没有门更坏**——写在案例页第四节 | C | interactive | 3条 | `tests/cases/test_double_slit_propagated.py` |

| [`spool_winding_growth`](spool_winding_growth/case.md) | **场景⑤第一条案例**（那个场景此前0/6、一位done都没有），只取其中**几何与记账**的那一半：匝数怎么变成半径、喂进来的材料去了哪。**判据强度靠恒等式而不是靠容差**：半径对闭式14档**零容差浮点`==`**；材料守恒`Σ R(k+½) == ∫₀ⁿR ds`五种形制×32匝**真的浮点`==`**；喂料的`喂进来 == 跨距里 + 盘上`1400档**整数`==`**；反解`长度→匝`往返最坏rel **1.3158e-16**（1 ulp）。另有一条**从下方单调逼近2**的收敛阶（同心圆理想化对真阿基米德螺线弧长闭式，七档1.9354→1.9986）。**与既有两处实现的关系被两道逐位门钉住而不是靠注释**：退化档与`drives.radius_mm`给同一个浮点数、台阶式与`modelgen.generate_spool`在冻结那组入参下同为`68.0`——三者不是三份重复，**自变量不同**（`drives`吃匝答力、`generate_spool`吃层答形、本模块补中间那段换算）。**抓到三道永远绿的空门**：① 守恒恒等式**对「两边同时错一个常数」是盲的**（删掉`min`让跨距无条件吃料，`7+(k−1−7)==k−1`照样成立而盘上是负的，整数守恒那道门一条不红）；② **金标与被验量出自同一支笔**——第一版闭式与被验函数逐字符同构，实测偏差**恰为0.0**，那条oracle当时验不了任何东西；③ **拿冻结值对它自己**（螺线弧长原在`expected`里而引擎有意不算它，测试只能把expected喂回`check_all`，**永远绿**），已挪进`inputs`。**数值稳定那一支没验到位**：换成坏的写法只红2处、两处参数都没进严重相消区，**「数值稳定」只有薄带极限才验得动而本批没扫到**——如实写在案例页第六节 | A | interactive | 4条 | `tests/cases/test_spool_winding_growth.py` |

| [`mutual_inductance_general`](mutual_inductance_general/case.md) | **一般位形（倾斜／偏心／非共轴）的Neumann双回路细丝离散**，金标走**三条互不引用的独立路径**（约化Neumann单重积分／Biot-Savart圆盘磁通／偶极-偶极式），生成器三路交叉验证后才落盘。**本案例最实的一条不是某个偏差，是"收敛阶不是一个数"**：中点切元求积在解析周期被积函数上是**几何收敛**（每加8段误差降固定倍数，实测log₂比3.9599/3.9399/3.9253/3.9093，比值不衰减、max/min=1.0129），而**折线弦（多边形细丝）离散是代数二阶**（同构型同档实测0.5253/0.4449/0.3854/0.3220，与二阶理论值0.5261/0.4448/0.3853/0.3219逐位吻合）——**两者在N=64处差六个数量级**。另有三条方向不同的：**互易`M(a,b) == M(b,a)`零容差逐位**（且分段数跟着交换，96/61与71/53）；**远场对偶极近似**三族取向各五档倍频程、12个收敛阶逐个钉住（1.98→2.00）而**不断言成2**；**正交零磁通**构型按对称性精确为零，四组分段数实测`|M| ≤ 7.3e-26`。**自感失败关闭、不做任何正则化**——必红门实测：绕开拒跑后同一条回路在N=48/96/192给出三个**都很正常**的数（单调增长、彼此不到2倍），**没有拒跑就没有任何东西看得出那是发散积分** | B | interactive | 8条 | `tests/cases/test_mutual_inductance_general.py` |

| [`narrow_phase_signed_distance`](narrow_phase_signed_distance/case.md) | **0090：网格窄相三选一之后落地的那一条**。EPA与MPR**在前置上就出局**（都要凸体，而0074实测三个collision网格凸性通过率0.15%/3.72%/26.19%，凸分解又已由0073裁掉；MPR还需要一个内点，非凸件上AABB中心可能根本不在体内）——**0052那条「今天裁会是猜」的待裁项按事实关掉**。四类判据：① **既有球/胶囊窄相逐位不变**（120对语料34对相交，`penetration_mm.hex()` **34/34**；挪1 ulp时`pytest.approx`抓不到、`float.hex()`抓得到）；② **闭式对拍**：球-球1.421e-14、球-半空间7.105e-15、盒-盒轴对齐7.105e-15、球-旋转盒外/内1.332e-15 / **0.0**；③ **场的偏差被声明成一个式子而不是一句话**（`(h²/6)·tr(∇²φ)`）：半空间恒**0.0**、凸球对`+h²/(3ρ)`实测**偏差/估计 = 0.999992/1.000000/1.000000**、**凹孔对`−h²/(6ρ)`**（**符号相反**）、非凸环面收敛比4.0459/4.0127；④ **法向与见证点**：抄Bullet那条自洽恒等式`pointOnA == pointOnB + distance·normalBtoA`（残差球胶囊族1.0049e-14、盒对8.1886e-15、**场那条0.0**），梯度对中心差分max<2e-6、模恒为1到末位。**两个独立算出来的东西撞到一起**：见证点偏离真面正好是**1.8116e-3 = h²/(3ρ)**。**四条如实留空**：「网格」两个字一个都没兑现（`MeshAsset`不带几何）、场只接球型探针、一般凸对凸穿透深度仍为零、`sampled_field`进不了接触管线 | A | interactive | 11条 | `tests/cases/test_narrow_phase_signed_distance.py` |

| [`angular_spectrum_propagation`](angular_spectrum_propagation/case.md) | **角谱：不做傍轴近似的那一条传播**。S4.5此前报partial的唯一理由是"角谱那一侧没有任何案例冻结金标"，本案例把它冻下来：`missing`点名的四条（平面波本征函数／倏逝波衰减／`z→0`逐位／半群）**全部进案例**，生成器走**二维直接求和**（无行列分解、无位反转、无旋转因子表，不import`physics_engine.optics`）。**两条是角谱独有、傍轴给不出的**：① 平面波本征值残差3.5253e-15，而精确与傍轴的差逐bin冻住（`λf=0.1978`→3.9161e-4、`0.9888`→**7.1041e-1**、倏逝四bin→7.8580e-1…1.0116）；② **倏逝功率损失就是一个数**——亚波长网格（100nm间距，32格里只有11格传播）4采样缝上，精确角谱**0.895526867531937**、傍轴**恰好1.0**（`|H|=1`逐bin ⟹ Parseval分子分母逐项相同 ⟹ 零容差），**差0.1045就是"傍轴给不出角谱"的量**。另有MFT-FFT等价9.9920e-16、`z→0`与`ifft2(fft2)`**逐位相同**、半群3.1402e-16、能量守恒1.1102e-16 | B | interactive | 4条 | `tests/cases/test_angular_spectrum_propagation.py` |

| [`mie_sphere_scattering`](mie_sphere_scattering/case.md) | **全仓第一条散射案例**（S4.7的`why`原文引plans/05对场景④的判词是"散射零"）。均匀球的**严格级数解不是近似**。**四条oracle的金标来路各不相同**，这是分辨力的来源：**50位十进制独立实现**（上升级数、两阶直接相除的`D_n`、自己的泰勒`sin/cos`——**连libm都不共用**，最坏偏差5.5511e-16）、**解析的0**（幺正性：无吸收球`Q_ext=Q_sca`最坏相对差1.281e-16、六构型中**四个恰为0**；逐阶`||a_n|²−Re a_n|`最坏2.220e-16）、**静电偶极子闭式**（退化到瑞利：只用严格级数验`C_sca ∝ 1/λ⁴`阶1.780/1.952/1.988，另对瑞利闭式阶1.665→1.996，**两条互不依赖**）、**解析标度律**（大球极限不止断"接近2"，**断它按`x^(−2/3)`趋近**：相邻比值0.634430/0.634339/0.633527/0.632662对理论`2^(−2/3)=0.629961`从上方单调趋近）。**截断显式声明＋失败关闭**（`n_max=ceil(x+4x^⅓+2)`，末项占比超申报当场炸；`x∈[1e-6,1e3]`、`|m|≤10`域外拒答不外推）。**抓到教科书算法本身的一个缺陷**：Bohren & Huffman的`BHMIE`把对数导数向下递推起点取`max(n_max,|mx|)+15`，实测x=500、m=1.33给`Q_ext=2.031189119014`而收敛值`2.030373894631`——**相对差4.0e-4，而它自称严格级数解、一个字都不报**（级数照收敛、能量照守恒、幺正性照满足）。**三条不利的如实写在案例页**：本案例只做效率因子、角分布一行没有；**没有对过任何已发表基准表**；有吸收球的逐阶系数无高精度参照 | A | interactive | 4条 | `tests/cases/test_mie_sphere_scattering.py` |

## 一之二、每个案例穿过引擎的哪几层（决策0048第三节通则）

**"验公式"与"验引擎"是两类，混在一个计数里计数就不再有意义。**
本节是那条通则的执行面：**55个案例**按**穿过引擎哪几层**分类，**不按案例数报成绩**。

> **2026-08-18点清**：这个数一度停在33，而目录里当时已经是35——波次三收口时登记的那句"计数由收口时一次点清"**没有被执行**，于是它安静地过期了两波。本次机械对账：目录36个、本表覆盖36个、各行相加36。**这类数字不该靠人记得回来改**——`tools/check_case_pages.py`今天校验的是"案例目录必须出现在本页"，管不到本表的分类与求和；补一道门让它们对不上就红，登记在plans/07。
>
> **2026-08-18第二次点清（波次五，轨A2与轨C2各加一条）**：那道门补上了（`tools/check_case_pages.py::layer_table_problems`，判自称条数／各行相加／逐案例覆盖三条），**而它当场证明了自己有用**。本波两条轨**各加一条案例**（`real_centerline_invariants`与`three_sphere_pyramid_rotational`），三个数同步改成**38**。
>
> **这一格是本波唯一的真同点冲突，而它的形态恰恰是git看不见的那种**：两条轨各自把`36`改成`37`——**文本相同，于是`ort`自动合并、一个冲突标记都没有**，合完之后表头写着37而各行相加已经是38。AGENTS.md末节记的"分槽规则挡得住互改文本，挡不住同点插入"是同一族，但这一次连"保留双方"都不适用：**两边改的是同一个数，正确答案是第三个数**。
>
> **靠人记得回来改是挡不住这个的**——两条轨各自都改对了自己那一步。实测：合并后`tools/check_case_pages.py`当场红（"分层表自称37个案例，实际目录38个"），改完才绿。

| 穿过的层 | 条数 | 案例 |
|---|---|---|
| **`state`→`energies`→`solve`（整条路）** | **12** | `cantilever_self_weight`、`large_deflection_cantilever`、`euler_buckling`、**`incline_slide_threshold`（第一条带接触的）**、**`friction_hysteresis_loop`（第一条改写历史的）**、**`three_sphere_pyramid`（第一条多体接触的）**、**`capstan_tension_ratio`（第一条多槽位同时滑移的）**、**`winding_line_endtoend`（第一条把张力、接触、摩擦、蹭边装在一条链路上的）**、**`roller_skew_lateral_drift`（第一条弯曲＋张力的梁-弦边值问题）**、**`groove_sweep_wall`（第一条沿真实中心线的扫掠槽壁；深度自由那一组真走`solve_equilibrium`）**、**`three_sphere_pyramid_rotational`（第一条转动自由度参与接触的：状态向量里除了节点位置还有转动块，而`solve_equilibrium`一个字没改）**、**`sdf_contact_convergence`（第一条接触的`g`不是解析式而是查出来的：窄带块稀疏＋三次B样条）** |
| **`state`→`sections`→`section_beam`→`energies`→`solve`** | **1** | **`kirchhoff_section_vertex_springback`（第一条WDS运动学全局截面站点与收敛后历史提交）** |
| **`state`→`rod`→`energies`→`solve`＋外层重输运** | **1** | **`anisotropic_rod_twist`（第一条有材料帧的整杆：双轴弯曲＋扭转，且`solve_equilibrium`之上还有一层retransport循环——**它是唯一一条求解入口被包了一层的**）** |
| **`state`→`sections`→局部平衡** | **1** | **`rectangular_section_springback`（第一条截面非线性与逐点材料历史）** |
| `state`→`energies`→`integrate` | **3** | `two_body_spring`、**`bouncing_ball_restitution`（第一条瞬态阻尼接触）**、**`ten_ball_funnel`（第一条10球耗散组合）** |
| `state`→`integrate`（不碰能量装配与求解器） | 3 | `ballistic_free_flight`、`harmonic_oscillator`、`rigid_body_free_flight` |
| `energies`协议层（梯度与Hessian，不求解） | 1 | `axial_stretch_hessian` |
| **`feed`→`winding`的卷绕记账（不碰`state`／`energies`／`solve`）** | **1** | **`spool_winding_growth`（场景⑤第一条）**——它算的是**几何与记账**不是力：匝→层→半径、喂进来的材料去了哪。**本仓无同类**，故新开一行 |
| **`optics/field`→`optics/propagation`的变换层（不碰`state`／`energies`／`solve`）** | **2** | **`double_slit_propagated`（第一条由传播算出来而不是闭式代入的）**、**`angular_spectrum_propagation`（第一条不做傍轴近似的）**——它**不属于下面那行闭式计算器**：那五条一层引擎都不碰，而这一条真走了本仓自己的FFT与传播算子，只是那两层不在力学那条路上 |
| **电磁域的数值求积（不碰引擎的任何一层）** | **1** | **`mutual_inductance_general`**——它**不属于下面那行闭式计算器**：那五条是把闭式代进去，本条是**数值求积**（中点切元＋细丝离散），闭式只作退化档的金标 |
| **`optics/spherical_bessel`→`optics/mie`的级数解（不碰引擎的任何一层）** | **1** | **`mie_sphere_scattering`（全仓第一条散射）**——它**不属于下面那行闭式计算器**：那五条是把一个闭式代进去，本条是**求一条严格级数的和**，且截断阶显式声明、域外拒答 |
| **闭式计算器（不碰引擎的任何一层）** | **5** | `scalar_diffraction_airy`、`fts_instrument_line_shape`、`two_beam_interference`、`mutual_inductance_coaxial`、`norris_thin_strip` |
| 几何查询／资产治理／模型生成（基座，不碰物理求解） | **7** | `segment_distance`、`rotated_aabb`、`broadphase_superset`、`mesh_asset_integrity`、`generator_determinism`、`peer_fcl_distance`、**`narrow_phase_signed_distance`（第一条把距离场当窄相查询的）** |
| **`rigidbody`→`contact_dynamics`→积分（转动进动态接触，不碰`energies`／`solve`）** | **2** | **`rolling_ball_incline`（第一条让转动自由度参与动态接触的）**——状态是刚体13维（质心位置/速度＋体系角速度＋四元数），力与力矩由`contact_dynamics`装配后交给`integrate_free_flight`的两个回调。**这一行是新开的**：本仓此前没有任何案例落在「接触产出几何量 → 力矩装配 → 积分器回调」这条链上。**`box_tipping_threshold`（第一条多接触点对同一平面的）**同住这一行，但它开的是另一半：球只有一个接触点、`r × F_n` 恒为零，**一个球不会翻倒只会滚**；翻倒要的是第二个接触点，于是`contact_dynamics`多出「一组体系支承点」那一档，而判据从「一个力矩」变成「哪几个点还在承载」 |
| **`transport`自带的输运推进（不碰`state`／`energies`／`solve`）** | **1** | **`free_span_tension_step`（第一条张力由速度差生成的；状态是"材料长度＋放线盘转速"两个标量，不进`State`——**这一行是新开的**，本仓此前没有任何案例落在它上面）** |
| **`motion`→`laydown`的纯运动学（不碰`state`／`energies`／`solve`）** | **1** | **`helix_laydown_closure`（第一条把位姿时间线与送带账放在一起判的）**——它**不是**力学案例：`laydown`只回答"这一瞬落位点在槽的哪里、槽标架朝哪、要放多快"，一个自由度都不解。**这一行是波次二收口时补的**：轨道C把它写进了上面的案例索引表却漏了本表，而**两张表漏一张，分类计数就与目录数对不上**（实测31对32） |
| **`transport`＋`drives`的闭环装配（不碰`state`／`energies`／`solve`）** | **1** | **`closed_loop_tension_step`（第一条把控制器接到对象上的）**——状态是"材料长度＋放线盘转速＋离合器扭矩＋控制器积分"四个标量，仍不进`State`。**它与上一行的`free_span_tension_step`是同一条链路的两半**：那里证明扰动是真的，这里回答压不压得住（答案：在真机那一档上**压不住，而且更坏**）。**这一行是新开的**，本仓此前没有任何案例落在"两个模块的装配层"上 |
| **张力测量/读出/动态装配（不碰`state`／`energies`／`solve`）** | **4** | **`tension_measuring_roll_resultant`/`tension_measuring_roll_installation`**把两侧张力变成测力轮gross/tare/net；**`tension_readout_calibration`**把它变成raw/zeroed桥路、ADC与标定显示值；**`tension_wind_unwind_dynamic`**再把完整测量通道接进`ClosedTensionLoop`。四条都不解通用`State`自由度，但验证的是测量物理、采样时钟和收放线动力学，不是孤立闭式计算器；四份金标生成器均不import被验模块 |
| **`model_snapshot`→`planned_motion`→`model_physics`→`Scene`输入/装配（不碰求解器）** | **2** | **`model_motion_physics_binding`**把visual/collision、时间/无时间规划、多track、static/kinematic/dynamic所有权、显式排除和虚拟frame钉成内容寻址输入包；**`model_scene_assembly`**再把已核SHA资产、资产安装偏置、时间运动、虚拟frame和显式接触候选接进既有Scene/CollisionQuery。两条都不计算力；后者只证明0s分离/1s broad-phase重叠 |
| **`model_scene`→`dynamic_body`→`dynamic_contact`→`rigidbody`** | **2** | **`dynamic_model_scene_free_flight`**把命名质量属性、geometry frame内质心、初始世界系线速度/体系角速度装入13维COM刚体状态，用RK4推500步后再反算geometry位姿；**`dynamic_two_body_contact`**进一步让两个dynamic体的显式候选逐子阶段走窄相法向/见证点，装配作用反作用力与两侧体系力矩，在同一26维RK4系统中推进。两条均不证明约束、摩擦历史或长时间守恒 |
| **`laydown`的中心线几何（不碰`motion`／`state`／`energies`／`solve`）** | **1** | **`real_centerline_invariants`（本仓第一条读真实工件的）**——它只回答"这条工件曲线弯了多少、扭了多少"，**一个力都不算、一个自由度都不解**，连`motion`都不碰（上一行的`helix_laydown_closure`要位姿时间线，这一行不要）。**这一行是新开的**：本仓此前36个案例的输入全是解析构造，而这一条的第三档吃的是GCW导出的真实CSV（选择进入）。它的价值不在判据强度而在**位置**——`plans/15`阶段二那条"让它读真实工件"，此前一条案例都没有落在上面 |
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
| 从｜同行C档13条标准案例（research/05第2.3节） | 算得**对不对** | **6/13**。**2026-08-18曾报7/13，当天退回**：0080让第7条进位的依据是research/05对Chrono`utest_DEM_pyramid.cpp`的转述「同一模型只改摩擦系数」，而那句**失真**——原测试是DEM完全动力学、两支各改两件事、自称验的是滚动摩擦、且没有任何μc闭式。**引擎实现是扎实的，站不住的是「进位」这个外部主张。** 逐条重数的口径见0049第十节 |

主分母的逐位机械计数当前为**17/42**，每场景位数`7/5/10/7/6/7`已由0057冻结；
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
