"""physics-engine：多域数字孪生的引擎内核（规范先行，参考实现随消费方需求生长）。

安装（舰队wheelhouse链路，decisions/0010）：

    git clone ts-orangepi:wheelhouse.git ~/wheelhouse   # 每台机器一次
    uv add "physics-engine==0.6.0" --find-links ~/wheelhouse
    # 或 pip install "physics-engine==0.6.0" --find-links ~/wheelhouse

0.6.0是wheelhouse当前最新已发布版；旧版字节永久共存，消费方必须钉精确版本并经
自己的回归门升级，不能因wheelhouse出现新版本而被动漂移。

**0.x语义**：仓库处于快速演进期（模块地图spec/01大半尚空），minor跳变可以
破坏兼容。消费方必须**钉精确版本**并经自己的门禁自觉升级；升级永远不是
被动漂移。每次发版有决策记录。

公开API分两档（档位即本包自己的面状态，轴1规则5）：

**稳定倾向**（对应已冻结轴的参考实现，破坏性改动须决策记录）：

- `physics_engine.facets` —— 面清册与失败关闭读取端（轴1）
- `physics_engine.canonical` —— 声明式规范化JSON读写对（轴3规则2）
- `physics_engine.identity` —— 稳定ID与单位后缀校验（轴2）
- `physics_engine.provenance` —— 溯源机械层：耐久写/no-replace/保护读（轴5）
- `physics_engine.run_package` —— 装配、语义复读、生命周期法则（轴4/5）

**实验档**（随spec/10、11、12、14草案演进，minor内可破坏）：

- `physics_engine.shapes` —— 模拟形状声明层（spec/11）
- `physics_engine.collision` —— broad phase + 球/胶囊族narrow phase（spec/10）
- `physics_engine.scene` —— 场景文件与数据层入口
- `physics_engine.geometry` —— 体积/质心/惯量张量与rounded-core SDF（spec/12）
- `physics_engine.materials` —— 材料记录：多域字段+证据分级+单位边界（spec/14）
- `physics_engine.oracles` —— oracle清单面：expected/tolerances/双哈希（轴7）
- `physics_engine.state` —— 状态形制与打包次序契约（spec/12）
- `physics_engine.integrate` —— 时间推进：四个平动积分器与耗散累计（spec/12）
- `physics_engine.energies` —— 能量项四方法协议与五个能量项（spec/12第三节）
- `physics_engine.solve` —— 准静态平衡：牛顿+回溯线搜索+带状求解（spec/12第4.1节）
- `physics_engine.rigidbody` —— 刚体姿态与自由飞行（力学域，spec/12）
- `physics_engine.contact` —— 接触：锚点布局、罚法向（半空间/球-球）、粘着弹簧、
  库仑return-map、线性法向dashpot、准静态步进器（力学域，决策0050、0055）
- `physics_engine.contact_pipeline` —— 动态球-球接触整合层：场景候选、碰撞窄相、
  罚势与dashpot响应（力学域，决策0058）
- `physics_engine.sections` —— 矩形纤维截面：逐点一维弹塑性本构、显式材料历史、
  N/M求积与局部回弹平衡（力学域，决策0059；**不是独立截面场自由度**）
- `physics_engine.section_beam` —— 一个WDS式三节点站点的easy-axis纤维弯曲装配：
  节点/边扭角运动学→全局残差/Hessian→收敛后历史提交（力学域，决策0060；
  0061有WDS默认关闭的线弹性单站候选，**不是整杆或正式迁移完成**）
- `physics_engine.rod` —— 整杆各向异性弯曲＋扭转：平行输运材料帧、逐顶点双轴刚度
  （`EI_easy`配`κ1`、`EI_hard`配`κ2`，配反即失败关闭）、边扭角与**retransport外层
  循环**（力学域，决策0065；**没有塑性、没有轴向、没有槽壁接触**——弹性双轴弯曲
  与扭转两项而已）。梯度/Hessian走`physics_engine.autodiff`的零依赖一/二阶jet，
  那是从`section_beam`提升上来的共享面（决策0064第4.1节）
- `physics_engine.feed` —— 喂料前沿：节点预算定死、布局定长，已喂/未喂是向量里的值
  （力学域，决策0062；**不做材料注入**——那是带材从轮面流过，与往前接长度不是一回事）
- `physics_engine.motion` —— 位姿时间线（spec/10 `MotionSource`）
- `physics_engine.actuators` —— 驱动器声明层（spec/10 `Actuator`；**`apply`的物理未实现**——
  物理在`drives`，理由是`actuators`在基座、基座不依赖物理域，见决策0062）
- `physics_engine.drives` —— 张力驱动链：磁粉离合器电流→扭矩、卷径换算、理想PID
  与闭环推进（力学域，决策0062；**不复现ATC600**，真机回路是拿不到参数的黑箱）
- `physics_engine.sensors` —— 传感器声明层（spec/10 `Sensor`）
- `physics_engine.modelgen` —— 参数化模型生成器（spec/11）
- `physics_engine.cli` —— `pe-scene`命令行的实现（数据层入口；
  **它是被`[project.scripts]`暴露的真公开面**，此前长期不在本清单里）
- `physics_engine.optics` —— 光学域：干涉/衍射/FTS仪器线型的闭式解（spec/15）
- `physics_engine.electromagnetics` —— 电磁域：互感与超导薄带的闭式解（spec/15）

**能力边界（诚实条款）**：本段回答"有什么"，**回答不了"能不能算你要算的"**——
后者按用户六场景逐条写在`README.md`同名小节与`docs/plans/04`，
**今天是0/6端到端**（同行C档13条标准案例6/13，两条分母必须并排报，见0048第二节）。

`integrate`能推进二阶系统并用独立入口累计声明的物理耗散；`rigidbody`能推姿态与自由飞行
（六个积分器`production_ready`全为`False`）。能量项有均匀重力、轴向拉伸、
小挠度弯曲、几何精确（DER）弯曲、点载荷；准静态平衡可解并可判是不是极小。
光学与电磁**全是闭式解**——没有网格、没有复数场、没有FFT。

**接触有了，但要说清有到哪**（决策0050、0055）：罚法向（半空间与球-球）、
粘着弹簧、库仑return-map、线性法向dashpot、准静态步进器；**多体接触成立**
（球-球两端都是自由度）；锚点是**真历史**、写回状态。欠阻尼与过阻尼的单次碰撞、
以及10球平面漏斗最小组合已走真实时间推进与耗散账。
十球案例的球-球响应已由场景候选经过broad/narrow动态检测驱动；解析平面仍直接
进入既有接触项。**仍缺**：网格窄相、空间索引、载荷控制失败后的真回退、
接触体的**转动自由度**（今天是质点+半径，**没有力矩**，故多点接触无意义）。

**截面积分点有了两片，但边界要说清**（决策0059—0061）：矩形截面可从轴向应变与曲率
得到逐点弹塑性应力，逐点塑性历史显式进`State`，并积分出轴力/弯矩、反解自由回弹。
其中easy-axis曲率已由一个WDS式三节点/两边扭角站点生成，弯矩与一致切线进入全局
`EnergyRegistry`/Newton；WDS另有默认关闭、准静态线弹性的本地单站采纳候选。0.6.0
已经发布，但消费方候选在合入main并过其正式门之前仍不算迁移完成；轴向N也未进全局装配。
积分点是本构/求积点，**不增加全局运动学未知量**；它们也不是S1.2所需的独立电磁场
自由度。尚无整杆多站、hard-axis、截面翘曲、压扁、剪切/扭转或二维任意截面。

**扭转与整杆各向异性弯曲有了，但只在`rod`那一片**（决策0065）：平行输运材料帧、
逐顶点双轴弹性弯曲、边扭角扭转、以及retransport外层循环，三道闭式门
（螺旋线运动学／易难轴互换／端扭矩`θ = ML/GJ`）各自有必红用例。
**它与`sections`/`section_beam`那一片今天没有接口**——`rod`是纯弹性的，
没有纤维塑性、没有轴向`N`、没有槽壁接触；`section_beam`那一片仍只有一个站点。
两片合并要另走决策。

**没有隐式时间积分族、没有约束、没有跨域耦合
（`couplings/`目录不存在）、没有场求解。** 结构性缺的四条（体积与厚度、
接触、耦合、历史）见`docs/plans/06`，**其中"历史"已由0050解、"接触"已开工**；
路线见`docs/plans/02`与`docs/plans/08`。

**加速档**：`pip install 'physics-engine[accel]'`装NumPy后走加速实现；
核心永不要求它（0014零设施承诺），两实现逐字节对拍是进仓门（0016甲案）。

**实验档模块不进顶层re-export，按全路径import**（`from physics_engine.geometry
import mass_properties`）。这不是疏漏是预算纪律：顶层eager import的模块数是
冷启动延迟的结构代理，`tests/perf/test_budgets.py`守着它；把三个实验档模块拉进
顶层要多付5个模块的import成本，而它们的调用方都明确知道自己要什么。
稳定倾向档才享有顶层便利——档位在这里是有价格的。

完整面见各模块`__all__`。
"""

from physics_engine.canonical import (
    FTS_PROFILE,
    WDS_PROFILE,
    CanonicalError,
    CanonicalProfile,
    canonical_bytes,
    canonical_file_bytes,
    canonical_sha256,
    strict_loads,
)
from physics_engine.facets import (
    Facet,
    FacetError,
    FacetRegistry,
    FacetStatus,
    parse_version,
)
from physics_engine.identity import (
    BASE_UNIT_SUFFIXES,
    CaseIdentity,
    IdentityError,
    assert_quantity_fields_have_units,
    has_unit_suffix,
    parse_case_identity,
    parse_namespace_id,
)
from physics_engine.provenance import (
    ProvenanceError,
    read_protected_file,
    rename_directory_noreplace,
    verified_bytes_snapshot,
    write_durable_exclusive,
)
from physics_engine.run_package import (
    assert_lifecycle_fail_closed,
    publish_package,
    read_verified_package,
)

__version__ = "0.7.0"

__all__ = [
    "BASE_UNIT_SUFFIXES",
    "FTS_PROFILE",
    "WDS_PROFILE",
    "CanonicalError",
    "CanonicalProfile",
    "CaseIdentity",
    "Facet",
    "FacetError",
    "FacetRegistry",
    "FacetStatus",
    "IdentityError",
    "ProvenanceError",
    "__version__",
    "assert_lifecycle_fail_closed",
    "assert_quantity_fields_have_units",
    "canonical_bytes",
    "canonical_file_bytes",
    "canonical_sha256",
    "has_unit_suffix",
    "parse_case_identity",
    "parse_namespace_id",
    "parse_version",
    "publish_package",
    "read_protected_file",
    "read_verified_package",
    "rename_directory_noreplace",
    "strict_loads",
    "verified_bytes_snapshot",
    "write_durable_exclusive",
]
