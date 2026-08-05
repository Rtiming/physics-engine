# 0038 MotionSource与Actuator接口（2026-08-05）

状态：已裁决并落地。spec/10冻结前提的五接口里，`Scene`与`CollisionQuery`有实现、
`Sensor`有声明层（0030）、`Episode`已摘出（0034），**只剩`MotionSource`与`Actuator`
是全空**。本批把这两条补上，落地形态是`src/physics_engine/motion.py`与
`src/physics_engine/actuators.py`，加`tests/test_motion.py`（90条）与
`tests/test_actuators.py`（73条）。

开工前提照0001三前提逐条对：轴已冻结的部分（轴2命名、轴3规则5）走冻结面；
spec/10自己未冻结，所以两个模块按0003纪律**只做声明、校验与纯运动学/时延机械**;
真实消费方两个都在——`MotionSource`是WII的`wii_motion_timeline`位姿时间线，
`Actuator`是WDS张力机放线端（其design/24用途4，`dynamic/end_driver.py`）。
既有产物字节不变（本批不改任何既有模块）。

## 一、两个接口的公开面

`motion.py`（`__all__`19项）：`Pose`（毫米平移+xyzw单位四元数）、`PoseSample`、
`PauseInterval`、`InterpolationSemantics`、`MotionSource`（spec/10三方法的
`runtime_checkable` Protocol）、两个实现`SampledPoseTimeline`与`AnalyticPose`、
门函数`assert_replayable_for_fingerprint`、`MotionError`，
外加六个白名单/常量（`TRANSLATION_INTERPOLATIONS`、`ROTATION_INTERPOLATIONS`、
`ROTATION_ARCS`、`PAUSE_HOLDS`、`EXTRAPOLATIONS`、`ACCEPTED_TRANSLATION_UNITS`）
与三个有名字的边界（`QUATERNION_NORM_ABS_TOL`、`SMALL_ANGLE_ONE_MINUS_COS`、
`POSE_TRANSLATION_UNIT`/`MILLIMETRES_PER_METRE`）。

`actuators.py`（`__all__`10项）：`CommandChannel`、`ActuatorDeclaration`、
`ActuationCommand`、`ActuationResult`、`ActuationDelayLine`、`ActuatorError`,
外加`REALIZABLE_ACTUATOR_KINDS`、`DELAY_QUANTIZATIONS`、`DELAY_STEP_MATCH_ULPS`、
`MAX_DELAY_STEPS`。

**两处不对称，都是有意的**：

- `pose_at`**实现了**，`apply`**没实现**。理由不是难易：`pose_at`只是把已声明的
  样点按已声明的语义取值，不产生新物理；`apply`要把命令变成力/力矩，那要接
  能量项与求解器，是另一块，在spec/10冻结前写进来就是替它拍板。
- `command_space`**缺席**，与`sensors.py`缺`observation_space`同一条理由：
  返回类型`Space`在spec/10里只有名字没有定义。本层给的是`CommandChannel`
  与声明期常数`command_dimension`。有一条门断言`apply`/`command_space`
  在任何公开类上都不存在。

## 二、插值语义：五条，逐条必须由声明者给出

**这是本批最重要的设计判断。** WII发布的`wii_motion_timeline`只给离散样点
（`sample_times_s`、`translation_parent_from_child_mm`、
`rotation_parent_from_child_xyzw`、显式的`pause_intervals`）。
**样点之间发生了什么，artifact里没有这个信息**——它在声明者的脑子里。
库替他挑一个"合理默认"的后果是确定的：两个调用方拿同一份样点算出不同的位姿、
不同的接触时刻、不同的物理，而两边都以为自己是对的。

五条各自的理由（都不是可有可无的补充说明，是**换一个取值就换一份物理**）：

| 语义 | 取值 | 为什么它必须被声明 |
|---|---|---|
| `translation_interpolation` | `linear` / `hold_previous` | 线性给分段常速度，保持给冲击。WII自己算twist时用的是分段线性（其`_linear_interval_rates`），而一个回放控制器设定点的消费方要的是保持 |
| `rotation_interpolation` | `slerp` / `nlerp` / `hold_previous` | slerp给等角速度，nlerp便宜但角速度不均匀 |
| `rotation_arc` | `shortest` / `as_declared` / `not_applicable` | `q`与`−q`是同一个旋转，插值却走**相反的两条弧**。WII在发布时按规范符号归一化四元数（其`_canonical_quaternion_xyzw`），**那不是取短弧**——它会把一段本该走长弧的运动翻成短弧 |
| `pause_hold` | `hold_interval_start` / `interpolate_through` | WII的时序模型明写零运动段"没有可推断的时长，必须用显式时间戳加`pause_intervals`编码"，于是暂停区间是一等公民 |
| `extrapolation` | `reject` / `clamp_to_endpoint` | 样点之外怎么办 |

三条附带裁决：

1. **`pause_hold`即使今天一个暂停区间都没有也必须声明。** 有没有暂停区间是
   **数据**的事，暂停区间怎么取值是**声明者**的事；今天没有不等于明天喂进来的
   那条时间线也没有，而那时语义不能因此退回"由库来猜"。
2. **不提供线性外推。** 外推是在没有任何样点支持的地方发明运动。白名单失败关闭，
   要加取值改`motion.py`并补测试。
3. **`AnalyticPose`只需要五条里的一条**（`extrapolation`）。另外四条问的都是
   "样点之间怎么办"，而解析轨迹没有样点。但定义域之外仍要声明——`horizon_s()`
   是一条被声明的边界。

实测证据（三条语义各自"换一个取值就换一份答案"，都在测试里）：
同一份样点上`pause_hold`两个取值在`t=1.25`给出`(100,0,0)`与`(100,12.5,0)`；
`translation_interpolation`两个取值在`t=0.5`给出`(50,0,0)`与`(0,0,0)`；
`rotation_arc`两个取值给出z分量符号相反的四元数。

**写测试时栽的一跤，记在这里**：第一版用`u=0.5`去验slerp与nlerp的差别，
实测差1.1e-16——因为**半程处nlerp就是两端的归一化平均，与slerp恰好重合**。
那条测试永远是假绿的。改用`u=0.25`后差别是可见的物理差（22.5°对约21°）。

## 三、时延不是`dt_s`整数倍时的语义：默认拒，唯一放行方式是显式声明取整方向

环形缓冲只表达得了`dt_s`的整数倍。于是`delay_s`是**声明**，
`realized_delay_s = steps × dt_s`是**实际**，两者之差`quantization_residual_s`
必须被算出来并如实带在对象上。非整数倍时：

- `quantization="exact"`（默认口径）：要求是整数倍，容差`DELAY_STEP_MATCH_ULPS`
  个ULP；不满足**当场炸**，消息里带残差与两条出路。
- `quantization="ceil_to_step"`：向**上**取整，`quantization_residual_s > 0`带出来。

**为什么只有向上、没有向下也没有就近**：欠时延是spec/10第三节点名的危险方向
（"没有时延训练出的策略上真机会翻车"），`floor`与`round`都可能欠时延,
而`ceil`最多多延不到一步。**多延是保守方向，少延是事故方向，两者不对称**,
所以白名单里只有保守的那一个。

**为什么不干脆默认`ceil`**：那样时延还是被悄悄改了，只是改的方向好看一点。
默认拒逼声明者做一次选择——**取整方向是一条声明，不是一个实现细节**。

两处实现细节值得记：

- `ceil`在"其实已经是整数倍、只差几个ULP"时会多给一步（实测：
  `delay_s = 13 × 0.0001`的商是13.000000000000002，裸`ceil`给14），
  所以`ceil_to_step`分支要用同一个ULP判据把那一步收回来。
- `ActuationResult.realized_delay_s`定义为`steps × dt_s`，**不是**
  `effective_at_s − issued_at_s`。那个减法在浮点下不恒等于前者，**而且随步号变**
  （实测：`dt_s=0.1`、`steps=7`，第2步的减法给0.7，`7 × 0.1`给0.7000000000000001）。
  一个随步号抖动的"实际时延"不是时延。

## 四、在途命令是真历史，所以时延线是值不是可变对象

spec/12规则1要求"内核是无状态纯函数……一切随时间变的量进显式状态"，
第2.2节把"真历史"与"求解器便利"分开、前者必须进状态并随状态被复现。
**缓冲里那几条在途命令是真历史**：丢了它们，同一个初始状态重跑会得到不同的轨迹。

所以`ActuationDelayLine`是**冻结的值**，`advance()`返回`(新的线, 结果)`。
代价如实说：每步产生一个新的元组（深度为`steps`），今天不是热点；
真成为热点时按spec/13义务1（profile先行）再评估，**那时才动形制**。
收益是一个真实的性质：同一条线可以分叉两次跑，两支互不影响（有测试守着）。

`initial_command`**必须显式给出**：缓冲在第0步之前就是满的，那是"本次运行开始
之前这台执行器在做什么"。默认填零等于替声明者断言"它当时闲着"。
前`steps`步的结果带`from_initial_fill=True`且`issued_at_s`为负——
那些命令确实早于本次运行，负值是实话。

## 五、`is_replayable`与轴3规则5的联动门

规范原话："`is_replayable`与轴3规则5（复现指纹）联动：不可重放来源的运行不得
声称指纹。"执行体是`assert_replayable_for_fingerprint(sources, *, run_label)`,
要声称指纹的运行在算指纹之前调它，一票否决。三种拒法：

1. 传进来的东西根本不是`MotionSource`（连"我可不可重放"都答不上）;
2. `is_replayable()`返回的不是真正的`bool`——返回`1`或`"yes"`是在**回避**问题,
   按失败关闭处理（与轴2规则5"留空装有"同一种病）;
3. 返回`False`——那就是本条要挡的那一类。

两个实现各自的可重放判定：

- `SampledPoseTimeline.is_replayable()`**恒真**，理由要说清楚：可重放性说的是
  **这个来源对象**能不能重放，不是那些数字当初怎么来的。样点一旦在手，它就是一张
  有限的表，配上已声明的确定性语义，同一个`t_s`永远给同一个位姿——哪怕那些数字
  来自一次一次性的实机采集。
- `AnalyticPose`拿到的是**任意可调用对象**，它可以读时钟、读随机数、读文件。
  **库证明不了一个函数是纯的**，所以这里做的是**证伪**：`replayable=True`时
  构造期在`replay_probe_times_s`上各求值两次，逐字节不同即当场拒。
  声称可重放却不给探测时刻，也拒——一条没有配证伪尝试的声明就是冒充。

**这道门只会拒，不会证。** 它挡不住一个谎称`True`的来源；今天唯一的证伪手段就是
上面那次双求值，而它抓不住"每逢质数秒返回别的值"那种。**写在这里,
比让读者以为过了门就等于指纹可信诚实。**

## 六、环归属：两个模块都归基座`scene`圈

登记在`tests/governance/test_domain_isolation.py`的`SUBSTRATE_RINGS["scene"]`,
spec/15第二节的表同步一行。

**判据是0035那条：import决定环，不是愿望决定环。** 两个模块的包内import
**只有`identity`**（基座contracts）：

- `motion`不import任何力学模块，因为spec/12第2.3节明写"位姿来源不是状态"——
  `MotionSource`喂进来的位姿时间线是外部驱动，不是被求解的自由度。它是纯运动学。
- `actuators`不import`state`，因为`apply`的物理没落地——今天它只做声明、校验
  与时延机械，够不着状态数组。

选`scene`圈而不是新开一个圈：`scene`圈就是spec/10那一页的内核接口面
（`scene.py`是其第一节，`motion`/`actuators`是第二、三节）。

**与`sensors`归力学并不矛盾，恰恰是同一条判据在起作用**：`sensors`import `state`
来判"这一路通道是不是在读一个自由度"，于是只能归力学（0035已裁）。
同一页的三个接口分在两个圈，是因为它们**实际伸手够到的东西**不同。

**触发条件写明**：`actuators`的`apply`物理落地那天它会import `state`，
届时域隔离门第③条（基座不依赖物理域）当场红。那时两条路——整体改归力学，
或按"声明层留在`scene` / 物理半边进力学"分裂——**两条都走决策记录**。
门有一个好性质：这件事无法悄悄发生。

## 七、验红记录（实测输出）

24条头条红例，逐条实跑（脚本产出，节选首行）：

```
== motion.py ==
[RED] 插值语义缺一（pause_hold=None）
      pause_hold must be declared explicitly — 插值语义是声明者的事，不是库替他猜的
[RED] 时间线以米声明（1000倍单位门）
      motion/x: translation_unit must be one of ['mm'], got 'm' — 以米声明的时间线会整整差1000倍
[RED] 样点时间不从run_start=0起
      motion/x: sample times must start at run_start=0, got 0.5
[RED] 反号样点 + rotation_arc='as_declared'
      两个相邻样点的四元数接近反号（q1 ≈ −q0）……转轴不唯一 (dot=-1.0)
[RED] 非纯函数声称可重放（构造期双求值证伪）
      motion/n: pose_fn gave two different poses at t_s=0.5
[RED] 不可重放来源进要指纹的运行（轴3规则5联动门）
      run/nightly: motion source motion/teleop is not replayable, so this run must not
      claim a reproduction fingerprint (spec/10第二节 × 轴3规则5)
[RED] is_replayable返回1而不是bool（回避）
      run/nightly: motion/dodger.is_replayable() returned 1, not a bool

== actuators.py ==
[RED] 零时延不给理由
      actuator/payout: delay_s=0 needs an explicit zero_delay_rationale
[RED] 非零时延却给理由
      actuator/payout: zero_delay_rationale is only meaningful when delay_s=0
[RED] delay_s不是dt_s整数倍（默认exact）
      actuator/payout: delay_s=0.004 is not an integer multiple of dt_s=0.0003
      (closest is 13 steps = 0.0039, residual 0.00010000000000000026s)
[RED] quantization='round'（白名单里没有它）
      quantization must be one of ['ceil_to_step', 'exact'] — 欠时延是规范点名的危险方向
[RED] 把4ms写成4s（单位门 = 缓冲深度上限）
      delay_s=4.0 at dt_s=1e-05 needs 4e+05 buffer steps, past MAX_DELAY_STEPS=100000
[RED] 命令越出声明界限（不夹取）
      command scalar 0 is 9000.0, outside the declared range [-8000.0, 8000.0]
[RED] 直接构造一个欠时延的时延线
      realized_delay_s=0.003 is shorter than the declared delay_s=0.004
```

其余红例覆盖：插值语义白名单外取值（5条）、弧向与旋转插值不相容（两个方向）、
样点非严格递增、单样点时间线、暂停区间重叠/无理由/区间非法、四元数非单位、
越界查询、查询时刻是NaN/Inf/字符串、`replayable`不是真bool、探测时刻越界、
horizon非正/非有限、`pose_fn`不返回`Pose`、ID不带命名空间、语义参数类型错、
未知驱动器种类、命令量无单位后缀、界限非法、界限长度不符、命令维数不符、
命令发错驱动器、命令含NaN、空通道表、重复通道、`dt_s`非正、
`realized_delay_s`与`steps × dt_s`不符、`pending`长度与深度不符。

### 反证（12条，证明红是那一条红的）

形制照0030：把某一条校验换成空操作（等价于写成`if False`），**同一个声明当场变绿**。
一律用`monkeypatch`，用例结束自动还原。

覆盖：插值语义白名单、弧向相容性、单位门、样点时间规则、暂停区间范围、
单位四元数、`AnalyticPose`的确定性双求值、指纹联动门、显式时延规则、
整数倍规则（两条，见下）、命令界限校验。

**反证翻出了一件本来不知道的事。** 原计划"关掉`_require_integer_step_delay`
就该变绿"，实测**没有**：`round(0.004 / 0.0003) = 13`给出0.0039，比声明的0.004短,
于是`ActuationDelayLine.__post_init__`里那条"只许多延不许少延"的门接住了它。
两条门互相独立——这一步把它证出来了，也说明**只写"必须红"而不写反证，
会看不见一条门其实被另一条门遮着**。现在两条各有一个反证：
只关第一条仍红（且是第二条的消息），两条一起关才变绿，
而变绿的那个**时延悄悄短了0.1ms、残差为负**——这正是四舍五入被禁的样子。

## 八、容差是算出来的（两个数，各自有推导与守护测试）

**`SMALL_ANGLE_ONE_MINUS_COS = 5.0e-7`**（slerp退化到nlerp的阈值，判据是`1 − |cos θ|`）。
逐角度实测slerp与nlerp的最大**分量**分歧（不用夹角度量——`acos`在`dot≈1`处
自损一半有效位，会把测量压在4.2e-8的地板上）：

| θ (rad) | 1 − cos θ | max‖slerp − nlerp‖∞ |
|---|---|---|
| 1e-2 | 5.0e-5 | 1.95e-9 ← 已越过单位四元数容差 |
| 3e-3 | 4.5e-6 | 5.27e-11 |
| 1e-3 | 5.0e-7 | 1.95e-12 ← **取这一档** |
| 3e-4 | 4.5e-8 | 5.27e-14 |

取θ=1e-3那一档：分歧1.95e-12比`QUATERNION_NORM_ABS_TOL = 1e-9`小约500倍,
在这以下两条路给出的四元数**在本仓的单位容差下不可分辨**；
再放宽一个数量级就越过容差了，所以不能更松。守护测试实测阈值处的分歧
低于容差、十倍阈值角处高于容差——**两侧都验**。

**`DELAY_STEP_MATCH_ULPS = 4`**（"`delay_s`是`dt_s`整数倍"的判据容差）。
对1e-5s到0.1s的常用步长（含1/60、1/120、1/240）与1..10000步的组合逐一实测，
两端各取：**真整数倍对的最大ULP距离是1.000**；把`delay_s`相对偏移1e-13
（远小于任何物理意义）之后，**最小ULP距离已是450**。取4给真整数倍留4倍余量,
同时比最近的假整数倍小两个数量级——两侧都不是紧的。守护测试把这两头都重算一遍。

**`QUATERNION_NORM_ABS_TOL = 1.0e-9`**不是新算的，是**与`shapes.PosedBody`对齐**
（其`abs_tol=1.0e-9`）。有一条门盯着两者一致：两处判"这是不是单位四元数"
必须给同一个答案，否则一个位姿在`motion`里合法、装进`PosedBody`却被拒。

## 九、明确的边界（如实登记，不许被"过了校验"盖过）

- **指纹门只会拒，不会证**（第五节已述）。
- **`AnalyticPose`的双求值是必要条件不是充分条件**：它抓得住读时钟/读随机数,
  抓不住"每逢质数秒返回别的值"。
- **`MAX_DELAY_STEPS`是资源门兼单位门，不是物理判据**。它挡的是
  "把毫秒当秒写"这一类；一个1秒时延的回路它拦不住，也不该拦。
- **`command_space`与`apply`都不存在**，`ActuationResult`**不带力**。
  有门断言这两件事。
- **`REALIZABLE_ACTUATOR_KINDS`是白名单不是穷举表**，作用与`sensors.py`的
  仪器表相同：逼声明者说出真机上那台东西叫什么。
- **`MotionSource` Protocol只有spec/10写的三个方法**，不加`source_id`——
  往未冻结的接口上加字段等于替它拍板。门函数用`getattr`取它，取不到用下标。

## 十、面与预算

**不需要新的面**（轴1规则1）。两个模块都不落盘、不跨边界。
位姿时间线哪天要写进场景文件或run package，那时才需要
`physics_motion_timeline`；在途命令哪天要随状态进run package，
那时才需要`physics_actuation_delay_line`。**两者都要先去`engine_facets.py`
登记再写字节——那个文件是闸门**（0017的教训）。

新公开名**只进各自模块的`__all__`，不进`src/physics_engine/__init__.py`**
（spec/15规则4 + 预算纪律）。实测`import physics_engine`的eager模块数仍是97,
一个没多。有两条门守着这一点。

**`source_bytes`要报警**：本批新增55358字节（`motion.py` 29636 +
`actuators.py` 25722），总量从236624升到291982，上限294912,
**占用99.0%、余量仅2930字节**。门今天仍是绿的（`accept.py full` overall=PASS），
但这个余量与0019当时"余量仅1224字节"是同一种没有意义的绿:
**下一个进仓的模块必破门**。按0018自己写的那条路，抬上限要在
`benchmarks/engine_budgets.baseline.json`的`source_bytes_ceiling_history`
记一行并引本决策——本批不动`benchmarks/`（它在本轮的禁改面内），
所以这一行留给收口批做。

## 十一、留给后面的事

1. **spec/10第二、三节升"已验证"的条件还没满足**：要一个消费方通过其全部门禁的
   实现，本批给的是引擎侧的接口与两个实现，WII/WDS侧的采纳还没发生。
2. **`apply`的物理**：接能量项与求解器那一块。落地时`actuators`的环归属要重判
   （第六节已写触发条件）。
3. **`CollisionQuery.check_path(source: MotionSource)`**（spec/10第五节）
   现在有对象可以接了——扫掠实现仍未做。
4. **spec/15第二节"两处需要说明的判断"值得补第三条**：解释同一页的三个接口
   为什么分在两个圈。本批只动了环表那一行（禁改面所限）。
