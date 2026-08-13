# WDS运动学三节点顶点的弹塑性纤维截面回弹

这是阶段4的第二片：把0059的单站局部截面本构接到一个真实全局状态布局、
`EnergyRegistry`与`solve_equilibrium`。案例身份
`case/kirchhoff_section_vertex_springback`，实现裁决见
[0060](../../docs/decisions/0060_WDS运动学截面站点全局装配_20260813.md)。

它是**引擎侧WDS运动学兼容切片**，不是WDS消费仓迁移。WDS当前物理源只读对拍，
没有改它的代码、依赖、案例产物或回执；因此本案例不能用来声称“WDS已采用”。

## 一、物理/几何设定

一个内顶点含三个节点`x0/x1/x2`与两条边的材料扭角`gamma0/gamma1`。曲率严格采用
WDS/Bergou离散杆定义：

```text
kb = 2*(e0 × e1) / (|e0|*|e1| + e0·e1)
m2_i = -sin(gamma_i)*d1_i + cos(gamma_i)*d2_i
kappa1_discrete = 0.5*(m2_0 + m2_1)·kb
kappa = (kappa1_discrete - natural_kappa1) / dual_length
```

`dual_length=(L0+L1)/2`。参考导向`d1/d2`逐边冻结，必须是正交单位向量；相邻边
近反平行时分母趋零，模型本身奇异，生产端失败关闭。

截面为`b=12 mm`、`h=4 mm`、64个等面积中点纤维；材料为一维理想弹塑性
`E=200000 N/mm²`、`sigma_y=250 N/mm²`。第一片固定`epsilon_0=0`，只把easy-axis
曲率接进全局装配。截面的单位长度增量势是：

```text
phi_i(epsilon) = 0.5*E*(epsilon-ep_old)^2                         弹性
               = sigma_y*|epsilon-ep_old| - 0.5*sigma_y^2/E      塑性
Psi = sum(A_i*phi_i)
U_vertex = dual_length*Psi(kappa(x,gamma))
```

所以全局残差与一致Hessian不是等效`EI`拼接，而是同一份势能的链式导数：

```text
g = dual_length*M*grad(kappa)
H = dual_length*[Dkk*grad(kappa)⊗grad(kappa) + M*hess(kappa)]
```

第二项是几何刚度，漏掉它时力可能对而Newton切线仍是错的。`grad/hess(kappa)`由纯
Python二阶jet解析求出；有限差分只在测试里作独立导数oracle，不进入生产路径。

状态布局固定如下：

| 段 | 宽度 | 数值角色 |
|---|---:|---|
| `node_positions_mm` | 9 | 三节点位置；也是`node_dof_count`的完整前缀 |
| `edge_twist_angles` | 2 | WDS式逐边材料扭角，属于运动学未知块 |
| `section_point_plastic_strain` | 64 | 逐点真历史，Newton自动固定 |
| `section_point_accumulated_plastic_strain` | 64 | 逐点真历史，Newton自动固定 |

因此共有139个状态标量、11个运动学坐标；本案例再固定其中10个，只释放`x2_y`这
一个全局未知量。状态宽度139绝不能报成“Newton解139个自由度”。

加载分两步：先位移控制，固定`L=100 mm`并把`x2_y`置为`640/51 mm`，几何精确
曲率恰为`1/800 mm⁻¹=2*kappa_y`；32个外侧纤维屈服。然后把这份点历史提交，移除
位移约束并在零外载下由全局Newton求`x2_y`回弹位置。失败Newton保留最后trial构型
供诊断，但返回的`committed_state`必须逐字节仍是加载态。

## 二、参考解出处

第一条金标由[`generate_oracle.py`](generate_oracle.py)独立生成，不import
`physics_engine.section_beam`或`physics_engine.sections`。平面几何可有理反解：令
`q=kappa*L=2*tan(theta/2)`，则

```text
v = L*q/(1-q^2/4)
```

因此加载位移、64点纤维应力/弯矩、`dU/dv`、`d²U/dv²`、卸载曲率与回弹位移全用
`Fraction`精确计算。生产端走sqrt、二阶jet、return-map与Newton，两边不共享数值实现。

第二条金标是实际消费方的**只读兼容夹具**。2026-08-13核验WDS提交`c1b8fe6`时，
参与夹具的两份物理源没有工作树改动：

| WDS源 | SHA-256 |
|---|---|
| `model/state.py` | `ea61bf2611ce30fb91248f9092d5cdf2eff82a0688926253ac9e929b30577c27` |
| `model/energies.py` | `2d3e4d1784c94898dd2efb185091e29c2041fe4513e569e0fb0e5b99c1ed7d77` |

用非平面节点、非零两边扭角、不同左右参考长度生成WDS的`kappa1`、弹性能与完整
11维梯度，再由本仓独立实现对拍。夹具钉进`oracle.json`，含状态次序
`x.ravel_then_gamma`。WDS仓其余映射设施当时有未提交工作，本批没有触碰；仓级产物
逐字节不变必须等它的独立迁移会话，不由本案例代签。

同行方法只作架构证人：OpenSees的`FiberSection3d`同样从广义截面变形求每纤维响应、
积分截面合力与切线，并逐纤维commit/revert；固定源码见
[试算与装配](https://github.com/OpenSees/OpenSees/blob/40f6a3d6d6c1efebb4cb7a85943bb6e163afa93a/SRC/material/section/FiberSection3d.cpp#L453-L528)和
[commit/revert](https://github.com/OpenSees/OpenSees/blob/40f6a3d6d6c1efebb4cb7a85943bb6e163afa93a/SRC/material/section/FiberSection3d.cpp#L684-L732)。
本仓采用相同的trial/commit边界，但不预支OpenSees的通用截面/单元注册体系。

## 三、判据表

| 量 | rel/abs | 理由 |
|---|---|---|
| 加载位移与曲率 | rel`0`；abs`2e-14 mm`/`2e-18 mm⁻¹` | Bergou曲率有理反函数；能判红小转角`kappa=v/L²`替代 |
| 加载弯矩 | rel`2e-15`，abs`1e-10 N·mm` | 64点`Fraction`逐纤维精确和；错误线弹性值偏约5002 N·mm |
| 全局结点力`dU/dv` | rel`3e-14`，abs`2e-11 N` | 独立有理链式法则，验证M真的装进全局残差 |
| 全局切线`d²U/dv²` | rel`3e-12`，abs`2e-9 N/mm` | 独立二阶链式法则，包含材料切线与几何刚度两项 |
| WDS曲率/能量/11维梯度 | abs`2e-18`/`1e-15`/`3e-14` | 对固定提交与两份源SHA的只读实测夹具；覆盖状态次序、frame、扭角与dual length |
| 全局回弹位移/曲率 | rel`2e-12`、abs`2e-10 mm`/`2e-14 mm⁻¹` | `Fraction`截面回弹再经有理几何反函数；容差含`1e-9 N`全局残差 |
| 回弹弯矩 | abs`2e-7 N·mm` | 由全局力残差经非零几何Jacobian反推并宽取，不拿位移像不像代替平衡 |
| 屈服点数、收敛、历史不变、失败不提交、逐字节重放 | 零容差 | 离散整数、布尔与规范字节没有模糊空间 |

生产Hessian另由对生产梯度的11列中心差分逐项验证，rel`2e-6`、abs`3e-9`；这道门
不作物理金标，只验证二阶jet与装配的链式导数。

## 四、已知失效清单

- **只有一个内顶点**：没有沿杆多站装配、边界站、积分站选择或历史投影。
- **只有easy-axis弯曲**：轴向应变固定为零；虽然局部响应仍能报轴力N，本模块没有
  把N装进节点残差。hard-axis、扭转、剪切、压扁和翘曲均失败在能力边界之外。
- **不是完整WDS各向异性弯曲**：兼容夹具把`EI_hard=0`，只对拍本批承诺的kappa1分量；
  WDS的kappa2、twist、stretch、接触和全杆求解没有迁入。
- **理想塑性非光滑**：屈服点处势能C1但非C2；当前案例的求解路径不把终点放在屈服
  折点。更一般载荷控制、信赖域、弧长法和载荷步回退仍不存在。
- **固定中点规则**：最外点不是表面，不能报“表面应力”；换规则或点数就是新布局，
  旧历史不能自动投影。
- **参考frame由调用方负责**：本模块验证`d1/d2`互相正交且单位化，不生成、重传输或
  证明它们与某条CAD中心线同源。
- **跨平台未实测**：macOS/arm64/CPython 3.13已测；Windows/Linux仍未验证。

## 五、档位与负载级

B档，**interactive**。确定性门是：全局回弹收敛且不超过8次Newton、历史提交边界
与独立金标全过。profile后零/一/二阶按需求值；收口时在宿主高负载下同进程交替跑
旧全二阶路径与优化路径，七组中位数为`0.046551 s→0.019071 s`（2.44×），仍为
4次Newton、0回溯；预算`0.1 s`。墙钟只记账，不进物理pass/fail；优化前后由零/一阶
对二阶通道的逐位门守结果不变。

## 六、本案例不是什么

- 不证明WDS已经消费`section_beam`。本批只读它的固定源与数值输出；WDS真实采纳须在
  自己仓内冻结diff、更新vendor wheel、重签回执并验证既有产物逐字节不变。
- 不让S5.1从partial升done：截面点仍是本构/求积点，不是宽厚方向独立运动学场；
  压扁、翘曲和局部屈曲仍不能表达。
- 不改变S1.2：机械纤维历史不是电流未知量，仍没有`K_i`、总电流约束或非局部磁核。
- 不证明多匝带材场景⑤端到端：没有整杆、多站、接触、喂料、半径生长、宽度运动学
  或消费方运行包。
- 不把“11个运动学坐标”说成“当前求11个未知量”：本案例固定10个，只解`x2_y`；
  128个历史槽又全部自动固定。

## 七、必须红矩阵

conformance逐格注入六种错法：小转角曲率、线弹性EI、常数几何Jacobian、零全局切线、
卸载丢历史、失败Newton偷提交。六格均走同一份oracle容差并必须红；正向结果本身不会
替判据证明判据有辨错能力。
