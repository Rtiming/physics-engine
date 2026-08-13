# 矩形弹塑性纤维截面的单调弯曲与自由回弹

这是plans/08阶段4的第一条真实截面案例：截面上同时存在弹性区与塑性区，逐点材料
历史写进`State`，弯矩由点响应求积，卸载后的残余曲率由局部平衡`M=0`求出。

它刻意不走线弹性`EI`恒等路径，也不把解析曲线采样后冒充求解。案例身份
`case/rectangular_section_springback`，实现裁决见[0059](../../docs/decisions/0059_截面积分点与弹塑性纤维截面_20260813.md)。

## 一、物理/几何设定

矩形截面宽`b=12 mm`、厚`h=4 mm`，中性轴在`y=0`；一维小应变理想弹塑性材料
`E=200000 N/mm²`、`sigma_y=250 N/mm²`。采用平截面假设与本仓明确的正号约定：

```text
epsilon(y) = epsilon_0 + kappa*y
N = sum(sigma_i*A_i)
M = sum(sigma_i*A_i*y_i)
```

截面沿厚度切成等高、等面积的中点纤维，跑`8/16/32/64`四档；每点权重
`A_i=b*h/n`。**中点不是表面点**：64点时最外点在`|y|=1.96875 mm`，不得把它的
应力命名为“表面应力”。求积规则身份固定为`section_rule/midpoint_equal_area/1`。

本构逐点走trial/commit分离的一维return-map。状态向量的编号是：

| 段 | 宽度 | 性质 |
|---|---:|---|
| `section_axial_strain` | 1 | 广义截面变形；本案例固定为0 |
| `section_curvature_per_mm` | 1 | 广义截面变形；回弹时的标量平衡未知量 |
| `section_point_plastic_strain` | n | 逐点真历史，不是全局运动自由度 |
| `section_point_accumulated_plastic_strain` | n | 逐点真历史，不是全局运动自由度 |

因此64点布局有`130`个状态标量和`2`个广义运动学坐标；本案例固定轴向应变，实际只解
曲率这`1`个标量平衡未知量。“状态宽度”、“积分点数”、“广义坐标数”和“全局未知
自由度”不能混称。几何、点数与求积规则的内容地址被绑进`layout_id`，同点数但厚度
不同的截面不能误读旧历史。

加载路径两步：先从处女态单调加载到`kappa=2*kappa_y`，再保持零轴向应变、令外弯矩
为零，在显式区间`[0,2*kappa_y]`内求自由回弹曲率。求解采用区间保护Newton：一致
切线的候选在夹根区间内才采用，否则退回二分。

## 二、参考解出处

同行方法只作**架构证人**，不作本案例数值金标：

- Abaqus的分析时积分梁截面从广义截面应变计算各截面点材料响应，再把应力积分为
  截面力；塑性反向加载需要更多点。见官方[Beam Section Behavior](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEELMRefMap/simaelm-c-beamsectionbehavior.htm)与
  [Using a Beam Section Integrated during the Analysis](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEELMRefMap/simaelm-c-usingbeamsection.htm)。
- OpenSees固定提交`40f6a3d`的`FiberSection3d`从截面变形算每纤维应变，逐点调用
  材料并积分N/M与切线；材料状态逐纤维commit/revert。见源码
  [L453-L528](https://github.com/OpenSees/OpenSees/blob/40f6a3d6d6c1efebb4cb7a85943bb6e163afa93a/SRC/material/section/FiberSection3d.cpp#L453-L528)与
  [L684-L732](https://github.com/OpenSees/OpenSees/blob/40f6a3d6d6c1efebb4cb7a85943bb6e163afa93a/SRC/material/section/FiberSection3d.cpp#L684-L732)。
- MOOSE明确把stateful material property按元素、求积点保存并警告内存代价；见官方
  [Materials System](https://mooseframework.inl.gov/syntax/Materials/index.html)。这支持
  “不加全局DOF仍会增加历史状态与内存”，不支持把积分点说成零成本。

本案例的**第1档数值金标**由本仓独立推导。令`c=h/2`、
`kappa_y=sigma_y/(E*c)`、`I=b*h³/12`、`M_p=b*sigma_y*c²`：

```text
|M| = E*I*|kappa|                                      |kappa| <= kappa_y
|M| = b*sigma_y*(c² - a²/3), a=sigma_y/(E*|kappa|)    |kappa| > kappa_y
```

所以`kappa=kappa_y`时`M=2*M_p/3=8000 N·mm`，`kappa=2*kappa_y`时
`M=11*M_p/12=11000 N·mm`，完全塑性极限`M_p=12000 N·mm`。无反向屈服的卸载
满足`M_unload=M_loaded+E*I*(kappa-kappa_loaded)`，连续截面的自由回弹曲率为
`1/2560 mm⁻¹`。

金标生成器[`generate_oracle.py`](generate_oracle.py)不import`physics_engine.sections`，
只用`Fraction`分段闭式与精确逐纤维求和；8/16/32/64点的屈服边界都恰落在纤维边界，
故中点弯矩误差加密一倍后精确缩小4倍。生产端则走逐点return-map与局部非线性求解。

## 三、判据表

| 量 | rel/abs | 理由 |
|---|---|---|
| 64点加载弯矩对连续闭式`11000 N·mm` | abs`2 N·mm` | 离散精确误差`1.953125 N·mm`，留2.4%离散余量；错误线弹性值`16000`远在门外 |
| 8→16→32→64点弯矩误差比 | abs`1e-10`对`4` | 屈服界面刻意与纤维边界对齐，中点规则二阶收敛是可证伪的，不以单点精度冒充 |
| 64点应力、塑性应变、屈服标志全分布 | 应力abs`1e-11`、历史abs`1e-17`、布尔零容差 | 只验汇总弯矩抓不住点序反转、共享材料状态或错误屈服前沿 |
| 初屈服弯矩与完全塑性矩 | abs`2 N·mm`／`1e-10 N·mm` | 同时钉住弹性端、混合区与塑性平台，防实现只在一个加载点凑对 |
| 64点离散回弹曲率 | abs`1e-15 mm⁻¹` | `Fraction`离散解析值对生产端受保护Newton，不拿连续离散误差掩盖求解误差 |
| 回弹曲率对连续闭式`1/2560 mm⁻¹` | abs`6e-8 mm⁻¹` | 64点离散误差`5.723443e-8`，余量4.8%；丢历史会得到0并明确失败 |
| 回弹后内弯矩 | abs`1e-9 N·mm` | 与调用方声明残差相同，平衡必须直接验，不能只验曲率看起来接近 |
| 卸载点应力、历史不污染、逐字节重放 | 应力abs`1e-9`；布尔零容差 | 同一曲率不同历史必须不同；同一路径状态规范字节必须相同 |

清单正本是[`oracle.json`](oracle.json)。所有判据均禁止静默skip。

## 四、已知失效清单

- **不是二维任意截面**：宽度只进面积权重，不能表示宽度方向应力梯度、翘曲或局部
  屈曲；第一片不预支三角形/多边形网格。
- **不是Timoshenko或壳**：平截面假设没有横剪、扭转、压扁、层间应力和厚度方向
  三维响应。
- **不是表面积分**：中点规则不含`y=±h/2`。若需要真实表面输出，应另裁包含端点的
  Simpson/Lobatto类规则，不能把最外中点改名。
- **材料范围有限**：小应变一维理想弹塑性、零硬化、无损伤、无温度；return-map可处理
  反向加载，但本案例的解析金标只覆盖单调加载后未反向屈服的卸载。
- **没有全局梁单元装配**：本案例解的是单站截面局部平衡，不是WDS整条杆的节点—截面
  联立求解。
- **点数固定**：加载中不自适应换点。换点数或规则就是新布局，已有历史不能自动投影；
  本仓还没有有证据的历史投影算法。
- **仍无电磁场自由度**：本案例不改变S1.2。Norris闭式的`K(x)`仍是闭式值，不是本轮
  通过未知`K_i`、总电流约束和非局部磁核解出的场。
- **跨平台未实测**：金标容差按IEEE-754基本运算与固定求和次序给出；macOS/arm64已测，
  Windows/Linux仍标未验证。

## 五、档位与负载级

B档，**interactive**。案例conformance本机约`0.08 s`；64点回弹单解经profile后由
纯二分40次截面装配优化为受保护Newton 2次装配，七组500次样本中位数
`0.000852 s/solve`，相对优化前`0.011474 s/solve`快约`13.5×`。墙钟只记账，
确定性门是“本构切线可用时迭代不超过8次”。

## 六、本案例不是什么

- 不证明“体积与厚度”结构性缺口已关闭；它只完成**局部截面本构/分布切片**。
- 不证明S5.1已done：截面上能放应力与材料历史，但没有宽度/厚度方向独立运动学场，
  不能压扁、翘曲或局部屈曲；S5.1最多从todo升partial。
- 不证明S1.2有电流场自由度；机械应力积分点不能跨域冒充电磁未知量。
- 不证明完整塑性梁已接进全局`EnergyRegistry`/`solve_equilibrium`；本案例走
  `state`→`sections`→局部平衡求解这一条新路径。
- 不证明中点规则优于Gauss/Simpson/Lobatto。当前选择是第一消费方所需的最小稳定
  纤维分槽；第二条真实截面需求出现后再用同一接口比较规则，不预支通用正交库。

## 七、必须红矩阵

`tests/cases/test_rectangular_section_springback.py::RED_MATRIX`逐格注入六种错法：

1. 永远用线弹性`EI`；
2. 把截面塌成质心一个点；
3. 点序反转却沿用旧历史；
4. 全部点共享一份塑性状态；
5. 卸载时丢历史、伪造零回弹；
6. 把点应力写成当前曲率的无历史函数。

六格均走与正向相同的oracle判据并必须抛`OracleError`；判据变松到放过任一典型错法，
测试本身就会红。
