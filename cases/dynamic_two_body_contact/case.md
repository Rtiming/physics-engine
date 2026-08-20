# dynamic_two_body_contact P3-M3：两dynamic刚体的检测—力矩—耦合推进

判据正本为同目录`oracle.json`，生成器为`generate_oracle.py`。清单身份：
`case/dynamic_two_body_contact`；负载级：`interactive`。

## 一、物理/几何设定

两个质量均为1kg、碰撞半径均为10mm的dynamic刚体只声明一对候选。几何球心初始位于
x=-9.5mm与x=9.5mm，故穿透1mm；法向罚刚度100N/mm、阻尼0。第一档把两体COM均放在
各自geometry frame的y=2mm处，检验碰撞见证点到COM的杆臂和两侧`r×F`；第二档令COM与
球心重合，检验两个等质量由1mm初始压缩静止释放的四分之一接触周期。

状态由`PreparedModelScene.initial_dynamic_states()`产生；每个RK4子阶段都同时评价两体的
26维组合状态，不顺序调用两个单体积分器。模型运动包只带一条恒定process virtual frame，
不驱动任何dynamic体。

## 二、参考解出处

生成器不import`dynamic_contact`、`model_scene`或`rigidbody`。偏心档只用球几何、
作用反作用和叉乘；对齐档使用两质量法向相对运动闭式：

```text
m_eff = m_a m_b / (m_a + m_b)
omega = sqrt(1000 k / m_eff)
delta(t) = delta_0 cos(omega t)
t_quarter = pi / (2 omega)
```

四分之一周期时穿透归零，相对分离速度为`delta_0 omega`；等质量各承担一半。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---:|---:|---|
| 穿透、法向、两见证点 | 1e-14 | 1e-14mm | 解析球几何与见证点恒等式 |
| 两侧力与体系力矩 | 1e-14 | 1e-14 | 作用反作用与同一事件的`r×F` |
| 世界系总力/总力矩 | 1e-14 | 1e-13 | 内力不能制造总线动量或总角动量 |
| 四分之一周期位置/速度 | 2e-8 | 见oracle | 等质量相对运动闭式 |
| 末态总动量/动能 | 1e-7 | 见oracle | 内力守恒与初始罚簧能 |
| RK4求值/归一化次数 | 0 | 0 | 确定性整数 |

## 四、已知失效清单

1. 只覆盖解析球—球、单一显式候选和法向罚簧/压缩阻尼；没有切向摩擦历史、滚动阻力或多个候选。
2. 显式RK4仍为`production_ready=False`；只自动执行接触刚度步长界，转动模态仍须另取更紧者。
3. 偏心质量属性是合成`hypothesis_only`输入，不证明真实工件质量分布。
4. 无静止/运动平面、胶囊/盒/网格动态响应、约束、关节和自适应步长。
5. 无WII adapter、WDS采用、真实资产或现场标定。
6. 无skip。

## 五、档位与负载级

**A档/交互级（`interactive`）**。两体26维状态、400个RK4步、每步四次单候选查询。

## 六、本案例不是什么

- 不是通用多体接触求解器；
- 不是摩擦或碰撞恢复系数标定；
- 不是网格不规则体落漏斗；
- 不是WII真实运动，也不改变17/42和端到端0/6。
