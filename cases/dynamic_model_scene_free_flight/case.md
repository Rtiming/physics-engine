# dynamic_model_scene_free_flight P3-M2：偏心几何绕质心自由转动

判据正本为同目录`oracle.json`，生成器为`generate_oracle.py`。本案例验证P3.1 dynamic
状态frame、质量属性和Scene几何位姿闭环，不读取WII或GCW。清单身份：
`case/dynamic_model_scene_free_flight`；负载级：`interactive`。

## 一、物理/几何设定

一个dynamic工件的geometry resource frame初始原点在世界系x=10mm。质量属性声明质心在
geometry frame内x=2mm，惯量绕质心且在geometry轴表达，故初始COM为x=12mm。body/COM轴与
geometry轴平行。

初始世界系线速度为0，体系角速度为绕z轴πrad/s；惯量为diag(1,1,2)kg·mm²。无外力、无外
力矩，以RK4、dt=0.001s推进500步，即0.5s。理论上COM不动，姿态绕z转90°，geometry原点相对
COM的向量由(-2,0,0)转成(0,-2,0)，所以末态geometry原点为(12,-2,0)mm。

模型运动包仍需满足至少两个样点的合同，因此带一条恒定process virtual frame时间线；它不驱动
dynamic体，也不伪造第二个状态所有者。

## 二、参考解出处

生成器不import`dynamic_body`、`model_scene`或`rigidbody`，只用匀角速度闭式、二维90°旋转、
向量加法和fixture文件SHA。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---:|---:|---|
| 初始COM | 0 | 0 | 10mm+2mm精确加法 |
| 初始geometry原点 | 2e-15 | 2e-15mm | 状态往返 |
| 末态COM | 2e-12 | 2e-12mm | 无外力 |
| 末态姿态 | 2e-11 | 2e-11 | 0.5s内绕z转90° |
| 末态geometry原点 | 5e-11 | 5e-11mm | 偏心向量绕COM转动 |
| 归一化次数、资格 | 0 | 0 | 确定性整数与证据等级 |

## 四、已知失效清单

1. 碰撞资产仍是合成fixture，只验证字节/SHA与frame接线，不是三角网格真值。
2. 无接触、重力、阻尼、摩擦、约束、张力和传感器。
3. 只验证单个自由刚体和主轴恒转速；不证明多dynamic体接触管线完成。
4. RK4不是辛积分器，长时间守恒不属于本案例。
5. 无WII adapter、WDS采用或现场标定。
6. 无skip。

## 五、档位与负载级

**A档/交互级（`interactive`）**。单体13维状态、500个RK4步。

## 六、本案例不是什么

- 不是dynamic接触求解；
- 不是网格质量属性计算；
- 不是WII真实运动；
- 不是场景⑥端到端完成。
