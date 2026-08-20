# model_scene_assembly P3-M1：模型输入装配为Scene

判据正本为同目录`oracle.json`，生成器为`generate_oracle.py`。本案例不读取WII或GCW，
只验证physics-engine自己的P3.0输入包能否经P3.1模块化装配进入既有Scene、MotionSource、
虚拟frame和CollisionQuery。清单身份：`case/model_scene_assembly`；负载级：`interactive`。

## 一、物理/几何设定

模型含张力机、工件和机器人显示组件。张力机与工件各有一份小型合成碰撞资产字节；
loader必须从案例包根重新读取并核SHA，再接显式AABB、凸性和保守方向。机器人只有visual，
其组件和显示运动track都明确排除，不进入Scene。

张力机为static体。工件为kinematic体，其组件位姿从x=20mm线性移动到0mm，碰撞资产相对
组件另有x=1mm安装偏置，所以资产实际位置依次为21mm、11mm、1mm。process frame从
x=10mm移动到12mm，只作为虚拟frame输出。

接触候选只声明“张力机—工件”一对。张力机AABB为正负2mm，工件为正负1mm；0s分离，
1s重叠。fixture没有解析网格表面，因此事件可信度只能是`broad_phase`。

## 二、参考解出处

生成器只做加法、线性中点和整数计数，不import`model_scene`、`scene_resources`、`motion`
或`collision`。两份资产SHA由生成器直接读取fixture字节计算。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---:|---:|---|
| body、排除组件/track、可信度、资格 | 0 | 0 | 身份与证据等级逐字 |
| 工件资产x位置 | 2e-15 | 2e-15mm | 组件位姿后乘1mm资产偏置 |
| process frame中点 | 2e-15 | 2e-15mm | 独立线性中点 |
| 候选数、事件数 | 0 | 0 | 显式一对；0s分离、1s重叠 |

## 四、已知失效清单

1. 资产是合成fixture，不是可制造网格；本案例只证明字节身份和AABB记录接线。
2. 事件是broad phase，不证明真实三角面接触、法向、侵入量或接触力。
3. 没有dynamic体；dynamic状态frame相对质心的语义未冻结，当前assembler失败关闭。
4. 没有材料、本构、摩擦、张力或传感器读数。
5. 没有WII adapter、official timeline、WDS采用或现场标定。
6. 无skip。

## 五、档位与负载级

**A档/交互级（`interactive`）**。两个物理体、三个运动track、两个样点和两份几十字节资产。

## 六、本案例不是什么

- 不是WII真实回放；
- 不是网格窄相验收；
- 不是dynamic刚体装配；
- 不是场景⑥端到端完成。
