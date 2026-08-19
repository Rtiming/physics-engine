# tools/model/sdf_bake/ —— SDF烘焙档（point-cloud-utils）

承接[decisions/0074](../../../docs/decisions/0074_真实网格进接触_碰撞归本仓_距离场是根本形制.md)第5.1节
与[decisions/0085](../../../docs/decisions/0085_网格与距离场_窄带块稀疏与三次B样条_20260818.md)第四节。
内核那一半在`src/physics_engine/contact/field.py`，**它不知道本目录存在**。

## 一、身份边界（照`tools/model/README.md`第一节，一个字不改）

1. **永不进`dependencies`。** 连`[optional-dependencies]`都不进。
2. **永不被`src/physics_engine` import。** 由
   `tests/governance/test_model_tools_stay_out_of_the_kernel.py`守着。
3. **重依赖装独立venv**：`tools/model/sdf_bake/.venv`，用`setup_sdf_bake_env.sh`建。

## 二、装上了没有：**装上了**（2026-08-18本机实测）

0074第5.1节写的是"选pcu"，本目录把它**真装了一次**。实测：

| 项 | 值 |
|---|---|
| 包 | `point-cloud-utils==0.34.0`，MIT |
| wheel | `point_cloud_utils-0.34.0-cp313-cp313-macosx_11_0_arm64.whl`，6.9 MB |
| 传递闭包 | `numpy==2.5.2`、`scipy==1.18.0`（**pcu还拉了scipy，0074没提到这一条**） |
| 平台 | macOS-26.5.2-arm64、Python 3.13 |
| 0074点名的两个API | `triangle_soup_fast_winding_number`✅、`signed_distance_to_mesh`✅ |

**所以没有"装不上"这条GAP。** 剩下的GAP是别的（见第四节）。

## 三、两条实测到的语义差异（**这一节比装没装上更值钱**）

形制照`cases/peer_fcl_distance`那条"语义差异清单"——
它是那条轨道最贵的一条发现，本目录复用同一条纪律：**同行库的文档不是金标，实测才是。**

### 3.1 `triangle_soup_fast_winding_number`的文档**写反了**

0.34.0的docstring原话是"positive for outside and negative for inside"。
**实测返回的是广义缠绕数本身**：球心处`1.00124`、球外`1.30e-05`、
球面外侧一点`-8.34e-04`。也就是**内≈1、外≈0**，
判内外的阈值是**0.5**而不是符号——按文档写"符号"会把球外那些`-8.34e-04`判成"内"。

### 3.2 `signed_distance_to_mesh`的符号**不来自面法向**

把整张网格的三角**全部反向**：

* 有符号距离最大只变 **3.86e-07 mm**，符号**一个点都没翻**；
* 缠绕数从`+1.00124`整体翻到`-1.00124`（与原值之和的最大绝对值 1.52e-07）。

**这正是0074第5.1节选它的那条性质的直接证据**——内外由体积判据定，不靠面法向投票。
**代价要一起写**：调用方**无法**通过反转网格来反转场的符号；要反号只能在数值上取负。

**那个3.86e-07不是零**。pcu的fast winding number带层次近似，
所以这里写的是实测量而不是"逐位不变"——把它说成逐位不变就是冒充一个没验过的性质。

## 四、跑法

```bash
bash tools/model/sdf_bake/setup_sdf_bake_env.sh      # 建环境（只装进本目录的.venv）

PYTHONPATH="${PWD}/src" tools/model/sdf_bake/.venv/bin/python \
    tools/model/sdf_bake/bake_sphere_probe.py \
    --out tools/model/sdf_bake/sphere_probe.report.json

PYTHONPATH="${PWD}/src" .venv/bin/python -m pytest tests/test_sdf_bake_tool.py -q
```

**两条命令用的是两个环境，中间只有一份JSON**——与`validation/run_comparison.py`
和`tools/view/`那两条链路逐字同源。报告**进版本控制**（几个数，很小），
所以即使没建venv，本仓也还记得这一轮量到了什么。

## 五、本目录留下的GAP（决策0085第五节有触发条件）

| GAP | 触发条件 |
|---|---|
| **只烘过解析球，没烘过真实件**（`e1_carrier`336三角是0074点名的第一个真实语料） | 真实资产落到本机、且`tools/model/centerline_csv.py`那条链路把它带进来时 |
| **窄带外的行为是失败关闭，没有背景符号瓦片** | 第一个真实件烘进来时；那时才需要"深在体内"与"远在体外"分得开 |
| **pcu还拉了scipy，0074第5.1节的选型表没算这一条** | 下一次比较烘焙候选时；scipy不改变裁决（都在venv里），但选型表应当如实补上 |
