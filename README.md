# MSAC-Net：用于乳腺癌 IHC 定义 Luminal 状态预测的病灶语义对齐与校准感知网络

> **Multimodal Semantic Alignment and Calibration Network (MSAC-Net)**
> 术前预测免疫组织化学（IHC）定义的 **Luminal / Non-luminal** 型乳腺癌，融合钼靶（MG，CC/MLO 双视图）、超声（US）与结构化临床变量，联合优化分类性能与置信度可靠性。

[![GitHub repo](https://img.shields.io/badge/GitHub-yueyuanli1%2FMSAC--Net-blue)](https://github.com/yueyuanli1/MSAC-Net)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 概述（Abstract）

乳腺癌治疗与预后高度依赖分子亚型，术前确定 Luminal 状态对制定个体化治疗方案尤为关键。现有单模态方法难以充分利用不同数据源中的互补信息，现有多模态融合方法在**跨模态特征交互、局部病灶表征和预测置信度可靠性**方面仍存在不足。

为此，本文提出 **MSAC-Net**，包含三个核心模块：

- **渐进式双向互学习注意力（PBMA, Progressive Bidirectional Mutual-learning Attention）**：通过双向稀疏掩码和两层级联的互学习块，实现 MG 与 US 之间由粗到细的 token 级交互，过滤冗余连接、降低计算量。
- **解剖语义对齐模块（ASAM, Anatomical Semantic Alignment Module）**：利用分割掩码作为病灶锚点，将 MG / US 的 ROI 特征投影到共享语义空间，以余弦相似度损失约束配对病灶表征一致。
- **置信度校准与反馈模块（CCFM, Confidence Calibration and Feedback Module）**：在独立校准集上估计分箱校准间隙，作为附加损失反馈至训练过程，实现分类性能与置信度可靠性的联合优化。

MSAC-Net 在中心 A 的 401 例患者上开发（患者级 5 折交叉验证），并在中心 B 的 99 例患者上独立测试：**内部 / 外部 AUC 分别为 0.8051 与 0.7965，ECE 分别为 12.20% 与 15.49%**，在分类性能与置信度校准指标上均取得具有竞争力的结果。

## 框架图

整体框架见 `fig/final-1.pdf`（多模态预测方法范式对比）与 `fig/final-2.pdf`（MSAC-Net 网络结构）。

```text
MG(CC/MLO) ─┐
US          ─┤── 骨干网络(ResNet-50) ──┐
分割掩码     ─┘  ROI 裁剪 + 特征门控     │
                                        ▼
                              PBMA（渐进式双向互学习注意力）
                                        │
                              ASAM（病灶级语义对齐）
                                        │
                              分类器 + CCFM（校准反馈）──► 预测 + 置信度
临床变量(年龄/绝经状态) ──────────────────┘
```

## 主要结果

**内部（中心 A，患者级 5 折，均值 ± 95% CI 半宽，单位 %）：**

| 方法 | 模态 | AUC↑ | ACC↑ | REC↑ | PRE↑ | F1↑ | ECE↓ | MCE↓ | NLL↓ | Brier↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| CDLS [31] | MG+US+C | 79.78 | 77.80 | 65.68 | 77.06 | 58.34 | 15.80 | 39.43 | 54.83 | 36.13 |
| **Ours** | **MG+US+C** | **80.51** | **83.05** | **88.69** | **88.69** | **88.37** | **12.20** | **34.30** | **53.98** | **33.83** |

**外部（中心 B，独立测试，均值 ± 95% CI 半宽，单位 %）：**

| 方法 | 模态 | AUC↑ | ACC↑ | REC↑ | PRE↑ | F1↑ | ECE↓ | MCE↓ | NLL↓ | Brier↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| CDLS [31] | MG+US+C | 78.55 | 78.79 | 93.33 | 81.82 | 86.60 | 16.54 | 49.35 | 91.45 | 37.82 |
| **Ours** | **MG+US+C** | **79.65** | **80.61** | **95.83** | **82.09** | **87.99** | **15.49** | **28.23** | **74.28** | **37.77** |

> 完整对比表见论文 Table I–VII；消融（组件 / 模态 / 主干 / 校准方法）与复杂度分析详见论文。

## 数据集与数据划分协议

- **中心 A（开发队列）**：401 例（Luminal 286 / Non-luminal 115），采用**患者级 5 折交叉验证**。
- **中心 B（外部测试队列）**：99 例（Luminal 81 / Non-luminal 18），仅用于外部测试，不参与训练、调参或阈值确定。

每折划分如下（相对中心 A 全队列）：

| 集合 | 比例 | 用途 |
|---|---|---|
| 参数训练集 | 60% | 梯度更新（反向传播） |
| 校准反馈集 | 10% | 计算校准间隙与校准损失（**不参与梯度更新**） |
| 内部验证集 | 10% | 选择最佳 epoch、调整学习率、早停、确定分类阈值 |
| 外层测试集 | 20%（约 80 例） | **仅在模型、超参数和阈值锁定后**做该折的最终评价 |

**关键约束**：外层测试集在模型、超参数和阈值锁定前**不得访问**。

**正类定义**：`0 = Luminal`，`1 = Non-luminal`；REC（灵敏度）/SPE/PRE/F1/AUC 以 **Luminal（label 0）为正类**计算（与论文「临床信息加入主要提升敏感度、减少 Luminal 型假阴性」的表述一致），可通过 `--positive_label` 指定。

内部评价采用五折指标的均值 ± 95% 置信区间半宽（基于五折指标的 t 分布，自由度 4）。

## 环境安装

```bash
conda env create -f environment.yml
conda activate mshf
```

或使用 pip：

```bash
pip install -r requirements.txt
```

主要依赖：PyTorch 2.10.0、torchvision 0.25.0、MONAI 1.5.2、timm 1.0.28、scikit-learn、opencv-python、SimpleITK、nibabel、tensorboard 等（详见 `requirements.txt`）。

## 数据准备

### 目录结构

在 `configs/config.yaml` 中配置数据路径：

```yaml
data_dir: "/path/to/data"
imgMG_dir: "mg_nii/mg_images_nii"      # MG 图像（CC/MLO）
maskMG_dir: "mg_nii/mg_masks_nii"      # MG 分割掩码
imgUS_dir: "us_nii/us_images_nii"      # US 图像
maskUS_dir: "us_nii/us_masks_nii"      # US 分割掩码
clinical_dir: "configs/clinical.json"  # 临床/标签信息
```

文件命名（`tumor_location` 记为 `loc`，`loc ∈ {0, 1}`）：

- MG CC：`{patient_id}_{loc*2+1}.nii.gz`
- MG MLO：`{patient_id}_{loc*2+2}.nii.gz`
- US：`{patient_id}_{loc+1}.nii.gz`

### clinical.json 格式

`configs/clinical.json` 为一个 JSON 数组，每个元素对应一位患者：

```json
[
  { "id": "P001", "label": 0, "age": 52, "menopausal_state": 1, "tumor_location": 0 }
]
```

- `label`：0 = Luminal，1 = Non-luminal。
- `age`：连续变量，仅用对应训练折的均值/标准差标准化（配置见 `age_mean` / `age_std`）。
- `menopausal_state`：0/1 二分类变量。
- `tumor_location`：用于定位对应 MG/US 文件的标识。

> 说明：论文经 XGBoost 特征重要性筛选后，仅采用「年龄 + 绝经状态」作为模型输入的临床变量，与代码一致。`config.yaml` 中的 `diameter_mean` / `diameter_std` 字段当前未被使用，保留仅作记录。

### 预训练权重

主干网络默认使用 RadImageNet 预训练权重（ResNet-50 / DenseNet-121 / InceptionV3），需将权重放置到：

```
pretrained/RadImageNet_pytorch/{ResNet50,DenseNet121,InceptionV3}.pt
```

ViT-B/16 与 VGG16 使用 ImageNet 权重，路径分别通过环境变量 `VIT_PRETRAINED_WEIGHTS`、`VGG16_PRETRAINED_WEIGHTS` 指定。

若权重文件缺失，模型将回退到随机初始化并打印警告（不会崩溃）。

## 训练

完整模型（使用分割掩码 + 校准反馈）：

```bash
python src/train.py \
  --config_path configs/config.yaml \
  --use_seg \
  --loss_type ce \
  --modalities mg,us,clinical \
  --backbone ResNet50 \
  --num_epochs 50 \
  --num_folds 5 \
  --monitor_metric AUC_BACC \
  --threshold_strategy youden \
  --output_root calibration
```

关键参数说明：

- `--num_folds 5`：患者级 5 折交叉验证（外层 80% / 20%）。
- `--use_seg`：启用分割掩码（ROI 裁剪 + 特征门控 + ASAM + ROI 一致性损失）。论文完整模型需开启。
- `--monitor_metric`：内部验证集上用于选择最佳 epoch 的指标。
- `--threshold_strategy youden`：在**内部验证集**上按 Youden 指数确定分类阈值；该阈值在测试集上锁定使用，测试集不重新计算阈值。
- `--calibration_feedback_weight` / `--overconfidence_weight`：CCFM 反馈损失权重。
- `--ccfm_bins 5`：CCFM 校准间隙分箱数（默认 5 箱，[0.5, 1] 等宽，与论文 Eq.(8) 一致）。
- `--calibration_bins 15`：ECE/MCE 报告分箱数（[0,1] 等宽，Guo et al. 标准）。

### 消融运行

通过开关各模块复现论文消融（Table III–V）：

- 关闭分割引导：去掉 `--use_seg`
- 关闭 CCFM：`--calibration_feedback_weight 0 --overconfidence_weight 0`
- 模态消融（Table IV）：`--modalities mg` / `us` / `mg,us` / `mg,clinical` / `us,clinical`
- 主干消融（Table II）：`--backbone DenseNet121` / `InceptionV3` / `VGG16` / `ViT-B_16`
- 损失消融：`--loss_type cb_focal`（类平衡 focal loss）

## 输出说明

训练结束后在 `--output_root/{时间戳}_{model}_{backbone}_{experiment_name}_KFold5/` 下生成：

- `fold_metrics.csv`：各折最终指标（基于锁定后的外层测试集）
- `summary_95ci.csv`：五折指标的均值与 95% 置信区间半宽
- `pooled_oof_metrics.csv/.txt`：合并五折外层测试预测的指标
- `fold_{i}/`：
  - `best_model.pth`：该折最优模型（内部验证集选出）
  - `epoch_metrics.csv`：逐 epoch 的训练 / 校准 / 内部验证指标
  - `best_metrics.txt`：内部验证集上的最优指标与阈值
  - `test_metrics.txt`：外层测试集上的最终锁定指标与阈值
  - `test_*`：外层测试集的校准 / 可靠性分析产物

## 项目结构

```text
MSAC-Net-main/
├── configs/
│   ├── config.yaml          # 数据路径与超参数
│   └── clinical.json        # 临床/标签信息
├── src/
│   ├── train.py             # 主训练脚本（5 折 + 60/10/10/20 划分 + CCFM）
│   ├── dataloader/
│   │   └── load_data.py     # MyDataset
│   ├── models/
│   │   └── MSHF_roi_consistency_progressive_sparse.py  # MSAC-Net 模型
│   └── plot_calibration_dca_roi_progressive_sparse.py  # 校准/可靠性/DCA 绘图
├── docs/
│   └── roi_consistency_progressive_sparse_module.md
├── fig/                     # 论文插图（框架/可靠性/热力图，PDF）
├── portable_calibration.py  # ECE/MCE/NLL/Brier 与后处理校准方法
├── environment.yml
├── requirements.txt
└── README.md
```

## 与论文的对应关系 / 已知说明

- 本仓库当前仅包含中心 A 的 5 折交叉验证训练代码；中心 B 外部测试（论文 Table VI）需自行补充独立测试脚本。
- `docs/` 中引用的 `scripts/` 运行脚本与原始训练脚本未随本仓库提供。
- 论文表 V 的 `NLL/BRIER` 为原始值 × 100 后以百分比展示，代码输出的 NLL/Brier 为原始数值，注意量纲换算。

## 引用（Citation）

如本工作对您有帮助，请引用：

```bibtex
@article{sun2026msac,
  title   = {MSAC-Net: 用于乳腺癌 IHC 定义 Luminal 状态预测的病灶语义对齐与校准感知网络},
  author  = {Sun, Yifei and Fan, Fenglei and Jia, Junhao and Deng, Wenming and Xu, Hongxia and Wang, Changmiao and Ge, Ruiquan},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026},
  note    = {（占位，请替换为实际发表的论文信息）},
  url     = {https://github.com/yueyuanli1/MSAC-Net}
}
```

## 致谢（Acknowledgement）

This work was supported by the Zhejiang Provincial Natural Science Foundation of China (No. LY21F020017), the Guangxi Science and Technology Program (No. FN2504240022), the National Natural Science Foundation of China (No. 61702146, 62076084, U22A2033, U20A20386), the Guangxi Key R&D Project (No. AB24010167), and the Guangdong Basic and Applied Basic Research Foundation (No. 2025A1515011617).

（本项目受浙江省自然科学基金 No. LY21F020017、广西科技计划项目 No. FN2504240022、国家自然科学基金 No. 61702146/62076084/U22A2033/U20A20386、广西重点研发计划项目 No. AB24010167、广东省基础与应用基础研究基金 No. 2025A1515011617 资助。）
