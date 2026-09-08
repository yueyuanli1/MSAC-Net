# ROI 一致性渐进式稀疏融合模块说明

本文档说明新增的 ROI consistency 版本。该版本是在现有 ViewPE + progressive sparse cross-attention 实验基础上扩展出来的，**不修改原有文件**。

## 新增文件

- `src/models/MSHF_roi_consistency_progressive_sparse.py`
- `src/train.py`
- `scripts/kfold_roi_consistency_progressive_sparse_toggle_confidence_70_10_20.sh`
- `docs/roi_consistency_progressive_sparse_module.md`

原始的模型、训练脚本和运行脚本都保持不变。

## 设计动机

原有的分割辅助分类主要把分割 mask 用在两个位置：

```text
分割 mask
    ↓
ROI 裁剪
    ↓
backbone feature map
    ↓
features * (1 + mask_down)
```

也就是说，原有 mask 主要用于输入级 ROI 聚焦和 feature map 级病灶区域增强。

新增版本保留这部分逻辑，但进一步把 mask 从 feature-level guidance 延续到 token-level supervision。这样三个创新点可以形成一个完整故事：

```text
分割先验 → 病灶中心 token 表征
ViewPE + 渐进式稀疏交叉注意力 → 跨模态证据融合
置信度校准反馈 → 可靠性训练
```

## 新增模块

新增模块可以称为：

```text
分割先验引导的跨模态 ROI 一致性增强模块
```

英文可写为：

```text
Segmentation-prior Guided Cross-modal ROI Consistency Module
```

在两层 progressive sparse cross-attention 之后，MG 和 US 仍然是两条分开的 token 序列：

```text
updated MG tokens: B, N_mg, C
updated US tokens: B, N_us, C
```

新增模型同时保留对应的 token-level ROI mask：

```text
MG ROI mask: B, N_mg
US ROI mask: B, N_us
```

然后只聚合 ROI 区域 tokens：

```text
mg_roi_feat = sum(mg_token_i * mg_roi_mask_i) / sum(mg_roi_mask_i)
us_roi_feat = sum(us_token_i * us_roi_mask_i) / sum(us_roi_mask_i)
```

得到两个病灶级语义向量：

```text
mg_roi_feat: MG 病灶区域表征
us_roi_feat: US 病灶区域表征
```

随后通过两个投影层把它们映射到共享语义空间，并计算余弦一致性损失：

```text
L_roi_consistency = 1 - cosine(project_mg(mg_roi_feat), project_us(us_roi_feat))
```

训练时总损失变为：

```text
L_total = L_cls + L_aux + L_calibration_feedback + lambda_roi * L_roi_consistency
```

其中 `lambda_roi` 对应脚本参数：

```text
ROI_CONSISTENCY_WEIGHT
```

## 相比当前 ViewPE Progressive Sparse 版本增加了什么

保留的内容包括：

- dataloader 中基于分割 mask 的 ROI 裁剪
- 模型中的 feature map ROI gating：`features * (1 + mask_down)`
- MLO/CC 独立 ViewPE
- MG-US 渐进式非对称双向稀疏交叉注意力
- clinical MetaFusion
- confidence calibration feedback

新增的内容包括：

- `forward_one_stream()` 额外返回 `roi_token_mask`
- MLO 和 CC 的 ROI token mask 拼接成 MG ROI mask
- `ProgressiveSparseCrossModalFusion.forward()` 额外接收 `mg_roi_mask` 和 `us_roi_mask`
- 在 sparse cross-attention blocks 之后，对 MG/US 的 ROI tokens 分别做 pooling
- 返回训练用的 `roi_consistency_loss`
- 训练脚本新增 `--roi_consistency_weight`
- shell 脚本新增 `ROI_CONSISTENCY_WEIGHT`

## 为什么这不是简单的小改动

原有 mask 的生命周期大致是：

```text
mask → ROI crop → feature gating → tokens → 后续不再显式使用 mask
```

新增模块把 mask 继续延伸到 token 层：

```text
mask
    ↓
ROI crop + feature gating
    ↓
tokens + roi_token_mask
    ↓
progressive sparse cross-attention
    ↓
ROI token pooling
    ↓
MG/US ROI consistency loss
    ↓
fusion + classification
```

因此，分割先验不再只是局部增强信号，而是进一步成为跨模态病灶语义对齐的监督信号。

可以把这个变化概括为：

```text
从隐式 ROI 特征增强，扩展为显式 ROI token 语义一致性约束。
```

## 与三个创新点的关系

这套版本可以把三个创新点串成一个整体框架：

```text
1. 分割先验提供病灶区域定位，并生成 token-level ROI mask。
2. ViewPE 区分 MLO/CC patch tokens，增强 MG 视图感知表征。
3. 渐进式稀疏交叉注意力完成 MG-US 由粗到细的 token 级交互。
4. ROI consistency 约束交互后的 MG/US 病灶表征在语义空间中保持一致。
5. 置信度校准反馈降低过置信错误预测，提高可靠性。
```

这样可以形成一条方法主线：

```text
看准病灶 → 融合证据 → 对齐语义 → 校准信心
```

## 推荐运行方式

默认运行：

```bash
bash scripts/kfold_roi_consistency_progressive_sparse_toggle_confidence_70_10_20.sh
```

推荐完整模块运行：

```bash
USE_SEG_GUIDANCE=1 USE_CONFIDENCE_MODULE=1 ROI_CONSISTENCY_WEIGHT=0.05 \
bash scripts/kfold_roi_consistency_progressive_sparse_toggle_confidence_70_10_20.sh
```

说明：

- `USE_SEG_GUIDANCE=1` 时才会读取分割 mask，ROI consistency loss 才有实际作用。
- 如果 `USE_SEG_GUIDANCE=0`，没有 ROI token mask，`roi_consistency_loss` 会自动为 0。
- `ROI_CONSISTENCY_WEIGHT` 控制新增 ROI 一致性分支的损失权重。

## 论文方法部分可用表述

可以这样描述该模块：

```text
在原有分割辅助分类的基础上，本文进一步提出分割先验引导的跨模态 ROI 一致性增强模块。该模块将分割 mask 从输入级 ROI 引导扩展到 token 级跨模态语义约束：首先根据分割先验保留 MG 与 US 的 ROI token mask，随后在渐进式双向稀疏交叉注意力之后，分别聚合 MG 和 US 的病灶区域 token 表征，并通过余弦一致性损失约束两种模态在病灶语义空间中的一致性。该设计使分割先验不仅用于指导模型关注病灶区域，还进一步参与跨模态融合后的语义对齐，从而增强病灶表征稳定性和预测可靠性。
```
