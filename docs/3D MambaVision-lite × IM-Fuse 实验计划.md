# 3D MambaVision-lite × IM-Fuse 实验计划

# 3D MambaVision-lite × IM-Fuse 实验计划

## 1. 项目背景与目标

🎯
**核心假设**：在保持 IM-Fuse 的 I-MFB 融合、skip connection、multimodal Transformer、decoder、loss 和训练流程完全不变的前提下，仅将每个模态分支的 Intra-modal Transformer 替换为 **MambaVision-lite Hybrid Token Encoder**，即可稳定提升缺失模态脑肿瘤分割性能。

### 1.1 研究动机

- **MambaVision** 论文证明：在视觉 backbone 中，浅层用 CNN 做快速特征提取、深层用 Mamba mixer + 最后层用 Self-Attention 的混合模式（**Late Attention**），能在精度和吞吐量上同时超越纯 Transformer 和纯 Mamba 模型
- **IM-Fuse** 论文证明：Mamba 的选择机制和长序列建模能力可以有效应对缺失模态下的多模态融合问题，其 **Interleaved Mamba Fusion Block (I-MFB)** 在 BraTS2023 上达到 SOTA
- 但 IM-Fuse 的单模态 token encoder 仍然沿用传统的 Intra-modal Transformer，**尚未利用 MambaVision 的混合 token mixing 策略来增强单模态表征**

### 1.2 实验目标

- [ ]  验证 MambaVision-lite Hybrid Token Encoder 对单模态 bottleneck 表征的增强效果
- [ ]  在 BraTS2023 的 15 种模态缺失组合下全面评估
- [ ]  特别关注 **T1c 缺失场景下 ET（Enhancing Tumor）** 的表现
- [ ]  确保显存增长 ≤ 15%、训练吞吐下降 ≤ 20%

---

## 2. 理论基础

### 2.1 MambaVision 核心设计

MambaVision 是第一个将 Mamba 与 Transformer 结合的视觉混合 backbone，其关键发现包括：

| **设计要素** | **具体方案** | **效果** |
| --- | --- | --- |
| 宏观架构 | 4 阶段层级结构：Stage 1-2 为 CNN ResBlock，Stage 3-4 为 Mamba + Transformer | 高分辨率快速提取 + 低分辨率全局建模 |
| MambaVision Mixer | 用 **regular conv** 替代 causal conv；增加 **对称 non-SSM 分支**；两路 **concat** 后投影 | ImageNet Top-1 从 80.5% → 82.3%（+1.8%） |
| Hybrid Pattern | 每 stage 的前 N/2 层用 MV-Mixer，后 N/2 层用 Self-Attention（**Last N/2**） | 优于 Random/First N/2/Mixed 等所有其他模式 |
| Self-Attention | 多头注意力 + 窗口机制（Stage 3: window=14, Stage 4: window=7） | 捕获全局上下文和长距离依赖 |

### 2.2 IM-Fuse 核心设计

IM-Fuse 利用 Mamba 作为多模态融合机制，在 BraTS2023 上达到 SOTA：

- **Hybrid Modality-specific Encoder**：Conv Encoder → Linear Tokenization → Intra-modal Transformer
- **Bernoulli Masking**：训练时随机置零缺失模态的特征
- **I-MFB（Interleaved Mamba Fusion Block）**：交错排列各模态 token 与 learnable token，通过 Mamba 融合
- **Multimodal Transformer**：融合后的表征再经 Transformer 建模长距离依赖
- **Conv Decoder**：对称 U-Net 解码器 + skip connections
- **Loss**：Dice + Weighted CE，附加 shared-weight 辅助 decoder

### 2.3 本方案的切入点

💡
**改动范围极小**：仅替换 IM-Fuse 中每个模态分支的 Intra-modal Transformer（1 个 Transformer Block）→ MambaVision-lite Hybrid Token Encoder（MV-Mixer × k + Transformer × 1）。其余所有模块完全不变。

---

## 3. 实验矩阵设计

### 3.1 模型版本定义

| **版本** | **Intra-modal Token Encoder 结构** | **目的** | **优先级** |
| --- | --- | --- | --- |
| **Baseline** | 原始 Intra-modal Transformer × 1 | 复现 IM-Fuse 基线 | 🔴 必做 |
| **E1: Mamba-only** | MV-Mixer × 2（无 Transformer） | 验证纯 Mamba mixer 是否足够 | 🟡 次要 |
| **E2: Hybrid-1** ⭐ | MV-Mixer × 1 + Transformer × 1 | **主方案**，最低成本验证 Late Attention | 🔴 必做 |
| **E3: Hybrid-2** | MV-Mixer × 2 + Transformer × 1 | 验证增加 mixer 层数的边际收益 | 🟡 次要 |

### 3.2 实验执行顺序

1. **Phase 0**：Sanity check（加载 IM-Fuse 检查点 → 构建 Hybrid 模型 → 单卡 overfit 20 case，验证前向/反向传播正确）
2. **Phase 1**：Baseline 确认（直接使用 IM-Fuse 检查点评估，确认基线性能）
3. **Phase 2**：**E2 (Hybrid-1) 全量训练**
    - 迁移 IM-Fuse 预训练权重（共享层直接复制，原 Transformer 权重映射到 Hybrid Encoder Attention block，MV-Mixer 用 Layer Scale γ=1e-5 抑制初始扰动，Attention Block 保持 γ=1.0 以保留原始行为）
    - 全部层解冻，分组学习率，5–10 epoch linear warmup
    - 总计 300–500 epochs，Poly LR scheduler
    - ⏱ 前 30 epoch 监控 val loss + T1c-missing ET Dice，若有正面信号继续；若 50 epoch 仍无改善可叠加蒸馏初始化作为备选
4. **Phase 3**：E1 (Mamba-only) 和 E3 (Hybrid-2) 使用相同全量训练策略并行训练
5. **Phase 4**：全面评估 + 消融分析

---

## 4. 训练协议

### 4.1 数据集

| **项目** | **配置** |
| --- | --- |
| 数据集 | BraTS2023-GLI（1,251 volumes） |
| 划分 | 70% train / 10% val / 20% test（与 IM-Fuse 相同 split） |
| 模态 | FLAIR, T1, T1c, T2 |
| 分割目标 | ET（Enhancing Tumor）、TC（Tumor Core）、WT（Whole Tumor） |
| 预处理 | 与 mmFormer 一致（IM-Fuse 沿用） |
| 数据增强 | 与 mmFormer 一致 |

### 4.2 迁移学习策略（方案 A：全量训练 + 分组学习率）

🔑
**核心思路**：利用已有的 IM-Fuse 预训练检查点进行权重迁移，通过 **Layer Scale 抑制新模块初始扰动 + 分组学习率保护预训练参数**，从第一个 epoch 起全部层解冻联合训练。这避免了分阶段冻结导致的梯度路径阻断问题，让新增 MV-Mixer 模块在端到端梯度驱动下快速适配。

### 4.2.1 权重迁移与初始化

| **模块** | **初始化方式** | **说明** |
| --- | --- | --- |
| Conv Encoder × 4 | 从 IM-Fuse 检查点直接复制 | 底层特征已充分学习 |
| encode_conv / decode_conv / pos | 从 IM-Fuse 检查点直接复制 | Token 化接口不变 |
| MV-Mixer Block(s)（新增） | 随机初始化 + **Layer Scale (γ=1e-5)** | Layer Scale 确保初始输出扰动接近零 |
| Hybrid Encoder Attention Block | 从原始 Intra-modal Transformer 映射 | 复用预训练的 Self-Attention + FFN 权重 |
| I-MFB + Multimodal Transformer | 从 IM-Fuse 检查点直接复制 | 融合模块完整保留 |
| Decoder_fuse + Decoder_sep | 从 IM-Fuse 检查点直接复制 | 解码器完整保留 |

**关键设计——差异化 Layer Scale**：

- **Block 0（MV-Mixer，新增，随机初始化）**：Layer Scale γ 初始值 **1e-5**，抑制随机噪声输出，初始时该 Block 近似 identity
- **Block 1（Attention，迁移自原始 Transformer）**：**不使用 Layer Scale**（γ=1.0），保留原始 Intra-modal Transformer 的完整行为
- Block 1 的 MLP 需设为 `mlp_ratio=8`（512→4096→512），与原始 Transformer 的 FFN 维度一致，确保权重可直接迁移

这样训练开始时，MV-Mixer 是透明的（≈identity），Attention block 完整执行原始 Transformer 的功能，**pipeline 初始行为真正接近原始 IM-Fuse**。随着训练推进，MV-Mixer 的 γ 通过梯度自动增长，逐步放大其对特征流的贡献。

### 4.2.2 分组学习率设置

通过差异化学习率保护预训练参数，同时加速新模块适配（base LR = 2 × 10⁻⁴）：

| **参数组** | **学习率** | **倍率** | **说明** |
| --- | --- | --- | --- |
| MV-Mixer Blocks（新增） | 2 × 10⁻⁴ | 1× base | 随机初始化，需要最快学习速度 |
| Hybrid Encoder Attention + MLP | 2–5 × 10⁻⁵ | 0.1–0.25× base | 从原 Transformer 迁移，小心微调 |
| I-MFB + Multimodal Transformer | 1 × 10⁻⁵ | 0.05× base | 融合模块，尽量保护 |
| Conv Encoder | 5 × 10⁻⁶ | 0.025× base | 底层特征已充分学习 |
| Conv Decoder + Aux Decoder | 1 × 10⁻⁵ | 0.05× base | 需适应新表征，但不宜过快 |
| 其他（encode_conv, decode_conv, pos 等） | 1 × 10⁻⁵ | 0.05× base | 接口层，保持稳定 |

### 4.2.3 Linear Warmup + LR Scheduler

- **Linear Warmup（5–10 epochs）**：所有参数组的学习率从目标值 × 0.01 线性升到目标值，为预训练模块提供平滑过渡期
- **Poly Scheduler**：warmup 结束后全程使用 `lr × (1 - epoch / total_epochs)^0.9`
- Optimizer 使用 AdamW（weight_decay=3e-5）

**总训练 epoch：300–500**（对比原始从头训练 1000 epochs，**节省 50%–70%**）

### 4.2.4 备选策略：蒸馏初始化

若方案 A 在 50 epoch 后仍无明显改善，可叠加蒸馏初始化作为增强：

1. **蒸馏阶段**（~50–100 epochs）：冻结原始 IM-Fuse 作为 Teacher，用 token-level MSE loss 训练 Hybrid Token Encoder 匹配原始 Intra-modal Transformer 的输出
2. **全量训练阶段**：将蒸馏后的 Hybrid Encoder 作为更好的初始化，再按方案 A 执行全量训练

此策略仅在方案 A 遇到困难时启用，不作为默认流程。

---

## 5. 风险控制

### 5.1 风险：收益太小

**原因**：仅改动单模态 token encoder，未触及 fusion 主体

**控制**：

- 前 30 epoch 即可观察 val 曲线和 T1c-missing ET Dice 是否有正面信号
- 若 50 epoch 后仍无改善，启用备选策略（蒸馏初始化 + 全量训练，见 4.2.4）
- 定期保存 checkpoint，通过 Dice 曲线分析定位瓶颈是否在新模块

### 5.2 风险：Mamba branch 训练不稳定

**原因**：3D 医学分割中 Mamba 在高分辨率/长序列下可能不稳定

**控制**：

- 第一版仅上 MV-Mixer × 1（Hybrid-1）
- 差异化 Layer Scale（MV-Mixer γ=1e-5，Attention γ=1.0）+ linear warmup 确保 MV-Mixer 初期影响极小且不破坏已迁移的 Attention 行为，训练更稳定
- 使用残差连接、LayerNorm、drop_path=0.1 的保守设置
- **先单卡 overfit 20 case** 确认训练稳定后再启动全量训练
- 定期保存 checkpoint，若出现训练崩溃可回退到最近稳定点继续

### 5.3 风险：预训练特征前期被扰动

**原因**：全部层同时训练，新增 MV-Mixer 的随机输出可能在初期干扰预训练特征流

**控制**：

- 差异化 Layer Scale：MV-Mixer Block γ=1e-5（抑制随机噪声），Attention Block γ=1.0（保留原始 Transformer 行为），确保 pipeline 初始行为真正接近原始 IM-Fuse
- 分组学习率：Conv Encoder 和融合模块使用极低 LR（0.025×–0.05× base），减少被扰动风险
- Linear warmup（5–10 epochs）提供平滑过渡
- 若前 30 epoch 出现预训练特征退化迹象（全模态 Dice 下降），可进一步降低底层模块 LR

### 5.4 风险：关键缺失场景退化

**原因**：缺失模态补偿不仅由 backbone 决定

**控制**：

- 必须单独监控 missing T1c → ET Dice
- 若此处退化，首选不是继续改 backbone，而是后续考虑加轻量补偿项