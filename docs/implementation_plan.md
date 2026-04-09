# MambaVision-lite × IM-Fuse 可行性分析与实施文档

## 1. 可行性结论

**整体结论：方案可行**，但存在若干需要特别处理的细节偏差。以下逐项分析。

---

## 2. 架构对照分析

### 2.1 IM-Fuse Intra-modal Transformer 精确结构

每个模态分支的 bottleneck token encoder 由以下部分组成（`depth=1`，即单个 Transformer Block）：

```
输入: (B, 128, 8, 8, 8)  ← Conv Encoder 输出
  ↓ encode_conv: Conv3d(128 → 512, k=1)
  ↓ reshape: (B, 512, 8, 8, 8) → (B, 512, 512)  // 512 tokens × 512 dim
  ↓ Transformer(embedding_dim=512, depth=1, heads=8, mlp_dim=4096):
      for j in range(1):
          x = x + pos                              // 加性位置编码
          x = Residual(PreNormDrop(SelfAttention))  // LN → Attn → Dropout + Residual
          x = Residual(PreNorm(FFN))                // LN → FFN(512→4096→512) + Residual
  ↓ reshape: (B, 512, 512) → (B, 512, 8, 8, 8)
```

**关键参数：**
| 参数 | 值 |
|---|---|
| `embedding_dim` | 512 |
| `depth` | 1 |
| `heads` | 8 |
| `mlp_dim` | 4096 (`mlp_ratio = 8`) |
| `dropout_rate` | 0.1 |
| 位置编码 | `nn.Parameter(zeros(1, 512, 512))` 可学习，**加性（additive）** |
| 残差模式 | Pre-Norm（LN 在前） |

**⚠️ 重要发现：IM-Fuse Transformer 中位置编码在每个 depth 循环内都会加一次。** 即 `x = x + pos` 在 attention 之前执行。在 `depth=1` 时这只发生一次，但若考虑 Hybrid Encoder 替换，需要决定 MV-Mixer Block 是否也需要位置编码。

### 2.2 MambaVision Block 精确结构

MambaVision 的 `Block` 类统一封装了 MV-Mixer 和 Attention 两种 mixer：

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4., layer_scale=None, ...):
        self.norm1 = LayerNorm(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(dim, num_heads, ...)   # Self-Attention
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio)
        self.gamma_1 = nn.Parameter(layer_scale * ones(dim)) if layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * ones(dim)) if layer_scale else 1
    
    def forward(self, x):
        x = x + drop_path(gamma_1 * mixer(norm1(x)))
        x = x + drop_path(gamma_2 * mlp(norm2(x)))
```

**MambaVisionMixer 参数（在 Block 中实例化时）：**
| 参数 | 值 |
|---|---|
| `d_state` | **8**（非论文默认 16） |
| `d_conv` | **3**（非标准 Mamba 的 4） |
| `expand` | **1**（非标准 Mamba 的 2） |
| `d_inner` | `expand * d_model = 1 × dim` |
| SSM 分支 | `d_inner // 2`（即 `dim // 2`） |
| conv1d | `padding='same'`，非因果 |

**MambaVision Attention 参数：**
| 参数 | 值 |
|---|---|
| `qkv_bias` | True（视 config） |
| 位置编码 | **无显式位置编码**（依赖窗口分割的隐式位置） |
| dropout | 0.0 |

### 2.3 IM-Fuse Transformer vs MambaVision Block 关键差异

| 特性 | IM-Fuse Transformer | MambaVision Block |
|---|---|---|
| **Pre-Norm 结构** | `LN → Attn → Dropout → +Residual` | `x + drop_path(γ * mixer(LN(x)))` |
| **FFN** | `Linear(512→4096) → GELU → Dropout → Linear(4096→512) → Dropout` | `Linear(dim→hidden) → GELU → Dropout → Linear(hidden→dim) → Dropout`（timm Mlp） |
| **mlp_ratio** | **8**（512→4096） | **4**（默认） |
| **位置编码** | 可学习加性 `pos`，每个 depth 迭代加一次 | **无** |
| **Dropout** | attention=0.1, FFN=0.1 | 0.0（attn_drop, proj_drop） |
| **Layer Scale** | **无** | 可选（γ 参数） |
| **Drop Path** | **无** | 可选 |
| **Attention qkv** | `nn.Linear(dim, dim*3, bias=False)` | `nn.Linear(dim, dim*3, bias=True/False)` |

---

## 3. 逐项可行性检查

### 3.1 ✅ 权重迁移——Conv Encoder × 4

**完全可行。** 四个模态编码器 `flair_encoder`, `t1ce_encoder`, `t1_encoder`, `t2_encoder` 结构不变，key 名称不变，可直接 `load_state_dict` 映射。

### 3.2 ✅ 权重迁移——encode_conv / decode_conv / pos

**完全可行。** 这些是 `Conv3d(128→512, k=1)` 和 `nn.Parameter`，结构和名称不变。

### 3.3 ⚠️ 权重迁移——Intra-modal Transformer → Hybrid Encoder Attention Block

**可行但需仔细处理结构差异。**

原始 IM-Fuse 的 Transformer 内部结构（以 `flair_transformer` 为例）：
```
flair_transformer.cross_attention_list.0.fn.fn       → SelfAttention
flair_transformer.cross_attention_list.0.fn.norm      → LayerNorm
flair_transformer.cross_ffn_list.0.fn.fn              → FeedForward
flair_transformer.cross_ffn_list.0.fn.norm            → LayerNorm
```

展开 state_dict key：
```
flair_transformer.cross_attention_list.0.fn.norm.weight       # LN (512,)
flair_transformer.cross_attention_list.0.fn.norm.bias         # LN (512,)
flair_transformer.cross_attention_list.0.fn.fn.qkv.weight     # (1536, 512)
flair_transformer.cross_attention_list.0.fn.fn.proj.weight     # (512, 512)
flair_transformer.cross_attention_list.0.fn.fn.proj.bias       # (512,)
flair_transformer.cross_ffn_list.0.fn.norm.weight              # LN (512,)
flair_transformer.cross_ffn_list.0.fn.norm.bias                # LN (512,)
flair_transformer.cross_ffn_list.0.fn.fn.net.0.weight          # FFN Linear1 (4096, 512)
flair_transformer.cross_ffn_list.0.fn.fn.net.0.bias            # (4096,)
flair_transformer.cross_ffn_list.0.fn.fn.net.3.weight          # FFN Linear2 (512, 4096)
flair_transformer.cross_ffn_list.0.fn.fn.net.3.bias            # (512,)
```

而 MambaVision Block（用作 Attention block）的 state_dict key：
```
block.norm1.weight / bias                    # LN
block.mixer.qkv.weight / bias               # (1536, 512)
block.mixer.proj.weight / bias               # (512, 512)
block.norm2.weight / bias                    # LN
block.mlp.fc1.weight / bias                  # (hidden, 512)
block.mlp.fc2.weight / bias                  # (512, hidden)
block.gamma_1                                # Layer Scale (可选)
block.gamma_2                                # Layer Scale (可选)
```

**差异与风险：**

1. **QKV bias**：IM-Fuse `SelfAttention` 的 `qkv_bias=False`（无 bias），MambaVision `Attention` 的 `qkv_bias` 可配置。**实施时必须设为 `qkv_bias=False`**，否则形状不匹配。

2. **mlp_ratio**：IM-Fuse FFN 是 `512→4096→512`（ratio=8），但 MambaVision 默认 `mlp_ratio=4`。**计划文档已正确识别这一点**（4.2.1 节提到 `mlp_ratio=8`），实施时需显式设置。

3. **Dropout**：IM-Fuse 使用 `dropout_rate=0.1`（attn 和 FFN 层内部均有 dropout），MambaVision 的 Attention 和 Mlp 分别通过 `attn_drop` 和 `proj_drop` 控制。Hybrid Encoder 的 Attention Block 应保留 0.1 的 dropout 以匹配原始行为。但注意：timm 的 `Mlp` 类 dropout 放的位置与 IM-Fuse FeedForward 略有不同（timm 在每个 Linear 后，IM-Fuse 也是，但 timm 用一个 `drop` 参数统一控制）。

4. **位置编码处理**：IM-Fuse 在 Transformer forward 中 `x = x + pos` 在每个 depth 循环最前面执行。MambaVision Block 没有位置编码。**对于 Hybrid Encoder，需要在 Attention Block 外部或第一个 Block 入口处注入位置编码**。可选方案：
   - 在 Hybrid Encoder 的 forward 开头加 `x = x + pos`（最简单，与原始行为等价）
   - 在 MV-Mixer Block 前加 pos（但 Mamba 本身对位置不敏感，pos 对其作用可能有限）

5. **Pre-Norm + Residual 封装差异**：IM-Fuse 使用 `Residual(PreNormDrop(SelfAttention))`，即 `x_out = x + Dropout(SelfAttention(LN(x)))` 形式。MambaVision 使用 `x + drop_path(γ * mixer(LN(x)))` 形式。两者在 Layer Scale = 1.0 且 drop_path = 0 时**语义等价**（除了 IM-Fuse 有 attention 内部 dropout 而 MambaVision 通过 attn_drop 参数控制）。

### 3.4 ✅ 权重迁移——I-MFB + Multimodal Transformer + Decoder

**完全可行。** 这些模块的结构和参数名在 MV-IM-Fuse 中保持不变，可直接复制。

### 3.5 ⚠️ MV-Mixer 的 3D 适配

**这是最重要的技术风险。**

MambaVision 的 `MambaVisionMixer` 是为 **2D 图像**设计的：
- 输入 `(B, L, D)` 中 L 来自 2D spatial flatten（H×W）
- `conv1d_x` 和 `conv1d_z` 是 1D depthwise conv，沿 token 序列维度操作
- 内部使用 Mamba 的 `selective_scan_fn`

对于 IM-Fuse 的 3D bottleneck tokens：
- 输入 `(B, 512, 512)` 中 512 tokens 来自 8×8×8 的 3D spatial flatten
- Token 顺序是将 3D 体素 flatten 为 1D 序列

**MambaVisionMixer 可以直接用于 3D tokens**，因为：
1. `selective_scan_fn` 对序列的空间来源无感知，它只处理 `(B, D, L)` 形状的序列
2. `conv1d` 沿序列维度做局部信息交换，在 flatten 后的 3D 序列上仍然有意义
3. 原始 IM-Fuse 的 I-MFB 中已经使用了标准 Mamba 处理 3D flatten 后的 token，证明这种方式在此任务中可行

但需要注意：**与 2D 不同，3D flatten 后的邻居关系更差**（raster scan 将 3D 立方打散为 1D），这也是 IM-Fuse 论文选择用 Mamba 做融合（而非单模态编码）的原因之一。MV-Mixer 的非因果 conv1d + 对称分支设计一定程度上缓解了这个问题（不依赖因果顺序），但效果需要实验验证。

### 3.6 ⚠️ `selective_scan_fn` 依赖

MambaVisionMixer 直接调用了 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`，而 IM-Fuse 通过 `mamba_ssm.Mamba` 高级 API 使用 Mamba。两者底层相同（均来自 `mamba_ssm` 包），但需要确保版本兼容。**当前 MV-IM-Fuse 已安装 `mamba_ssm`**（从 `IMFuse.py` 的 `from mamba_ssm import Mamba` 可见），只需额外确认 `selective_scan_fn` 可正常导入。

### 3.7 ✅ 训练流程兼容性

- **LR Scheduler**：已有 Poly scheduler（`lr_scheduler.py` 中 `LR_Scheduler`），但它只修改 `optimizer.param_groups[0]`。分组学习率需要修改为遍历所有 param_groups。
- **Optimizer**：当前使用 `RAdam`，计划使用 `AdamW`。这是一个有意的变更。
- **Loss**：Dice + Weighted CE，完全不变。
- **Training loop**：`train_poly.py` 结构支持 `--pretrain` 和 `--resume`，可以扩展支持权重迁移。

---

## 4. 详细实施方案

### 4.1 Step 1：创建 `MambaVisionMixer3D` 模块

从 MambaVision 仓库提取 `MambaVisionMixer`，适配为 3D 使用（实际上无需修改，因为它已经是处理 `(B, L, D)` 序列的）。

**文件**：`mv_mixer.py`（新建）

```python
"""
3D MambaVision Mixer — 从 MambaVision 仓库提取并适配。
实际上 MambaVisionMixer 本身就是序列维处理，无需修改。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat


class MambaVisionMixer(nn.Module):
    """原封不动地从 MambaVision 提取自 mamba_vision.py"""
    def __init__(self, d_model, d_state=8, d_conv=3, expand=1, 
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        # dt 初始化
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.conv1d_x = nn.Conv1d(
            self.d_inner // 2, self.d_inner // 2,
            bias=conv_bias, kernel_size=d_conv,
            groups=self.d_inner // 2,
        )
        self.conv1d_z = nn.Conv1d(
            self.d_inner // 2, self.d_inner // 2,
            bias=conv_bias, kernel_size=d_conv,
            groups=self.d_inner // 2,
        )

    def forward(self, hidden_states):
        """hidden_states: (B, L, D)"""
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias,
                            padding='same', groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias,
                            padding='same', groups=self.d_inner // 2))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(x, dt, A, B, C, self.D.float(),
                              z=None, delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True, return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
```

### 4.2 Step 2：创建 `HybridTokenEncoder` 模块

这是核心新增模块，替代原始的 `Transformer` 类。

**文件**：`hybrid_encoder.py`（新建）

```python
"""
Hybrid Token Encoder: MV-Mixer × k + Transformer × 1
替代 IM-Fuse 原始的 Intra-modal Transformer。
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch.cuda.amp import autocast


class HybridMixerBlock(nn.Module):
    """MV-Mixer Block，带 Layer Scale 和 Drop Path。"""

    def __init__(self, dim, d_state=8, d_conv=3, expand=1,
                 mlp_ratio=8., drop=0.1, drop_path=0.1,
                 layer_scale_init=1e-5):
        super().__init__()
        from mv_mixer import MambaVisionMixer

        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaVisionMixer(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Layer Scale
        self.gamma_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma_2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class HybridAttentionBlock(nn.Module):
    """
    Attention Block，复用 IM-Fuse 原始 Transformer 权重。
    结构与 IM-Fuse 的 Transformer(depth=1) 语义对齐：
      x = x + pos
      x = Residual(PreNormDrop(SelfAttention))
      x = Residual(PreNorm(FFN))
    """

    def __init__(self, dim, heads=8, mlp_ratio=8., dropout_rate=0.1):
        super().__init__()
        # 直接复用 IM-Fuse 的组件结构
        from IMFuse import SelfAttention, FeedForward, PreNormDrop, PreNorm, Residual
        self.attn_block = Residual(
            PreNormDrop(dim, dropout_rate, SelfAttention(dim, heads=heads, dropout_rate=dropout_rate))
        )
        self.ffn_block = Residual(
            PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), dropout_rate))
        )

    def forward(self, x, pos):
        x = x + pos
        x = self.attn_block(x)
        x = self.ffn_block(x)
        return x


class HybridTokenEncoder(nn.Module):
    """
    Hybrid Token Encoder：MV-Mixer × num_mixer_blocks + Attention × 1

    取代原始 Transformer(embedding_dim, depth=1, heads, mlp_dim)。
    """

    def __init__(self, dim=512, heads=8, mlp_ratio=8., dropout_rate=0.1,
                 num_mixer_blocks=1, mixer_layer_scale_init=1e-5,
                 mixer_d_state=8, mixer_d_conv=3, mixer_expand=1,
                 drop_path=0.1):
        super().__init__()

        self.mixer_blocks = nn.ModuleList([
            HybridMixerBlock(
                dim=dim, d_state=mixer_d_state, d_conv=mixer_d_conv,
                expand=mixer_expand, mlp_ratio=mlp_ratio, drop=dropout_rate,
                drop_path=drop_path, layer_scale_init=mixer_layer_scale_init,
            )
            for _ in range(num_mixer_blocks)
        ])

        self.attn_block = HybridAttentionBlock(
            dim=dim, heads=heads, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate
        )

    def forward(self, x, pos):
        # MV-Mixer blocks（不使用位置编码，Mamba 对位置不敏感）
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        # Attention block（使用位置编码，与原始 Transformer 行为一致）
        x = self.attn_block(x, pos)
        return x
```

### 4.3 Step 3：修改 `IMFuse` 模型类

在 `IMFuse` 类中，将 4 个 `self.{modal}_transformer` 替换为 `HybridTokenEncoder`。

**修改文件**：`IMFuse.py`

```python
# 替换 IntraFormer 部分
# 原始：
# self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
# ...

# 新：
from hybrid_encoder import HybridTokenEncoder

self.flair_transformer = HybridTokenEncoder(
    dim=transformer_basic_dims, heads=num_heads, mlp_ratio=mlp_dim/transformer_basic_dims,
    num_mixer_blocks=num_mixer_blocks,  # E2=1, E3=2
)
# ... 对 t1ce, t1, t2 同样替换
```

`forward()` 方法中的调用签名保持不变：
```python
flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
```

### 4.4 Step 4：权重迁移脚本

**文件**：`scripts/transfer_weights.py`（新建）

核心逻辑：

```python
def transfer_weights(imfuse_ckpt_path, hybrid_model):
    """
    从 IM-Fuse checkpoint 迁移权重到 Hybrid 模型。
    """
    ckpt = torch.load(imfuse_ckpt_path, map_location='cpu')
    src_sd = ckpt['state_dict']

    # 去除 DataParallel 的 'module.' 前缀
    src_sd = {k.replace('module.', ''): v for k, v in src_sd.items()}

    tgt_sd = hybrid_model.state_dict()
    transferred = {}
    skipped = []

    for key, value in src_sd.items():
        # 判断是否属于 IntraFormer Transformer
        is_intra_transformer = any(
            key.startswith(f'{m}_transformer') for m in ['flair', 't1ce', 't1', 't2']
        )

        if is_intra_transformer:
            # 映射 Transformer 权重到 HybridTokenEncoder 的 attn_block
            new_key = remap_transformer_key(key)
            if new_key and new_key in tgt_sd:
                if value.shape == tgt_sd[new_key].shape:
                    transferred[new_key] = value
                else:
                    skipped.append((key, new_key, 'shape_mismatch'))
            else:
                skipped.append((key, new_key, 'not_found_in_target'))
        elif key in tgt_sd:
            # 直接复制
            if value.shape == tgt_sd[key].shape:
                transferred[key] = value
            else:
                skipped.append((key, key, 'shape_mismatch'))
        else:
            skipped.append((key, None, 'not_in_target'))

    # 加载迁移的权重
    tgt_sd.update(transferred)
    hybrid_model.load_state_dict(tgt_sd, strict=False)

    return transferred, skipped


def remap_transformer_key(old_key):
    """
    将 IM-Fuse Transformer state_dict key 映射到 HybridTokenEncoder.attn_block key。

    原始 key 格式:
      {modal}_transformer.cross_attention_list.0.fn.norm.weight
      {modal}_transformer.cross_attention_list.0.fn.fn.qkv.weight
      {modal}_transformer.cross_attention_list.0.fn.fn.proj.weight
      {modal}_transformer.cross_attention_list.0.fn.fn.proj.bias (注意：无 qkv.bias)
      {modal}_transformer.cross_ffn_list.0.fn.norm.weight
      {modal}_transformer.cross_ffn_list.0.fn.fn.net.0.weight   (Linear1)
      {modal}_transformer.cross_ffn_list.0.fn.fn.net.0.bias
      {modal}_transformer.cross_ffn_list.0.fn.fn.net.3.weight   (Linear2)
      {modal}_transformer.cross_ffn_list.0.fn.fn.net.3.bias

    目标 key 格式（HybridAttentionBlock 复用 IM-Fuse 原始组件）:
      {modal}_transformer.attn_block.attn_block.fn.norm.weight
      {modal}_transformer.attn_block.attn_block.fn.fn.qkv.weight
      ...
      {modal}_transformer.attn_block.ffn_block.fn.norm.weight
      {modal}_transformer.attn_block.ffn_block.fn.fn.net.0.weight
      ...
    """
    import re
    # 提取模态前缀和剩余路径
    m = re.match(r'^(\w+_transformer)\.(.+)$', old_key)
    if not m:
        return None

    prefix = m.group(1)
    suffix = m.group(2)

    # cross_attention_list.0.xxx → attn_block.attn_block.xxx
    suffix = suffix.replace('cross_attention_list.0.', 'attn_block.attn_block.')
    # cross_ffn_list.0.xxx → attn_block.ffn_block.xxx
    suffix = suffix.replace('cross_ffn_list.0.', 'attn_block.ffn_block.')

    return f'{prefix}.{suffix}'
```

### 4.5 Step 5：分组学习率训练脚本

**文件**：`train_hybrid.py`（新建，基于 `train_poly.py` 修改）

关键修改部分：

```python
def build_param_groups(model, base_lr=2e-4):
    """
    分组学习率设置。
    """
    mixer_params = []
    attn_params = []
    imfb_params = []
    conv_enc_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'mixer_blocks' in name:
            mixer_params.append(param)
        elif 'attn_block' in name and any(f'{m}_transformer' in name for m in ['flair','t1ce','t1','t2']):
            attn_params.append(param)
        elif 'mamba_fusion' in name or 'multimodal_transformer' in name:
            imfb_params.append(param)
        elif '_encoder' in name:
            conv_enc_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': mixer_params,    'lr': base_lr * 1.0,   'name': 'mv_mixer'},
        {'params': attn_params,     'lr': base_lr * 0.1,   'name': 'hybrid_attn'},
        {'params': imfb_params,     'lr': base_lr * 0.05,  'name': 'imfb'},
        {'params': conv_enc_params, 'lr': base_lr * 0.025, 'name': 'conv_encoder'},
        {'params': decoder_params,  'lr': base_lr * 0.05,  'name': 'decoder'},
        {'params': other_params,    'lr': base_lr * 0.05,  'name': 'other'},
    ]
    return param_groups
```

**LR Scheduler 修改**（支持多 param_group + warmup）：

```python
class LR_Scheduler_Grouped(object):
    def __init__(self, num_epochs, warmup_epochs=10, warmup_factor=0.01):
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor

    def __call__(self, optimizer, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * epoch / self.warmup_epochs
        else:
            # Poly decay
            factor = (1 - (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)) ** 0.9

        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * factor

        return factor
```

---

## 5. 权重映射详细对照表

### 5.1 直接复制（key 不变）

| 模块 | key 前缀 | 参数量（约） |
|---|---|---|
| 4× Conv Encoder | `{modal}_encoder.*` | 4 × ~0.6M |
| 4× encode_conv | `{modal}_encode_conv.*` | 4 × 65K |
| 4× decode_conv | `{modal}_decode_conv.*` | 4 × 65K |
| 4× pos | `{modal}_pos` | 4 × 262K |
| fused_pos | `fused_pos` | 262K |
| I-MFB | `mamba_fusion_layer.*` | ~2M |
| Multimodal Transformer | `multimodal_transformer.*` | ~5M |
| multimodal_decode_conv | `multimodal_decode_conv.*` | 262K |
| Tokenize layers | `tokenize.*` | 无参数 |
| Mamba Fusion Layers | `mamba_fusion_layers.*` | ~15M |
| Decoder_fuse | `decoder_fuse.*` | ~0.5M |
| Decoder_sep | `decoder_sep.*` | ~0.5M |
| MaskModal | `masker.*` | 无参数 |

### 5.2 需要映射（Transformer → HybridAttentionBlock）

对每个模态（flair/t1ce/t1/t2），深度均为 1，仅 index `0`：

| IM-Fuse key (source) | Hybrid key (target) | Shape |
|---|---|---|
| `{m}_transformer.cross_attention_list.0.fn.norm.weight` | `{m}_transformer.attn_block.attn_block.fn.norm.weight` | (512,) |
| `{m}_transformer.cross_attention_list.0.fn.norm.bias` | `{m}_transformer.attn_block.attn_block.fn.norm.bias` | (512,) |
| `{m}_transformer.cross_attention_list.0.fn.fn.qkv.weight` | `{m}_transformer.attn_block.attn_block.fn.fn.qkv.weight` | (1536, 512) |
| `{m}_transformer.cross_attention_list.0.fn.fn.proj.weight` | `{m}_transformer.attn_block.attn_block.fn.fn.proj.weight` | (512, 512) |
| `{m}_transformer.cross_attention_list.0.fn.fn.proj.bias` | `{m}_transformer.attn_block.attn_block.fn.fn.proj.bias` | (512,) |
| `{m}_transformer.cross_ffn_list.0.fn.norm.weight` | `{m}_transformer.attn_block.ffn_block.fn.norm.weight` | (512,) |
| `{m}_transformer.cross_ffn_list.0.fn.norm.bias` | `{m}_transformer.attn_block.ffn_block.fn.norm.bias` | (512,) |
| `{m}_transformer.cross_ffn_list.0.fn.fn.net.0.weight` | `{m}_transformer.attn_block.ffn_block.fn.fn.net.0.weight` | (4096, 512) |
| `{m}_transformer.cross_ffn_list.0.fn.fn.net.0.bias` | `{m}_transformer.attn_block.ffn_block.fn.fn.net.0.bias` | (4096,) |
| `{m}_transformer.cross_ffn_list.0.fn.fn.net.3.weight` | `{m}_transformer.attn_block.ffn_block.fn.fn.net.3.weight` | (512, 4096) |
| `{m}_transformer.cross_ffn_list.0.fn.fn.net.3.bias` | `{m}_transformer.attn_block.ffn_block.fn.fn.net.3.bias` | (512,) |

共 4 个模态 × 11 个参数 = **44 个参数需要 key 重映射**。

### 5.3 随机初始化（MV-Mixer 新增参数）

每个模态的 `HybridMixerBlock` 中（以 `dim=512, expand=1` 为例）：

| 参数 | Shape | 说明 |
|---|---|---|
| `norm1.weight/bias` | (512,) | LayerNorm |
| `mixer.in_proj.weight` | (512, 512) | 无 bias |
| `mixer.x_proj.weight` | (256, 48) | dt_rank=32, d_state=8 |
| `mixer.dt_proj.weight/bias` | (256, 32) / (256,) | |
| `mixer.A_log` | (256, 8) | |
| `mixer.D` | (256,) | |
| `mixer.out_proj.weight` | (512, 512) | |
| `mixer.conv1d_x.weight/bias` | (256, 1, 3) / (256,) | depthwise |
| `mixer.conv1d_z.weight/bias` | (256, 1, 3) / (256,) | depthwise |
| `norm2.weight/bias` | (512,) | |
| `mlp.fc1.weight/bias` | (4096, 512) / (4096,) | |
| `mlp.fc2.weight/bias` | (512, 4096) / (512,) | |
| `gamma_1` | (512,) | Layer Scale = 1e-5 |
| `gamma_2` | (512,) | Layer Scale = 1e-5 |

约 **~5.5M 新增参数/模态**，4 个模态共 **~22M 新增参数**。

---

## 6. 风险评估与偏差分析

### 6.1 🔴 风险：MV-Mixer 的新增参数量可能超出预期

**问题**：每个 MV-Mixer Block 新增约 5.5M 参数（主要来自 `mlp_ratio=8` 的 MLP，仅 MLP 就 4096×512×2 ≈ 4.2M）。4 个模态合计新增 ~22M 参数。

**影响**：原始 IM-Fuse 总参数约 30-40M，新增 22M 意味着参数量增加约 50-70%。这可能导致：
- 显存增长超出计划的 15% 限制
- 不一定 overfit，但训练收敛更慢

**缓解方案**：
- **可选 A**：MV-Mixer Block 使用较小的 `mlp_ratio=4`（而非 8），将 MLP 参数减半（~11M 新增）。代价是 MV-Mixer 的 MLP 和 Attention 的 MLP 维度不一致。
- **可选 B**：使用 `expand=1, mlp_ratio=4` 的 lean 配置（与 MambaVision 在 Block 中的默认配置一致），接受 MV-Mixer MLP 比 Attention MLP 小。
- **推荐**：MV-Mixer Block 使用 `mlp_ratio=4`，Attention Block 保持 `mlp_ratio=8`。这也符合 MambaVision 论文中的做法（Mamba stage 和 Attention stage 使用相同的 `mlp_ratio=4`）。

### 6.2 🟡 风险：`expand=1` 下 MV-Mixer 容量受限

**问题**：MambaVision 的 Block 使用 `expand=1`（`d_inner = dim`），SSM 分支宽度仅 `dim/2 = 256`。相比标准 Mamba 的 `expand=2`（SSM 宽度 = dim = 512），信息瓶颈更窄。

**影响**：在 512 维 token 上，SSM 每次仅处理 256 维信息。对于 3D 医学图像的复杂特征，这可能不够。

**缓解**：这正是 Hybrid 设计的初衷——Mamba 做粗略的长程混合，Attention 做精细的全局建模。如果 E2 (Hybrid-1) 效果不佳，E3 (Hybrid-2) 增加一个 MV-Mixer 层可以进一步测试是否是容量问题。

### 6.3 🟡 风险：Dropout 策略不一致

**问题**：
- IM-Fuse 原始 Transformer 在 attention 和 FFN 中使用 `dropout=0.1`
- MambaVision MV-Mixer 默认 `drop=0.`
- 计划文档未明确 MV-Mixer 的 dropout 设置

**建议**：MV-Mixer Block 的 MLP `drop=0.1` 以与整体 pipeline 一致。Mixer 本身（`MambaVisionMixer`）内部无 dropout，这是合理的（SSM 内部一般不加 dropout）。

### 6.4 🟡 风险：LR Scheduler 只更新第一个 param_group

**问题**：当前 `LR_Scheduler` 实现只更新 `optimizer.param_groups[0]`：
```python
def _adjust_learning_rate(self, optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
```

**影响**：分组学习率需要遍历所有 param_groups。

**修复**：必须修改 scheduler 为按比例更新所有 groups（见 4.5 节的 `LR_Scheduler_Grouped`）。

### 6.5 🟡 风险：`InitWeights_He` 可能覆盖 MV-Mixer 的特殊初始化

**问题**：`IMFuse.__init__` 末尾调用 `self.apply(InitWeights_He(1e-2))`，这会对所有模块应用 He 初始化。但 MV-Mixer 有自己的精心设计的初始化（`dt_proj.bias` 通过 `inv_dt` 初始化，`A_log` 通过 `log(arange)` 初始化，`D` 初始化为全 1）。

**影响**：`InitWeights_He` 可能覆盖这些特殊初始化值。

**修复**：
1. 在 `self.apply(InitWeights_He(1e-2))` **之后**创建 Hybrid Encoder（或重新初始化 MV-Mixer）
2. 或者修改 `InitWeights_He` 跳过 MV-Mixer 的特定参数（检查 `_no_reinit` 标志）
3. **推荐**：在模型构建后，由权重迁移脚本处理：先 load 预训练权重，MV-Mixer 部分自动保持随机初始化（因为 src 中无对应 key），然后手动重新初始化 MV-Mixer 的特殊参数。

### 6.6 🟢 确认：位置编码处理合理

**结论**：计划中未详细讨论位置编码，但实施方案（4.2 节）的 `HybridTokenEncoder.forward()` 将 pos 只传给 Attention Block（在 Attention 前 `x = x + pos`），MV-Mixer 不使用 pos。这是合理的：
- MV-Mixer 通过 conv1d 实现局部位置感知，不需要显式位置编码
- Attention Block 保留了原始 Transformer 的完整行为（`x = x + pos` 后做 Attention）

### 6.7 🟢 确认：Pipeline 初始行为接近原始 IM-Fuse

当 `layer_scale_init=1e-5` 时：
- MV-Mixer Block 输出 ≈ `x + 1e-5 * mixer(LN(x)) + 1e-5 * mlp(LN(x))` ≈ `x`（近似 identity）
- 随后 Attention Block 接收几乎不变的 `x`，执行完整的 `x + pos → Attn → FFN`
- **确认**：初始行为确实接近原始 IM-Fuse

### 6.8 🟢 确认：Optimizer 从 RAdam → AdamW

当前 `train_poly.py` 使用 `RAdam`，计划使用 `AdamW`。这是一个有意识的选择，AdamW 在迁移学习场景中更常用。两者在收敛后性能相近，但 AdamW 的 weight decay 机制与分组学习率配合更好。

### 6.9 🔴 风险：`SelfAttention` 的 `qkv` 没有 bias

**关键发现**：IM-Fuse 的 `SelfAttention` 定义为 `self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)`，而 `Transformer` 实例化时 **未传 qkv_bias 参数**，默认为 `False`。

这意味着如果 `HybridAttentionBlock` 复用 IM-Fuse 的 `SelfAttention`（如 4.2 节方案），则 qkv 无 bias，与预训练权重匹配。**但如果改用 MambaVision 的 `Attention` 类**，需要显式设 `qkv_bias=False`。

**推荐**：直接复用 IM-Fuse 的 `SelfAttention`、`FeedForward`、`PreNormDrop`、`PreNorm`、`Residual` 类，避免任何结构差异。实施方案（4.2节）已采用此策略。

---

## 7. 实施 Checklist

### Phase 0：Sanity Check

- [ ] 创建 `mv_mixer.py`，从 MambaVision 提取 `MambaVisionMixer`
- [ ] 验证 `selective_scan_fn` 可正常导入
- [ ] 创建 `hybrid_encoder.py`，实现 `HybridTokenEncoder`
- [ ] 修改 `IMFuse.py`，添加 `hybrid=True` 选项（保留原始 Transformer 作为默认）
- [ ] 验证 Hybrid 模型可正常前向传播（随机权重）
- [ ] 验证 Hybrid 模型可正常反向传播
- [ ] 验证 `InitWeights_He` 不会破坏 MV-Mixer 特殊初始化
- [ ] 单卡 overfit 20 case

### Phase 1：权重迁移

- [ ] 实现 `scripts/transfer_weights.py`
- [ ] 打印 source / target state_dict 的完整 key 列表，人工确认映射正确
- [ ] 加载迁移权重后，与原始 IM-Fuse 做前向对比（相同输入，Hybrid 模型输出应接近原始）
- [ ] 验证迁移后的模型在 val set 上的 Dice（应接近原始基线，因为 MV-Mixer γ≈0）

### Phase 2：分组学习率训练

- [ ] 创建 `train_hybrid.py`
- [ ] 实现 `build_param_groups()` 并打印每组参数数量
- [ ] 实现 `LR_Scheduler_Grouped` 支持 warmup + poly + 多 param_group
- [ ] 前 30 epoch 监控：val loss、all-modal Dice、T1c-missing ET Dice
- [ ] 全量训练 300-500 epochs

### Phase 3：消融实验

- [ ] E1 (Mamba-only): `num_mixer_blocks=2`, 去掉 Attention Block
- [ ] E3 (Hybrid-2): `num_mixer_blocks=2` + Attention × 1

---

## 8. 文件结构规划

```
MV-IM-Fuse/
├── mv_mixer.py              # MambaVisionMixer（从 MambaVision 提取）
├── hybrid_encoder.py         # HybridTokenEncoder
├── IMFuse.py                 # 修改：添加 hybrid 模式
├── train_hybrid.py           # 新训练脚本（分组 LR + warmup + 权重迁移）
├── scripts/
│   └── transfer_weights.py   # 权重迁移工具
└── docs/
    ├── 3D MambaVision-lite × IM-Fuse 实验计划.md
    └── implementation_plan.md  # 本文档
```

---

## 9. 关键数值汇总

| 项目 | 值 |
|---|---|
| Token 维度 | 512 |
| Token 数量 | 512 (= 8³) |
| MV-Mixer: expand | 1 |
| MV-Mixer: d_state | 8 |
| MV-Mixer: d_conv | 3 |
| MV-Mixer: d_inner | 512 |
| MV-Mixer: SSM 宽度 | 256 |
| MV-Mixer: mlp_ratio | 4（推荐）或 8 |
| Attention: heads | 8 |
| Attention: mlp_ratio | 8（必须，与预训练匹配） |
| Attention: qkv_bias | False |
| Attention: dropout | 0.1 |
| Layer Scale (MV-Mixer) | 1e-5 |
| Layer Scale (Attention) | 无（γ=1.0） |
| Drop Path | 0.1 |
| 新增参数/模态 (mlp_ratio=4) | ~3.4M |
| 新增参数/模态 (mlp_ratio=8) | ~5.5M |
| 总新增参数 (4 模态, mlp_ratio=4) | ~13.6M |
| Base LR | 2e-4 |
| Warmup | 10 epochs |
| 总 Epochs | 300-500 |
| Optimizer | AdamW (weight_decay=3e-5) |
