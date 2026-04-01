# Muon 优化器特性

在大规模语言模型的训练中，优化器的选择对收敛速度和最终性能有着重要影响。传统的 Adam/AdamW 优化器虽然通用性强，但对于大规模矩阵参数（Linear 层的权重）的更新策略并非最优。Muon 优化器通过引入动量正交化（Momentum Orthogonalization）技术，针对 2D 矩阵参数实现了更高效的梯度下降策略。

## 实现原理

Muon 优化器的核心创新在于对 2D 参数（矩阵）采用不同于传统优化器的更新机制。其理论基础源于以下观察：

1. **传统优化器的局限**：Adam/AdamW 等优化器将参数视为独立的一维向量进行更新，忽略了矩阵参数的结构信息
2. **动量正交化思想**：在每一步更新时，将动量分解为与当前梯度方向平行和垂直的两个分量，仅保留正交分量以实现更稳定的收敛

### 参数分配策略

torchtitan-npu 目前采用了 **Muon + AdamW 混合优化器** 策略：

- **2D 参数**（如 Linear 层的权重矩阵）→ 使用 Muon 优化器
- **非 2D 参数**（如偏置、LayerNorm、Embedding）→ 使用 AdamW 优化器


这种混合策略既能发挥 Muon 在矩阵参数上的收敛优势，又能保证非矩阵参数的稳定训练。

### 学习率调整模式

Muon 优化器支持两种学习率调整模式（通过 `muon_adjust_lr_fn` 配置），其核心区别在于如何根据矩阵形状调整学习率，以及是否需要独立的超参数调优。

| 模式 | 调整公式 | 说明 |
|------|----------|------|
| `original` | $\gamma \leftarrow \gamma \cdot \sqrt{\max(1, A/B)}$ | Keller Jordan 原始实现，根据矩阵宽高比调整 |
| `match_rms_adamw` | $\gamma \leftarrow 0.2 \cdot \gamma \cdot \sqrt{\max(A, B)}$ | Moonshot 实现，直接复用 AdamW 的 lr 和 weight_decay |

#### original 模式详解

该模式源自 Muon 创始人 Keller Jordan 的原始实现。调整公式为：

$$\gamma_{\text{adjusted}} = \gamma \times \sqrt{\max\left(1, \frac{A}{B}\right)}$$

其中 $A$ 和 $B$ 是矩阵的两个维度。这个调整的目的是：**让正交化后的梯度更新在不同形状的矩形矩阵上具有一致的 RMS（Root Mean Square）**。

- 当 $A \le B$（宽矩阵，如 FFN 中的中间层）时，系数为 1，不做额外调整
- 当 $A > B$（高矩阵，如输出层）时，按 $\sqrt{A/B}$ 缩放

这种设计使得 Muon 可以在不同形状的矩阵上保持相似的收敛行为。由于调整幅度较大，通常需要单独为 Muon 调优学习率（即配置 `muon_lr`），一般来说可以将Adamw的学习率放大10倍来作为Muon的学习率。这就是为什么该模式需要配合 `MuonLRSchedulersContainer` 使用。

#### match_rms_adamw 模式详解

该模式来自 Moonshot 团队的论文 [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)。调整公式为：

$$\gamma_{\text{adjusted}} = 0.2 \times \gamma \times \sqrt{\max(A, B)}$$

这个模式的设计目标是：**让 Muon 可以直接复用已经为 AdamW 调优好的学习率和权重衰减超参数**，无需额外的超参数搜索。

论文实验表明，使用此调整后，Muon 在大模型训练任务上可以达到与 AdamW 相近的收敛效果，同时利用动量正交化获得更快的收敛速度。由于调整后的学习率与 AdamW 处于同一量级，Muon 和 AdamW 可以共享相同的基础学习率，使用标准的 `LRSchedulersContainer` 即可。

#### 模式选择建议

- **使用 `match_rms_adamw`**：如果你已经为 AdamW 调优好了超参数，希望直接尝试 Muon 而不想重新调参
- **使用 `original`**：如果你愿意投入时间单独调优 Muon 的学习率，追求可能的更好收敛效果

## 配置选项

在训练任务的 TOML 配置文件中，找到对应的 `[optimizer]` 节，并添加以下配置以启用 Muon 优化器：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `name` | str | "AdamW" | 优化器类型，设置为 "Muon" 启用本特性 |
| `muon_lr` | float | None | Muon 专用学习率，若不设置则使用 base lr |
| `muon_momentum` | float | 0.9 | Muon 的动量因子 |
| `muon_enable_nesterov` | bool | False | 是否启用 Nesterov 动量 |
| `muon_ns_steps` | int | 5 | 步数参数，影响正交化计算的迭代次数 |
| `muon_adjust_lr_fn` | str | "match_rms_adamw" | 学习率调整模式：`"original"` 或 `"match_rms_adamw"` |

### 配置示例

```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"    # 使能本代码仓的自定义配置

[optimizer]
name = "Muon"                        # 使用 Muon 混合优化器
lr = 3e-4                            # 基础学习率（AdamW 部分使用）
muon_lr = 1e-3                       # Muon 专用学习率（可选）
weight_decay = 0.01
muon_momentum = 0.9                  # Muon 动量因子
muon_enable_nesterov = false         # 是否启用 Nesterov 动量
muon_ns_steps = 5                    # 正交化步数
muon_adjust_lr_fn = "original"       # 使用独立的 lr 调度器
```

### 注意事项

1. **单设备限制**：Muon 优化器目前仅支持单设备训练，不支持多设备分布式场景。若在多设备环境下启用，将抛出 `NotImplementedError`：

   ```python
   is_distributed = torch.distributed.is_initialized()
   world_size = torch.distributed.get_world_size() if is_distributed else 1
   if world_size > 1:
       raise NotImplementedError(
           "Muon optimizer currently only support single device"
       )
   ```

2. **与 Swap Optimizer 互斥**：Muon 优化器不支持与 Swap Optimizer 特性同时启用：

   ```python
   if getattr(optimizer_config, "swap_optimizer", False):
       raise ValueError(
           "Muon optimizer does not support swap_optimizer. "
           "Please set swap_optimizer=false in your config."
       )
   ```


## 参考文献

- [PyTorch Muon 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
- [Muon 优化器指南：快速上手与关键细节](https://kexue.fm/archives/11416)
- [Muon 优化器赏析：从向量到矩阵的本质跨越](https://www.spaces.ac.cn/archives/10592)
