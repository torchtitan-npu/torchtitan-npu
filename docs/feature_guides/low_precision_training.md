# 低精度训练特性（MXFP8 / HiFloat8）

在大规模语言模型的分布式训练中，矩阵乘法运算（GEMM）占据了绝大部分计算开销。传统的 BF16/FP16 混合精度训练虽然已大幅降低了显存占用，但在超大规模模型（如 DeepSeek-V3 671B）上仍面临计算效率瓶颈。低精度训练通过将线性层和 MoE 专家层的矩阵乘法降至 8-bit 浮点精度执行，在保持训练收敛性的前提下，显著提升计算吞吐并降低显存消耗。

本特性在 `torchtitan` 框架中引入了对 **MXFP8** 和 **HiFloat8** 两种 8-bit 浮点格式的支持，覆盖普通线性层（nn.Linear）和 MoE 专家层（Grouped MM）两大场景。

## 硬件要求

低精度训练特性仅支持 **Ascend 950 及更高架构**的 NPU 设备。系统在初始化时会通过 `torch_npu.npu.get_device_name()` 进行硬件检测，不满足要求时将抛出异常。

## 实现原理

### 整体架构

本特性通过 `torchtitan` 的 converter 机制实现，在模型构建完成后对目标模块进行无缝替换。相关代码主要分布在以下文件中：

| 文件路径 | 修改作用 |
| --- | --- |
| `torchtitan_npu/converters/quant_converter.py` | 对上游 `MXLinearConverter` 和 `MXGroupedMMConverter` 的初始化与转换逻辑进行 NPU 适配替换 |
| `torchtitan_npu/converters/kernels/quant_linear.py` | 实现低精度线性层 `MXLinear`，包含 MXFP8 和 HiFloat8 的前向/反向自定义算子 |
| `torchtitan_npu/converters/kernels/quant_gmm.py` | 实现低精度分组矩阵乘法，包含 MXFP8 和 HiFloat8 的前向/反向自定义算子 |
| `torchtitan_npu/patches/quantization/quant_config.py` | 定义量化配置数据类 `MXLinearConfig`、`MoETrainingConfig` 及对应的 recipe 枚举 |
| `torchtitan_npu/patches/quantization/quantize.py` | 提供 `linear_quantize_` 和 `grouped_quantize_` 函数，实现模型模块的递归遍历与替换 |

### 线性层低精度（`quantize.linear.mx`）

通过 converter 机制，系统将模型中符合条件的 `nn.Linear` 模块替换为自定义的 `MXLinear` 模块。`MXLinear` 继承自 `nn.Linear`，在 `forward` 方法中根据配置的 `recipe_name` 调用对应的低精度自定义算子：

- **MXFP8 模式**：使用 `torch_npu.npu_dynamic_mx_quant` 对激活和权重进行 per-block 量化（block size=32），每 32 个元素共享一个scale，再通过 `torch_npu.npu_quant_matmul` 执行低精度矩阵乘法。前向传播的线性变换及反向传播的输入梯度、权重梯度计算以 FP8 精度执行，最终输出恢复为原始精度。
- **HiFloat8 模式**：使用 `torch_npu.npu_dynamic_quant` 对激活和权重进行 per-tensor 动态量化，整个张量共享一个 scale，再通过 `torch_npu.npu_quant_matmul` 执行低精度矩阵乘法。

用户可通过 `filter_fqns` 配置项，指定不进行低精度替换的线性层（如 `output` 层和 `router.gate` 层），以避免对精度敏感的模块产生影响。

### MoE 专家层低精度（`quantize.grouped_mm.mx`）

对于 MoE（Mixture of Experts）架构中的专家层，系统通过替换 `npu_grouped_mm` 函数为低精度版本来实现量化加速：

- **MXFP8 模式**：在前向传播中，使用 `torch_npu.npu_dynamic_mx_quant` 对输入和权重分别进行 per-block 量化（block size=32），再调用 `torch_npu.npu_grouped_matmul` 执行低精度分组矩阵乘法。反向传播中，梯度计算同样在 FP8 精度下完成，其中权重梯度的计算使用 `torch_npu.npu_grouped_dynamic_mx_quant` 进行 per-block 量化。
- **HiFloat8 模式**：先根据专家数和 EP 并行度计算分组大小 `g_size`，再通过 `reshape(g_size, -1)` 将张量重塑为 `g_size` 行，利用 `torch_npu.npu_dynamic_quant` 的默认 per-token 量化模式对每一行（实际对应每个专家分组）独立量化，每个专家分组共享一个scale。量化后调用 `torch_npu.npu_grouped_matmul` 执行低精度分组矩阵乘法。

> **注意**：MoE 低精度功能依赖 `npu_gmm` converter 提供的分组矩阵乘法基础实现，因此在 `converters` 配置中 `"npu_gmm"` 必须位于 `"quantize.grouped_mm.mx"` 之前。


## 配置选项

在训练任务的 TOML 配置文件中，通过 `[model]` 节的 `converters` 字段启用低精度 converter，并在对应的 `[quantize.linear.mx]` 和 `[quantize.grouped_mm.mx]` 节中设置详细参数。

### 线性层低精度配置（`quantize.linear.mx`）

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `recipe_name` | str | "mxfp8" | 量化方式。可选值：`"mxfp8"`、`"hif8"`。 |
| `filter_fqns` | list[str] | [] | 不进行低精度替换的模块名称列表。匹配规则为子字符串包含，如 `"output"` 将过滤所有全限定名（Fully Qualified Name, FQN）中包含 "output" 的线性层。 |

### MoE 专家层低精度配置（`quantize.grouped_mm.mx`）

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `recipe_name` | str | "mxfp8" | 量化方式。可选值：`"mxfp8"`、`"hif8"`。 |
| `fqns` | list[str] | [] | 需要进行低精度替换的 MoE 模块名称列表。匹配规则为子字符串包含，如 `"experts"` 将匹配所有全限定名（Fully Qualified Name, FQN）中包含 "experts" 的模块。 |

### 配置示例

首先在配置文件中使能本代码仓的自定义配置，随后在 `[model]` 节中配置 `converters` 并添加对应的量化参数节：

**示例一：仅对线性层启用低精度训练**

```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"    # 使能本代码仓的自定义配置

[model]
converters = ["quantize.linear.mx"]

[quantize.linear.mx]
recipe_name = "mxfp8"                     # 可选 "mxfp8" 或 "hif8"
filter_fqns = ["output", "router.gate"]   # output 和 router.gate 层不做低精度替换
```

**示例二：同时对线性层和 MoE 专家层启用低精度训练**

```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"

[model]
# npu_gmm 必须在 quantize.grouped_mm.mx 之前
converters = ["npu_gmm", "quantize.linear.mx", "quantize.grouped_mm.mx"]

[quantize.linear.mx]
recipe_name = "mxfp8"                     # 可选 "mxfp8" 或 "hif8"
filter_fqns = ["output", "router.gate"]

[quantize.grouped_mm.mx]
recipe_name = "mxfp8"                     # 可选 "mxfp8" 或 "hif8"
fqns = ["experts"]
```

## 验证清单

1. **确认 converter 生效**：启动日志中应出现以下关键字（MXFP8 和 HiFloat8 均适用）：
   - 线性层：`MX training active with recipe <recipe_name>`（其中 `<recipe_name>` 为 `mxfp8` 或 `hif8`）和 `Swapped to MXLinear_NPU layers`
   - MoE 专家层：`<recipe_name> MoE training enabled` 和 `[MXFP8/Hif8 GMM] Replaced <N> NPU GMM methods/functions`
2. **确认模块替换数量**：日志中 `Replaced <N> NPU GMM methods/functions` 的数量应与预期的 MoE 专家模块数一致；线性层可通过 `model.named_modules()` 检查 `MXLinear` 类型的模块数量。
3. **常见未生效场景排查**：
   - `converters` 顺序错误：`"npu_gmm"` 未放在 `"quantize.grouped_mm.mx"` 之前，导致 MoE 专家层替换失败
   - `filter_fqns` / `fqns` 匹配不到目标模块：检查模块的 FQN 是否包含配置的子字符串（注意大小写敏感）
   - 硬件不满足要求：日志报错 `[MXFP8/Hif8] is only supported on Ascend950 or higher architecture`
