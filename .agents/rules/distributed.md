---
paths:
  - "torchtitan_npu/distributed/**/*.py"
  - "torchtitan_npu/patches/distributed/**/*.py"
---

# 分布式代码规则

适用范围：`torchtitan_npu/distributed/`、`torchtitan_npu/patches/distributed/`

## 继承上游规则

### 显式断言 Mesh 和 Placement

- 永不假设 1D mesh。使用前显式断言 mesh 维度。
- 显式验证 tensor placement（Replicate、Shard、Partial）。
- 对普通 tensor 输入强制非 None 字段时，附带清晰的错误信息。

### 标注并行语义

使用 `DTensor.to_local` 时显式指定 `grad_placements`，尤其当原始 DTensor placement 包含 `Replicate` 或 `Partial` 时。包括通过 `local_map`（`in_grad_placements`）和 `full_tensor`（`grad_placements`）的间接调用。

### 考虑所有并行组合

新增或修改分布式代码时，思考与每个并行维度（TP/DP/PP/CP/EP）的交互。一种配置的修复可能破坏另一种。在 bug 报告和测试描述中包含并行配置信息。

### 模型无关代码放通用目录

适用于任何模型的 helper 函数（如 `maybe_enable_async_tp`、`NoParallel`、tensor parallel 工具）放在 `torchtitan_npu/distributed/`，不放在模型特定文件中。

### 保守修改

分布式训练代码难以全面测试。修改现有行为时：

- 在多种并行配置下验证数值一致性。
- 警惕静默正确性问题（错误的 gradient placement、破坏 DCP 的 identity 操作）。
- 修改已收敛并验证的代码需提供充分理由和全面测试。

## 插件仓扩展规则

### NPU 通信特殊处理

- NPU 设备上的集合通信可能有不同于 CUDA 的行为差异，新增分布式 patch 时需在 NPU 上实测验证。
- 涉及 HCCL（NPU 通信库）的改动需注意与 NCCL 的语义差异。

### Patch 分布式模块的注意事项

- 在 `patches/distributed/` 中 patch 上游分布式工具时，确保不破坏上游的 mesh 初始化和进程组管理。
- Patch 前后的行为差异必须在注释中说明。
