---
paths:
  - "torchtitan_npu/models/**/*.py"
  - "torchtitan_npu/models/**/train_configs/*.toml"
---

# 模型代码规则

适用范围：`torchtitan_npu/models/`

## 继承上游规则

### 保持模型精简可读

- 模型文件只包含模型架构，不含训练基础设施。
- 权重初始化放在配置或专用 init 函数中，不散布在 `Module.__init__` 中。
- 模型变更后确保原始 checkpoint 仍能正确加载。

### 审计所有变体

修改共享组件（attention、normalization、MoE routing）时，检查并更新**所有**模型变体：llama3、llama4、qwen3、deepseek_v3、deepseek_v32 以及后续新增的模型。不要在相邻模型中留下过时模式。

### 跨模型统一

- 不要给每个模型创建功能相同的独立 wrapper。尽量只有一个通用 wrapper 供所有模型共享。
- 多个模型有近乎相同的代码（如 `apply_fsdp`、`apply_ac`、`apply_compile`）时，合并到 `torchtitan_npu/models/common/` 或上游 common 目录。
- 新增 rotary embedding、MoE router 等组件前，检查已有实现是否已支持该用例。

### 标准模型目录结构

每个模型目录遵循一致的模式：

- `config_registry.py` — 注册模型配置（size、超参）
- `infra/parallelize.py` — 定义模型的并行化策略
- `train_configs/*.toml` — 训练配置文件
- 模型定义文件（架构、层）

### 不要过度特化

仅一个模型需要的功能在该模型目录实现。不要为单个模型的需求修改共享基础设施或基类。

### Forward 中的控制流

保持控制流（路由决策、条件逻辑）在 `forward` 方法中。不要把重要分支逻辑埋在 helper 方法中。

## 插件仓扩展规则

### 模型注入机制

- 插件仓通过 `_inject_module()` 将新模型注入 `sys.modules`，使上游代码能透明发现。
- 新增模型时需在 `__init__.py` 的 `_apply_patches()` 中同时注册到 `_supported_models` 并调用 `_inject_module()`。

### NPU 模型适配

- NPU 特有的模型修改（如算子替换、内存优化）应通过 converter 或 patch 实现，不要直接 fork 上游模型代码。
- 如必须 fork 模型代码，在文件头注释中说明 fork 原因和对应的上游文件位置。
