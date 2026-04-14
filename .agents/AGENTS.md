# torchtitan-npu 开发指南

`torchtitan-npu` 是 [torchtitan](https://github.com/pytorch/torchtitan) 的 **Ascend NPU 插件仓**。
本仓不直接修改上游 torchtitan 代码，而是通过 monkey-patch、ModelConverter、模型注入等机制将 NPU 适配能力叠加到上游之上。

## 构建与测试

```bash
# 安装依赖
pip install -r requirements.txt -r requirements_dev.txt

# Lint & 格式化（PR 前必须通过）
pre-commit run --all-files

# 运行单元测试
pytest tests/ -x
```

### 数值验证

非计算性改动（如重构、activation checkpointing 调整）必须保证修改前后 **loss 完全一致**（使用 `--debug.seed=42 --debug.deterministic`）。
计算性改动需在代表性数据集（如 C4）上展示 loss 收敛。

相同并行策略和 debug 选项下，两次运行应产生 bit-wise 一致的 loss 和 grad_norm。
stdout 仅打印 5 位有效数字，可能不够精确，请使用 `scripts/loss_compare.py` 开启 profiling 并从 TensorBoard 结果中检查。

**禁止** 使用 `--debug.deterministic_warn_only`。

## 核心原则

1. **PyTorch 原生训练技术。** torchtitan 核心的训练基础设施和并行代码不依赖非 PyTorch 库。作为插件仓，torchtitan-npu 可使用 torch_npu 等外部库，但应尽可能复用 PyTorch 原生接口。

2. **查明根因再修复。** 不做绷带式修补。在提出方案前理解 *为什么* 出错。如果一个改动看似有效但无法解释原因，需要更深入排查。

3. **复用优于重复。** 新写代码前，检查已有实现是否已覆盖需求。尽量统一跨模型的相似代码路径，不要给每个模型创建独立 wrapper。若上游（torchao、PyTorch）已提供功能，优先使用。

4. **不要将实验泄漏到核心。** 插件仓中如需实验性代码，务必与核心适配逻辑隔离，不要在核心 patch/converter 文件中添加 `if experiment_x:` 分支。

5. **保护已验证的代码路径。** 修改已收敛的代码时务必谨慎。标记可能导致现有用户代码或 checkpoint 静默失效的改动。存疑时主动询问。

6. **审计所有调用点。** 修改共享代码（公共模型组件、配置字段、分布式工具）时，检查并更新所有调用点。这包括所有模型变体如 llama3、llama4、qwen3、deepseek_v3、deepseek_v32 等。

## 插件仓专属原则

1. **绝不修改上游代码。** torchtitan-npu 不直接修改 torchtitan 源码。所有适配通过以下机制实现：
   - **Patches**（`torchtitan_npu/patches/`）：monkey-patch 上游模块的函数或类
   - **Converters**（`torchtitan_npu/converters/`）：通过 ModelConverter 注册表替换算子或注入自定义 kernel
   - **模型注入**（`torchtitan_npu/models/`）：补充或覆盖上游模型实现

2. **理解 patch 生效机制。** 所有 patch 在 `torchtitan_npu/__init__.py` 的 `_apply_patches()` 中注册，包导入即生效。新增 patch 必须在此函数中添加对应 import。

3. **Converter 遵循注册表模式。** 自定义算子转换必须通过 `torchtitan_npu/converters/registry.py` 注册。不要在模型文件中硬编码算子替换逻辑。

4. **上游同步是常态。** 本仓需定期跟踪上游 torchtitan 变更。同步基线信息维护在 `docs/source/community/versioning_policy.md` 的分支同步表。每次同步后必须更新此表。

5. **Patch 目标随上游变化。** 上游重构后，patch 的目标函数/类可能已不存在或签名已变。每次上游同步必须检查所有 patch 是否仍有效。

## 代码风格（继承上游 torchtitan）

### 命名

- 名称必须 **准确、描述性、反映实际作用域**。不要在生产代码中使用 "toy/test/temp" — 这类上下文放在 docstring 中。
- 遵循上游约定：匹配 torchao 和 PyTorch 的命名。
- 计数使用 `num_` 前缀（如 `num_expert_groups` 而非 `n_expert_groups`）。

### 代码放置

代码放到 **最通用的适用位置**：

| 目录 | 职责 |
| --- | --- |
| `torchtitan_npu/patches/` | 对上游 torchtitan、PyTorch、torch_npu 等模块的 monkey-patch，按 patch 目标分子目录 |
| `torchtitan_npu/converters/` | 算子转换器注册表与自定义 kernel，通过 registry 机制注入 |
| `torchtitan_npu/models/` | 模型实现（覆盖或扩展torchtitan），含并行化策略与训练配置 |
| `torchtitan_npu/distributed/` | NPU 专属分布式工具 |
| `torchtitan_npu/config/` | NPU 自定义配置扩展 |
| `torchtitan_npu/tools/` | 训练辅助工具（flight_recorder、profiling 等） |
| `torchtitan_npu/train.py` | 训练流程 patch（如模型专属训练逻辑） |
| `torchtitan_npu/entry.py` | 训练入口点 |

不要把模型无关的功能放在模型特定文件中。

### 断言与错误处理

- **`ValueError`** 用于用户可见的错误（配置错误、无效输入）。
- **`assert`** 仅用于表示程序员错误的内部不变量。
- 分布式代码中显式验证 mesh 维度、tensor placement 和配置值 — 不要假设 1D mesh 或特定 placement。
- 代码路径静默跳过用户配置时，**发出 warning**。

### 参数与配置

- 重要参数放前面，次要参数放后面。
- 首个位置参数之后优先使用 keyword-only 参数。
- 必需配置字段不要用 `None` 默认值。
- `dataclasses.replace()` 是浅拷贝：嵌套 dataclass 和 list/dict 字段共享引用。需要深拷贝时显式处理。

### 注释与文档

- 仅为真正不明显的内容添加注释：维度语义、并行梯度 placement、workaround 存在的原因。
- 使用 TODO 注释标记已知限制并附简要说明。
- 描述放在 docstring 中，不要放在名称里。

## 领域专项规则

针对不同代码领域的详细规则，请参阅 `.agents/rules/` 下的专项文件：

| 规则文件 | 适用范围 |
| --- | --- |
| `rules/config.md` | 配置系统（`torchtitan_npu/config/`） |
| `rules/distributed.md` | 分布式训练代码（`torchtitan_npu/distributed/`、`torchtitan_npu/patches/distributed/`） |
| `rules/models.md` | 模型实现（`torchtitan_npu/models/`） |
| `rules/patches.md` | Patch 机制（`torchtitan_npu/patches/`） |
| `rules/converters.md` | Converter / Kernel 机制（`torchtitan_npu/converters/`） |

## PR 要求

1. **先 lint:** 提交前运行 `pre-commit run --all-files` 并修复所有问题。
2. **展示loss曲线:** 任何非 trivial 改动需包含 loss 对比曲线。
3. **解释"为什么"而不只是"做了什么"。**
4. **添加测试:** 新功能至少需要 CPU 单元测试；涉及并行的需要 NPU 集成测试。
5. **保持模型代码精简:** 模型变更后确保原始 checkpoint 仍能加载，并说明变更原因。
6. **验证 patch 兼容性:** 新增或修改 patch 后，确认 `_apply_patches()` 注册正确，不与现有 patch 冲突。
7. **更新 converter 注册:** 新增算子转换后，确认 registry 注册正确。

## 可用 Skills

| Skill | 说明 |
| --- | --- |
| `accuracy-debug` | 有基线对照的训练精度异常定位（loss 偏离、NaN/Inf），基于代码审查 + detect_anomaly + msprobe dump/compare 流程 |
| `oom-analysis` | NPU 训练 OOM 问题诊断，按日志分类 → 静态内存估算 → Memory Snapshot 深度分析 → 优化建议的流程定位和解决问题 |
| `torchtitan-sync` | 上游 torchtitan 分支同步与适配，读取 versioning_policy.md 分支同步表，生成变更分析并完成代码适配 |

## 关键路径速查

| 类别 | 路径 |
| --- | --- |
| 训练入口 | `torchtitan_npu/entry.py` |
| 训练 patch | `torchtitan_npu/train.py` |
| Patch 注册（包初始化） | `torchtitan_npu/__init__.py` |
| Patches 目录 | `torchtitan_npu/patches/` |
| Converters 目录 | `torchtitan_npu/converters/` |
| Converter 注册表 | `torchtitan_npu/converters/registry.py` |
| 自定义配置 | `torchtitan_npu/config/custom_config.py` |
| 模型实现 | `torchtitan_npu/models/` |
| 分布式工具 | `torchtitan_npu/distributed/` |
| 训练辅助工具 | `torchtitan_npu/tools/` |
| 训练配置 | `torchtitan_npu/models/*/train_configs/*.toml` |
| 版本策略 / 同步基线 | `docs/source/community/versioning_policy.md` |
