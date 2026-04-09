---
paths:
  - "torchtitan_npu/patches/**/*.py"
---

# Patch 机制规则

适用范围：`torchtitan_npu/patches/`

## 核心原则

### 不修改上游源码

所有对上游 torchtitan、PyTorch、torch_npu 的行为修改，必须通过 monkey-patch 方式实现。
绝不直接修改上游仓库中的文件。

### 统一注册入口

所有 patch 必须在 `torchtitan_npu/__init__.py` 的 `_apply_patches()` 函数中注册（通过 import 触发）。
遗漏注册的 patch 不会生效，且难以排查。

### 目录组织

按 patch 目标分子目录：

- `patches/torchtitan/` — patch 上游 torchtitan 核心模块
- `patches/torch/` — patch PyTorch 原生模块
- `patches/torch_npu/` — patch torch_npu 模块
- `patches/distributed/` — patch 分布式相关工具
- `patches/optimizer/` — patch 优化器
- `patches/quantization/` — patch 量化模块
- `patches/tools/` — patch 训练辅助工具

新增 patch 时，放入对应的子目录。如果目标不属于上述任何类别，评估是否需要新建子目录。

## 编写规范

### 最小化 patch 范围

- 每个 patch 文件只 patch 一个明确的功能点。
- 避免在同一个 patch 文件中修改多个不相关的模块。
- Patch 函数应尽可能少地复制上游代码，优先使用 `functools.wraps` 或装饰器模式。

### 文档化 patch 目标

每个 patch 文件头部必须注释说明：

- Patch 的目标模块和函数/类。
- Patch 的原因（为什么需要这个 patch）。
- 对应的上游版本或 commit（便于上游同步时检查）。

### 上游同步兼容性

- 上游重构后，patch 的目标函数/类可能已不存在或签名已变。
- 每次上游同步必须逐一检查所有 patch 是否仍能正确应用。
- 如果上游已提供等价功能，应删除对应的 patch 而非继续维护。

### 测试

- 添加或修改 patch 后，确保在有 patch 和无 patch 两种条件下的行为均已验证。
- 涉及数值计算的 patch 必须完成 loss 对比验证。
