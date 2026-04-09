---
paths:
  - "torchtitan_npu/converters/**/*.py"
---

# Converter / Kernel 机制规则

适用范围：`torchtitan_npu/converters/`

## 核心原则

### 通过注册表管理

所有算子转换必须通过 `torchtitan_npu/converters/registry.py` 注册。不要在模型文件中硬编码算子替换逻辑。

### 不修改上游 ModelConverter 接口

Converter 实现必须遵循上游 `ModelConverter` 接口约定。如果上游接口变更，适配工作在 converter 层完成，不要修改上游代码。

## 目录结构

- `registry.py` — 转换器注册表
- `base_converter.py` — 基础转换器抽象
- `npu_converter.py` — NPU 设备转换器
- `quant_converter.py` — 量化转换器
- `convert_utils.py` — 转换工具函数
- `kernels/` — NPU 自定义 kernel 实现
- `features/` — 特性适配（Feature-based 注入）

## 编写规范

### Kernel 实现

- 自定义 kernel 放在 `converters/kernels/` 目录。
- 每个 kernel 文件头部注释说明：替换的原始算子、替换原因（性能/精度/兼容性）、NPU 硬件要求。
- Kernel 必须附带精度对比数据或引用已有的精度验证结果。

### Converter 注册

- 注册时确保不与已有 converter 冲突。
- 注册后验证 `_apply_patches()` 中 converter 的 import 顺序不影响功能。

### 上游同步注意事项

- 上游新增或修改算子时，检查对应 converter 是否需要更新。
- 上游模型代码使用新的算子调用方式时，确保 converter 仍能正确拦截。
