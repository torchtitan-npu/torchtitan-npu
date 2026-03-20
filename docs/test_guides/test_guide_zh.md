# 测试使用指南

## 常用命令
### 单元测试
```bash
# 运行全部单元测试，并生成报告
sh build.sh -u --generate-report

# 只运行本仓 `torchtitan-npu` 的单元测试
RUN_TORCHTITAN_UT=false sh build.sh -u --generate-report
```

### 冒烟测试
```bash
# 运行默认 smoke 套件（core + extended）
sh build.sh -s --generate-report

# 只运行 core smoke
ONLY_CORE_SMOKE=true sh build.sh -s --generate-report

# 只运行 extended smoke
ONLY_EXTENDED_SMOKE=true sh build.sh -s --generate-report

# 只运行 upstream smoke
ONLY_UPSTREAM_SMOKE=true sh build.sh -s --generate-report
```

### 模型并行专项命令
```bash
# 基础模型并行冒烟测试
python3 -m pytest -v tests/smoke_tests/model_parallel/

# 多进程模型并行冒烟测试
RUN_MODEL_PARALLEL_MULTI_RANK=true torchrun --nproc_per_node=4 -m pytest -v tests/smoke_tests/model_parallel/
```

## 什么时候用哪个命令
| 命令 | 适用场景 |
|---|---|
| `build.sh -u` | 修改的是硬件无关逻辑，比如 converter、config、helper、patch |
| `build.sh -s` | 修改的是真实 NPU 执行链路或 wrapper 行为，并希望跑默认的 core + extended smoke |
| `ONLY_CORE_SMOKE=true` | 修改了最小训练主链路 |
| `ONLY_EXTENDED_SMOKE=true` | 修改了本仓特性或模型并行行为 |
| `ONLY_UPSTREAM_SMOKE=true` | 修改依赖上游 torchtitan 集成链路的逻辑，或需要单独跑更重的 upstream smoke |

## 快速判断
- 只改了硬件无关逻辑：先跑 `build.sh -u`
- 改了 NPU 特性链路或 wrapper：跑 `build.sh -s`
- 改了训练主链路接线：至少跑 `ONLY_CORE_SMOKE=true build.sh -s`
- 改了模型并行行为：跑 `ONLY_EXTENDED_SMOKE=true build.sh -s`
- 需要检查上游集成兼容性：单独跑 `ONLY_UPSTREAM_SMOKE=true build.sh -s`

## 测试报告
- 输出目录：`test_reports/`
- 常见产物：
  - `*.xml`：JUnit 结果
  - `*.html`：开启 `--generate-report` 后生成的 HTML 报告
  - `coverage/`：单元测试覆盖率报告
  - `README.md`：自动生成的报告索引

## 使用建议
1. 先跑和改动最匹配的最小命令。
2. 不依赖 NPU 的改动，优先跑 `build.sh -u`。
3. 能定向跑 smoke 子集时，就不要默认全量跑。
4. 如果测试布局或执行方式变了，记得同步更新文档。
