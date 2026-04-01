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

### 集成测试 (Integration Test)

`tests/smoke_tests/integration_test.py` 是端到端集成测试入口，用于验证：
- 新增模型功能支持情况
- 特性兼容性
- 并行策略兼容性

#### 运行方式

```bash
# 通过 build.sh 运行（默认运行 core + extended smoke）
ONLY_CORE_SMOKE=true sh build.sh -s --generate-report

# 独立运行 integration_test.py
python tests/smoke_tests/integration_test.py output_dir \
    --test_name all \
    --ngpu 2
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | 无（必填） | 测试输出目录 |
| `--config_path` | `./tests/smoke_tests/base_test.toml` | 基础配置文件路径 |
| `--test_name` | `all` | 指定测试用例名称 |
| `--ngpu` | `2` | 最大 GPU 数 |

#### OverrideDefinitions 使用说明

`OverrideDefinitions` 是定义集成测试用例的配置类：

```python
OverrideDefinitions(
    override_args=[[...]],  # 必填：命令行参数列表
    test_descr="...",        # 必填：测试描述
    test_name="...",         # 必填：测试名称
    ngpu=2,                  # 可选：所需 GPU 数
    disabled=False,          # 可选：是否禁用
)
```

#### 新增测试用例步骤

1. 打开 `tests/smoke_tests/integration_test.py`
2. 在 `generate_smoke_tests()` 函数的 `smoke_cases` 列表中添加新配置：
```python
OverrideDefinitions(
    [
        [
            "--model.name your_model",
            "--model.flavor your_flavor",
            "--parallelism.tensor_parallel_degree 2",
        ],
    ],
    "Your Model TP Test",
    "your_model_tp",
    ngpu=2,
)
```
3. 运行测试验证：
```bash
python tests/smoke_tests/integration_test.py ./outputs --test_name your_model_tp
```

#### base_test.toml 配置文件

`tests/smoke_tests/base_test.toml` 是集成测试的基础配置，所有测试都会基于这个配置文件运行，`override_args` 中的参数会覆盖基础配置中的同名参数。

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
| `ONLY_CORE_SMOKE=true` | 修改了最小训练主链路（即 integration_test 中定义的端到端集成测试） |
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
