# TorchTitan-NPU 集成测试指南

本目录维护 TorchTitan-NPU 的集成测试（Integration Tests）定义、执行入口与校验逻辑。

集成测试分为两种运行模式：
* **本地单机集成测试**：在本地开发或测试机上直接运行，使用本仓内置的集成测试 Runner；
* **云端 Nightly CI 集群测试**：通过 GitHub Actions 触发，由内网 Lite Actions 调度机在 A3 / A5 真实集群上排队执行。

---

## 1. 现有测试用例全景矩阵

### 1.1 本地单机集成测试 Suite

| Suite 名称 | 覆盖模型 / 场景 | 主要特性与说明 | 运行命令示例 |
| :--- | :--- | :--- | :--- |
| **`models`** | DeepSeek-V4、DeepSeek-V3.2 | 多卡基础训练冒烟与 Loss 对齐校验 | `--test_suite models` |
| **`deepseek_v4_checkpoint`** | DeepSeek-V4 分布式续训 | 校验 DCP 状态、两阶段训练与续训 Loss 精确比对 | `--test_suite deepseek_v4_checkpoint` |
| **`deepseek_v4_quantized`** | DeepSeek-V4 Block-FP8 | 量化冻结基座 + 浮点 Adapter 训练（需 Ascend 950） | `--test_suite deepseek_v4_quantized` |
| **`deepseek_v4_1_engram_hf`** | DeepSeek-V4.1 Engram | Engram MXFP8 专项校验与 DCP/HF 权重回环测试 | `--test_suite deepseek_v4_1_engram_hf` |

#### 本地核心用例矩阵清单

| Case 名称 | 对应模型 | 并行配置 | 卡数 (ngpu) | Check Loss | 说明 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `dsv4_golden_1rank` | DeepSeek-V4 | 1 Rank 参考配置 | 1 | 是 | 校验 Deterministic 数值 loss |
| `dsv4_golden_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | 是 | 校验分布式数值 loss |
| `dsv4_muon_swap_ep2_fsdp2` | DeepSeek-V4 | DistMuon + NovaSwap | 2 | 否 | NPU 融合算子与 Swap 冒烟 |
| `dsv4_lora_ep2_fsdp2` | DeepSeek-V4 LoRA | EP2 + FSDP2 | 2 | 否 | 验证基座冻结与 PEFT 导出 |
| `dsv4_checkpoint_resume_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | 动态比对 | Step 2 恢复到 Step 4 精确比对 |
| `dsv4_smla_ep2_fsdp2` | DeepSeek-V4 SMLA | EP2 + FSDP2 | 2 | 否 | SMLA 算子图构建与执行冒烟 |
| `dsv3_2_dsa_ep2_fsdp2` | DeepSeek-V3.2 | DSA + EP2/FSDP2 | 2 | 是 | DSA 算子 100-step loss 比较 |

---

### 1.2 Nightly CI 集群用例 Suite（真实 NPU 集群看护）

Nightly CI 用例专为真实物理集群设计，由内网调度系统拉取并执行：

| Suite 名称 | Case ID | 硬件规格 | 优化器与训练配置 | 验证指标 |
| :--- | :--- | :--- | :--- | :--- |
| **`a3_8p_tests`** | `dsv4_flash_a3_8p_example` | A3 单机 8P | Muon，Eager，5 steps | Exit Code 0、TensorBoard 步数 |
| **`a3_8p_tests`** | `dsv4_flash_a3_8p_adamw` | A3 单机 8P | AdamW + Virtual Optimizer，4K seq，5 steps | Exit Code 0、TensorBoard 步数 |
| **`a3_16p_tests`** | `dsv4_flash_a3_16p_example` | A3 双机 16P | AdamW，EP16，Eager，双机各 8P，5 steps | Exit Code 0、双机通信与步数 |
| **`a5_64p_tests`** | `dsv4_pro_a5_64p` | A5 八机 64P | DeepSeek-V4 Pro，八机各 8P | 待硬件环境验收后开放 |

---

## 2. 如何运行已有测试

### 2.1 本地直接运行单机用例

在具备 NPU 环境的开发机上执行：

```bash
# 运行指定 suite（如 models）
python -m tests.integration_tests.run_tests ./test_reports/integration \
  --test_suite models \
  --ngpu 4

# 仅运行某一个特定用例
python -m tests.integration_tests.run_tests ./test_reports/checkpoint \
  --test_suite deepseek_v4_checkpoint \
  --test_name dsv4_checkpoint_resume_ep2_fsdp2 \
  --ngpu 2
```

> **参数说明**：
> - `./test_reports/...` 为必填输出目录，运行前请确保为空；
> - `--ngpu` 指定使用的卡数；如需串行运行可追加 `--no-parallel`。

---

### 2.2 触发云端 Nightly CI 测试

可以通过两种方式触发在真实集群上的 CI 测试：

#### 方式一：GitHub Actions 网页触发
1. 打开仓库的 [Actions 页面](https://github.com/torchtitan-npu/torchtitan-npu/actions)；
2. 在左侧列表中选择目标 Workflow（如 `A3-8p-CI-Example` 或 `A3-16p-CI-Example`）；
3. 点击右侧 **Run workflow** 下拉框，选择分支并点击运行。

#### 方式二：命令行触发 (`gh CLI`)

```bash
# 执行 A3 8P Suite 下的全部用例（自动拆分为独立 Matrix Job）
gh workflow run a3-8p-lite-actions.yml -R torchtitan-npu/torchtitan-npu --ref ci/lite-actions-validation \
  -f 'test_cases=[{"suite":"a3_8p_tests"}]'

# 仅执行特定的 16P 用例
gh workflow run a3-16p-lite-actions.yml -R torchtitan-npu/torchtitan-npu --ref ci/lite-actions-validation \
  -f 'test_cases=[{"test_id":"dsv4_flash_a3_16p_example"}]'
```

---

## 3. 如何新增一个 Nightly 集成用例（开发者实操）

向 Nightly CI 系统添加新用例非常简单，**只需在模型仓修改 Python 代码，无需改动调度机配置**。

### 步骤一：找到用例定义文件

Nightly 用例统一存放于：
`tests/integration_tests/nightly_all_models_test/`

根据用例的卡数规格，选择对应文件编辑（或新建 `*_tests.py` 文件，由动态发现机制加载）：
- 单机 8P 用例：`a3_8p_tests.py`
- 双机 16P 用例：`a3_16p_tests.py`
- 八机 64P 用例：`a5_64p_tests.py`

---

### 步骤二：编写 `OverrideDefinitions`

每个用例都是一个 `OverrideDefinitions` 对象。核心字段说明如下：

| 字段名称 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`test_name`** | `str` | 用例唯一标识（不可重复，用于 GitHub Matrix 调度命名与结果报告） |
| **`nnodes`** | `int` | 所需节点数（单机写 `1`，双机写 `2`） |
| **`ngpu`** | `int` | 单节点卡数（通常为 `8`） |
| **`override_args`** | `dict` | 传给训练脚本的命令行参数（覆盖默认 toml 配置） |
| **`env_vars`** | `dict` | 运行时环境变量（CANN 脚本路径、模型权重路径、优化器覆盖等） |
| **`expected_steps`** | `int` | 预期训练步数（用于 TensorBoard 与日志断言） |

---

### 步骤三：代码编写模板

打开对应的 `tests/integration_tests/nightly_all_models_test/a3_8p_tests.py`，在 `build_test_list()` 中添加新的用例定义：

```python
from torchtitan_npu.testing.framework import OverrideDefinitions

def build_test_list():
    tests = [
        # 现有用例...

        # 新增用例示例：
        OverrideDefinitions(
            test_name="dsv4_flash_a3_8p_my_feature",
            nnodes=1,
            ngpu=8,
            override_args={
                # 训练配置文件与步数
                "--job.config_file": "torchtitan_npu/models/deepseek_v4/train_configs/dsv4_flash_8p.toml",
                "--training.steps": "5",
                # 如有需要可覆写模型或并行参数
                "--training.batch_size": "1",
            },
            env_vars={
                # 必填：执行机上的 CANN 环境配置脚本
                "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/ascend-toolkit/latest/set_env.sh",
                # 必填：Tokenizer 与模型资产路径
                "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
                # 必填：权重 Checkpoint 路径
                "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
                # 可选：如果使用 Virtual Optimizer 优化 AdamW 显存占用
                "OPTIMIZER_OVERRIDES": "torchtitan_npu.override.common.optimizer.virtual",
                # 可选：设置共享 Hugging Face 缓存，防止 Docker 满盘
                "HF_HOME": "/mnt/share/ci_tests/.lite-actions-hf-cache",
            },
            expected_steps=5,
        ),
    ]
    return tests
```

---

### 步骤四：本地静态语法与解析自测（无需 NPU）

在提交代码前，可以在普通的 CPU / 开发机环境下执行 `inspect` 子命令，验证用例语法与参数是否合法：

```bash
# 验证 8P suite 能够正确解析并包含你的新增用例
python3 -m tests.integration_tests.tools.lite_actions.entrypoint inspect --suite a3_8p_tests
```

输出将以 JSON 格式打印出所有用例的 `test_id`、`nnodes`、`ngpu` 及环境变量。确认你的 `test_name` 正确显示且无 Python 报错即可。

---

### 步骤五：提交 PR 并在 GitHub 触发运行

将代码推送到组织仓的 `ci/*` 验证分支，创建目标为 `master` 的 PR；在 PR 阶段使用 `gh workflow run --ref ci/<分支名>` 手动验证，完成回归后再合并。不要以个人仓 Actions 作为验收入口。

---

### 2.3 在组织仓 PR 中调试 CI

正式 Actions 仅部署于 [TorchTitan-NPU 组织仓](https://github.com/torchtitan-npu/torchtitan-npu)。个人 fork 的 Actions 已禁用，模型侧的 `prepare_request.py` 和 `wait_result.py` 对仓库身份实施组织仓白名单校验，其他仓库无法产生受信任的训练请求。

调试时在组织仓建立 `ci/*` 分支并创建到 `master` 的 PR，在该分支使用 `workflow_dispatch` 触发。触发身份由 GitHub 的组织仓写入权限控制；调度机必须独立配置精确分支白名单（例如 `A3_CI_ALLOWED_BRANCHES=ci/lite-actions-validation`）。GitHub Waiter 还要求结果评论的作者与本次 Workflow 触发者一致，以防伪造 PASS。

**验收原则：** PR 合并前分别确认 Matrix Prepare、Dispatcher 派单、NPU 真实训练、退出码、TensorBoard、资源释放与 GitHub 结果；A5 64P 在正式开放前只验证预期的禁用准入行为。不要用历史 Run 代替当前 PR SHA 的训练通过证明。

---


## 4. 测试断言与排障说明

### 4.1 成功判定条件
CI 判定用例 PASS 必须同时满足：
1. 训练主进程 Exit Code 为 0；
2. 实际完成步数达到 `expected_steps`；
3. TensorBoard 正确记录且无 NaN / Inf 异常；
4. 训练结束后 NPU 物理资源成功释放。

### 4.2 常见错误与排查建议
* **`ASCEND_SET_ENV_PATH` 或权重路径找不到**：
  检查 `env_vars` 中的路径是否在所有执行机（如 `/mnt/share/...` 挂载点）均真实存在。
* **显存 OOM**：
  若在反向传播或优化器更新时 OOM，确认是否需要开启 `OPTIMIZER_OVERRIDES`（如 `virtual` 模式）。
* **多机通信超时 (HCCL Timeout)**：
  确认两台机器的通信小网 IP 是否连通，以及 CANN 环境变量是否完整。
* **物理机器或调度故障**：
  若任务一直排队未执行或出现网络异常，请联系集群运维或查阅 Lite Actions 调度系统运维文档。
