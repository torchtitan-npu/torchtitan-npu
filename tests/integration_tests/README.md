# TorchTitan-NPU 集成测试指南

本目录遵循 TorchTitan 的 `tests/integration_tests` 架构，负责维护集成测试（Integration Tests）定义、执行入口、数值精度对齐校验以及集群 Nightly CI 调度规范。

根据运行环境与执行规模的不同，仓内集成测试分为两种核心形态：
1. **流水线典型用例冒烟测试（A3 4die环境）**：面向 PR 门禁与日常开发快速回归，在配备 1~4 卡（4-die）的开发机或流水线容器内运行，通过仓内集成的并行 Runner 执行典型短步数用例，支持 Deterministic 数值 Loss 精确逐值比对与核心算子冒烟；
2. **集群 Nightly 全量用例测试**：面向真实的 A3 / A5 物理多机集群（单机 8P / 双机 16P / 八机 64P），覆盖生产级完整规格、跨机分布式通信拓扑与真实权重资产，通过内网 [Lite Actions](https://github.com/depeng1994/lite-actions) 调度机排队执行。

---

## 一、流水线典型用例冒烟测试（A3 4die环境）

### 1.1 运行入口与基本命令

#### 脚本入口
CI 流水线主要通过以下脚本一键拉起集成测试：

```bash
.ci/integration_test.sh
```

该脚本会自动准备 CANN 环境变量与运行解释器环境。

#### Python 原生 Runner 入口
在开发机上手动运行测试时，可直接调用 Python 模块：

```bash
python -m tests.integration_tests.run_tests \
  ./test_reports/integration \
  --test_suite models \
  --ngpu 4
```

> **参数说明**：
> - `./test_reports/integration`：**必填**测试输出目录，运行前必须确保为空；
> - `--test_suite`：指定要执行的测试套件（如 `models`、`deepseek_v4_checkpoint` 等）；
> - `--ngpu`：指定单次测试使用的总卡数；
> - `--test_name`：（可选）仅执行该套件内的单条特定用例；
> - `--no-parallel`：（可选）强制串行执行，默认会开启多用例并行。

---

### 1.2 冒烟测试全景矩阵（1~4 卡）

4die 冒烟测试包含 4 个核心 Suite：
* **`models`**：DeepSeek-V4、DeepSeek-V3.2 多卡基础训练冒烟与 Loss 对齐校验；
* **`deepseek_v4_checkpoint`**：DeepSeek-V4 分布式断点续训两阶段比对；
* **`deepseek_v4_quantized`**：DeepSeek-V4 Block-FP8 量化（需 Ascend 950 芯片）；
* **`deepseek_v4_1_engram_hf`**：DeepSeek-V4.1 Engram MXFP8 专项校验与 DCP/HF 权重回环测试。

#### 核心用例清单矩阵

| Case 名称 | 对应模型 | 并行配置 | 卡数 (ngpu) | Check Loss | 核心校验点与说明 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`dsv4_golden_1rank`** | DeepSeek-V4 | 1 Rank 参考配置 | 1 | **是** | 严格 Deterministic 模式，精确逐值比对参考 Loss |
| **`dsv4_golden_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2 | 2 | **是** | 分布式并行环境下的数值 Loss 精确比对 |
| **`dsv4_checkpoint_resume_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2 + AdamW NovaSwap | 2 | **动态比对** | 两阶段连续训练 vs 断点恢复训练，逐步比对 Loss 与 Grad Norm |
| **`dsv4_lora_ep2_fsdp2`** | DeepSeek-V4 LoRA | EP2 + FSDP2，训练 LoRA A/B | 2 | 否 | 校验基座参数冻结、Adapter 权重更新及 PEFT 导出完整性 |
| **`dsv4_muon_swap_ep2_fsdp2`** | DeepSeek-V4 | NPU 融合算子 + DistMuon NovaSwap | 2 | 否 | NPU 融合算子（RMSNorm/RoPE/SparseAttention）与 Swap 路径冒烟 |
| **`dsv4_anticipatory_recovery`** | DeepSeek-V4 | 1 Rank 原生 Checkpoint 回滚 | 1 | 否 | 一次性扰动训练 Loss，断言 Override 机制与完整恢复周期 |
| **`dsv4_smla_1rank_aot_eager`** | DeepSeek-V4 | 1 Rank，`aot_eager` 编译 | 1 | 否 | SMLA 算子图在单卡下的编译构图与前反向冒烟 |
| **`dsv4_smla_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2，`aot_eager` | 2 | 否 | SMLA 算子在 EP+FSDP 分布式环境下的执行与 MoE Dispatcher 验证 |
| **`dsv4_smla_cp2_ep2_fsdp2`** | DeepSeek-V4 | CP2 + EP2 + FSDP2 | 4 | 否 | 混合并行（Context Parallel + Expert Parallel）算子冒烟 |
| **`dsv4_mtp_smla_cp2_headtail`** | DeepSeek-V4 MTP | CP2 + headtail | 2 | 否 | 覆盖 MTP forward、chunked loss 及 backward 算子路径 |
| **`dsv3_2_dsa_1rank`** | DeepSeek-V3.2 | 1 Rank，DSA 算子 | 1 | **是** | 校验 DSA attention override 并在 1-rank 下做数值 Loss 比对 |
| **`dsv3_2_dsa_ep2_fsdp2`** | DeepSeek-V3.2 | DSA + EP2/FSDP2 | 2 | **是** | 校验分布式 DSA 算子 100-step 精度对齐 |
| **`dsv3_2_dsa_cp2`** | DeepSeek-V3.2 | DSA + CP2 | 2 | 否 | 校验 DSA 在 CP2 模式下的训练触发与通信 |
| **`dsv4_ema_ep2_fsdp2`** | DeepSeek-V4 | Golden + EP2/FSDP2 + EMA CPU Offload | 2 | 否 | 校验 DCP Checkpoint Metadata 完整包含 `ema_optimizer.*` |

---

### 1.3 核心校验机制与调度原则

#### 1. 精度对齐判定原则：`use_golden` 与 `check_loss`
集成测试严格区分**算子实现选择**与**数值精确对齐**两个独立维度：
* **`use_golden`**：决定模型使用 Golden 参考标准实现，还是使用 SMLA / NPU 专用算子优化 override；
* **`check_loss`**：决定是否启用确定性计算（`--debug.deterministic`）、固定随机种子，并严格读取参考基准 Loss 执行逐值比对。

**判定规则**：
1. 设置 `check_loss=True` 的用例（如 Golden 用例与 DSA 对齐用例），会自动比对 TensorBoard 中的标量 `loss_metrics/global_avg_loss`，**要求执行步骤集合和每一个浮点数值均严格相等（不允许浮点容差）**；
2. 设置 `check_loss=False` 的用例主要用于验证算子图在特定编译模式（如 `aot_eager`）或并行拓扑下的稳定性与执行路径，只要正常完成指定步数且主进程退出码为 0 即判定通过。

#### 2. 核心特色用例设计机制
* **DCP 动态断点续训比对 (`dsv4_checkpoint_resume_ep2_fsdp2`)**：
  采用**两阶段动态自校验**设计，严格验证在分布式 AdamW NovaSwap 场景下的 Checkpoint 保存与恢复一致性：
  - **阶段一（连续基准）**：固定 `seed=42` 开启确定性模式，连续训练 4 步，并在 step 2 保存完整 DCP Checkpoint；
  - **阶段二（续训恢复）**：新进程通过 `--checkpoint.load-step=2` 加载 step 2 权重，继续训练第 3、4 步；
  - **断言比对**：比对两次运行在 TensorBoard 中的 `global_avg_loss` 与 `grad_norm`。第 3、4 步数值必须逐值精确相等，不舍入、不设容差。
  ```bash
  python -m tests.integration_tests.run_tests ./test_reports/lora-resume \
    --test_suite deepseek_v4_checkpoint \
    --test_name dsv4_checkpoint_resume_ep2_fsdp2 \
    --ngpu 2
  ```

* **LoRA 训练与 PEFT 导出 (`dsv4_lora_ep2_fsdp2`)**：
  在模型并行化后冻结 Base 基座权重，仅针对 LoRA A/B 结构训练 2 步：
  - 检查冻结层参数与 routing bias 保持不变；
  - 检查可训练 adapter 权重正常更新；
  - 训练结束后通过真实 Checkpointer 导出 PEFT 权重，自动断言导出文件的 key、shape、数值以及 rank/alpha 配置。

* **Block-FP8 量化场景 (`deepseek_v4_quantized`)**：
  需要在支持 Block-FP8 的 Ascend 950 芯片及对应 CANN 运行时下运行，依赖 `torchao==0.17.0` 与可选扩展包 `experiments/torchao-npu`：
  ```bash
  # 安装扩展包
  pip install ./experiments/torchao-npu

  # 执行量化测试
  python -m tests.integration_tests.run_tests ./test_reports/lora-fp8 \
    --test_suite deepseek_v4_quantized \
    --test_name dsv4_lora_block_fp8_ep2_fsdp2 \
    --ngpu 2
  ```

* **Engram HF 专项验证（超节点手动）**：
  在具备 Engram MXFP8 Host 通信环境的超节点上，验证 V4.1 Engram 参数在原生 DCP 与 FP32 HF 权重格式之间的双向回环保存与恢复一致性：
  ```bash
  HF_ASSETS_PATH=/path/to/v41-tokenizer \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  python -m tests.integration_tests.run_tests /tmp/engram-hf-output \
    --test_suite deepseek_v4_1_engram_hf \
    --ngpu 4 --no-parallel
  ```
  成功执行后控制台输出 `HF_ROUNDTRIP PASS`。

#### 3. 4die 并发调度（`GPUPool`）与通信端口防冲突
当执行包含多个用例的套件时，`run_tests` 默认复用 TorchTitan 的 `GPUPool` 机制并发执行测试，自动按卡号切分：
* **设备池管理**：若环境中预设了 `ASCEND_RT_VISIBLE_DEVICES`，仅在指定集合内切分卡；否则自动通过 `torch.npu.device_count()` 获取总卡数。用例按所需卡数由大到小排序提交调度；
* **超时与安全回收**：支持为用例配置超时时间，超时后会自动向训练进程组发送 `SIGTERM`（限时后 `SIGKILL`），杜绝孤儿 `torchrun` 残留占卡；
* **同机多任务防踩踏重点**：
  > [!IMPORTANT]
  > 如果在同一台机器上手动同时启动多个独立的测试进程或分布式 run，必须为各个运行分配**互不重叠的通信端口范围**：
  > ```bash
  > export HCCL_NPU_SOCKET_PORT_RANGE="60000-60020"  # 任务 A
  > export HCCL_NPU_SOCKET_PORT_RANGE="60021-60040"  # 任务 B
  > ```
  > 否则节点间 HCCL 端口绑定冲突会导致初始化死锁或通信超时。

---

## 二、集群 Nightly 全量用例测试（A3 / A5 真实集群）

### 2.1 测试矩阵与规格定义

Nightly CI 用例专为真实物理集群设计，由内网 [Lite Actions](https://github.com/depeng1994/lite-actions) 调度机通过 SSH 远程拉起：

| Suite 名称 | Case ID | 硬件规格 | 优化器与训练配置 | 验证指标 |
| :--- | :--- | :--- | :--- | :--- |
| **`a3_8p_tests`** | `dsv4_flash_a3_8p_example` | A3 单机 8P | Muon，Eager，5 steps | Exit Code 0、TensorBoard 步数 |
| **`a3_8p_tests`** | `dsv4_flash_a3_8p_adamw` | A3 单机 8P | AdamW + Virtual Optimizer，4K seq，5 steps | Exit Code 0、TensorBoard 步数 |
| **`a3_16p_tests`** | `dsv4_flash_a3_16p_example` | A3 双机 16P | AdamW，EP16，Eager，双机各 8P，5 steps | Exit Code 0、跨机通信与步数 |
| **`a5_64p_tests`** | `dsv4_pro_a5_64p` | A5 八机 64P | DeepSeek-V4 Pro，八机各 8P | 待硬件环境验收后开放 |

---

### 2.2 本地极简调试入口（物理机手动直跑）

若直接在配备 NPU 的物理机器上手动自测 Nightly 用例，无需通过 GitHub Actions，使用单行命令即可拉起：

#### 单机 8P（单节点自测）
单机用例无需指定跨机 IP，内部自动通过 `run_tests` 执行：
```bash
python3 -m tests.integration_tests.tools.lite_actions.entrypoint launch \
  --test-id dsv4_flash_a3_8p_adamw \
  --output-dir /tmp/test_8p
```

#### 双机 16P（跨节点分布式自测）
两台机器分别执行同一行命令，通过前置 `NODE_IPS` 指定两节点的通信 IP 即可：
```bash
NODE_IPS="192.168.1.3,192.168.1.4" python3 -m tests.integration_tests.tools.lite_actions.entrypoint launch \
  --test-id dsv4_flash_a3_16p_example \
  --output-dir /tmp/test_16p
```
> **提示**：脚本会自动根据本机 IP 匹配并确定 Master（Rank 0）与 Worker（Rank 1）角色；环境中的单节点卡数等参数已在用例定义内配好，无需额外 export。

---

### 2.3 开发者实操：如何新增一个 Nightly 集成用例

向 Nightly CI 系统添加新用例非常简单，**只需在模型仓修改 Python 代码，无需改动调度机配置**。

#### 步骤一：找到用例定义文件
Nightly 用例统一存放于：
`tests/integration_tests/nightly_all_models_test/`

根据用例的卡数规格，编辑对应文件（或新建文件并在 `__init__.py` 中暴露）：
- 单机 8P 用例：`a3_8p_tests.py`
- 双机 16P 用例：`a3_16p_tests.py`
- 八机 64P 用例：`a5_64p_tests.py`

#### 步骤二：编写 `OverrideDefinitions`
每个用例都是一个 `OverrideDefinitions` 对象。核心字段说明如下：

| 字段名称 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`test_name`** | `str` | 用例唯一标识（不可重复，用于 GitHub Matrix 调度命名与结果报告） |
| **`nnodes`** | `int` | 所需节点数（单机写 `1`，双机写 `2`） |
| **`ngpu`** | `int` | 单节点卡数（通常为 `8`） |
| **`override_args`** | `dict` | 传给训练启动脚本的命令行参数（覆盖默认 toml 配置） |
| **`env_vars`** | `dict` | 运行时环境变量（CANN 脚本路径、模型权重路径、优化器覆盖等） |
| **`expected_steps`** | `int` | 预期训练步数（用于 TensorBoard 与日志断言） |

#### 步骤三：代码编写模板
打开 `tests/integration_tests/nightly_all_models_test/a3_8p_tests.py`，在 `build_test_list()` 中添加新的用例定义：

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
                "--job.config_file": "torchtitan_npu/models/deepseek_v4/train_configs/dsv4_flash_8p.toml",
                "--training.steps": "5",
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

#### 步骤四：本地静态语法与元数据解析自测（无需 NPU）
在提交代码前，在普通的 CPU 开发机环境下执行 `inspect` 子命令，验证用例语法与参数是否合法：

```bash
# 验证 8P suite 能够正确解析并包含你的新增用例
python3 -m tests.integration_tests.tools.lite_actions.entrypoint inspect --suite a3_8p_tests
```

输出将以 JSON 格式打印出所有用例的 `test_id`、`nnodes`、`ngpu` 及环境变量。确认你的 `test_name` 正确显示且无 Python 报错即可。

---

## 三、CI 门禁流程与调度安全准入

为了保证内网物理 NPU 集群的执行安全，防止恶意 PR 注入未经审查的代码，CI 调度实行**受控安全准入**机制：

### 3.1 组织仓 PR 自动 CI 触发
1. **自动触发**：所有面向 `torchtitan-npu/torchtitan-npu:master` 提交的 PR（含来自外部 Fork 仓库的 PR）会自动触发组织仓的 GitHub Actions 工作流；
2. **安全防护**：
   - 外部 Fork 的首次 PR 受 GitHub 官方的 Workflow 审批策略保护（需要 Maintainer 批准运行）；
   - 调度机根据 PR API 严格比对 PR 编号、目标分支以及精确的 Head Commit SHA，坚决拒绝过期或关联异常的 Run；
   - 调度机在拉取代码时，仅下载该特定 PR Head SHA 对应的代码，杜绝中间人篡改；
   - 执行机**完全不持有 GitHub Token**，仅内网调度机单点具有回写 Comment 权限。

### 3.2 手动调度（Maintainer 冒烟与调试）
维护者亦可通过 GitHub Actions 网页界面或 GitHub CLI 手动发起调度：

```bash
# 执行 A3 8P 全部用例（Muon + AdamW 独立 Matrix Case，资源不重叠即可并发）
gh workflow run nightly_fulltests.yml -R torchtitan-npu/torchtitan-npu --ref master \
  -f 'test_cases=[{"suite":"a3_8p_tests"}]'

# 仅执行指定 16P 用例
gh workflow run nightly_fulltests.yml -R torchtitan-npu/torchtitan-npu --ref master \
  -f 'test_cases=[{"test_id":"dsv4_flash_a3_16p_example"}]'
```

### 3.3 CI 矩阵并发与动态端口分配

在 GitHub Actions CI 流水线中，每个 Matrix Case 是独立的调度与执行单元：
* **并发调度**：Lite Actions 调度机会根据物理集群的空闲 NPU 资源池，将互不冲突的 Case 并发拉起执行（如单机 8P 各用例并发执行，或双机 16P 跨节点独立分配）。
* **动态端口防冲突**：当同机或跨机并发拉起多个任务时，调度机会为每个任务动态分配互不冲突的通信端口（通过注入环境变量 `LITE_CI_MASTER_PORT`、`LITE_CI_HCCL_IF_BASE_PORT`、`LITE_CI_HCCL_HOST_PORT_RANGE` 与 `LITE_CI_HCCL_NPU_PORT_RANGE`）。仓内的 `nightly_all_models_test/runner.py` 会自动读取并映射这些变量，覆盖用例的默认端口。开发者编写用例时无需硬编码特殊端口，使用默认值即可。

### 3.4 硬件环境与调度故障排障
若任务长时间卡在排队状态、远端 NPU 报告 Quarantine 或出现 SSH 连接异常，属于物理集群调度机层面的环境状态问题，请参阅 [Lite Actions 调度系统文档](https://github.com/depeng1994/lite-actions) 的日常运维与应急排障小节。


## Lite Actions 2.0 调度配置

只保留 `nightly_fulltests.yml` 与 `lite_actions_dispatcher_smoke.yml` 两个 Lite Actions 业务入口，均支持手动与面向 master 的 PR 触发。默认不增加定时任务。

Workflow 提供 `default_worker`、`default_target_pool`、`default_docker`。Python 的 `ci_overrides` 按字段覆盖，缺省继承；测试拓扑仍由 `nnodes` 和 `ngpu` 定义。

```python
ci_overrides={"worker": "yellow", "target_pool": "a3_shared_yellow", "docker": "已登记的镜像引用"}
```

`ci_kind` 默认为 `train`；NPU 冒烟为 `npu_smoke`；`worker_probe` 仅使用 worker 身份，不继承训练资源。完整执行计划使用 schema 3，与源码 SHA、run、attempt 和逐 case 身份绑定。

现有 A3 只由 blue (`ssh ycy`) 管理：`113.46.17.205:153`、`113.46.17.205:154`。默认环境为 `dsv4.1_train_torchtitan_a3:cann_master_newest_0916`，由部署配置绑定已有容器。yellow (`ssh hwy`) 首版只验收系统探测，`a3_shared_yellow` 和 `a5_shared_yellow` 保持禁用。

每个 case 回传日志最多 20 行、每行最多 180 字符；继续限制 Comment 正文和序列化请求体大小。Summary 只显示配置与状态，不追加训练日志、TensorBoard 或训练产物。
