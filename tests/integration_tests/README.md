# 集成测试基础设施

> **Lite Actions V2 Phase 1（分层发布状态）**：调度侧 depeng1994/lite-actions 已合入 main 并由生产 V2 Dispatcher 运行，V1 协议兼容 Smoke [38061822173](https://github.com/depeng1994/torchtitan-npu/actions/runs/38061822173) PASS；本模型仓的 8P/16P V2 Matrix Workflow 仍位于 refactor/v2-phase1-serial / Draft PR #28，尚未合入 master。特性分支已有 8P/16P 实机验收证据，但不能冒充模型 master 发布证据。A5 Workflow 有定义但通道持续禁用，且无实机验收。


本目录遵循 Torchtitan 的 `tests/integration_tests` 布局，负责维护集成测试定义、测试入口以及可选的 loss 精确比较。基础架构代码由
torchtitan 迁移而来。

默认 `models` suite 覆盖 DeepSeek-V4、DeepSeek-V3.2 和 Qwen3.5；checkpoint、量化及 Engram HF 另有独立 suite。

## 入口

CI 通过以下脚本启动本目录的 integration ST：

```bash
.ci/integration_test.sh
```

`.ci/smoke_test.sh` 只负责 `tests/smoke_tests` 的 smoke 阶段，不再触发本目录的 ST。
两个入口都先 source `.ci/common.sh`，由它完成 CANN 环境、解释器 shim 和 torchtitan
checkout 准备。

或直接运行 Python 入口：


```bash
python -m tests.integration_tests.run_tests \
  ./test_reports/integration \
  --test_suite models \
  --ngpu 4
```
其中`./test_reports/integration` 是必填的测试输出目录，运行前需要确保该目录为空。

直接运行上述 Python 命令仅执行 integration tests。`--test_suite models` 与 CI 的集成测试配置保持一致，覆盖 DeepSeek-V4 和 DeepSeek-V3.2。完整 CI 流程还会在此之前执行 `tests/smoke_tests`。

## 主要测试矩阵

| Case 名称 | 模型 | 并行配置 | Rank 数 | 编译配置 | Check Loss | 不检查 Loss 原因 |
|---|---|---|---|---:|---|---|
| `dsv4_anticipatory_recovery` | DeepSeek-V4 | 1 Rank、原生 checkpoint 回滚 | 1 | - | 否 | 测试入口一次性扰动真实训练 loss，断言 override 和完整恢复周期 |
| `dsv4_golden_1rank` | DeepSeek-V4 | 1 Rank 参考配置 | 1 | - | 是 | - |
| `dsv4_golden_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | - | 是 | - |
| `dsv4_muon_swap_ep2_fsdp2` | DeepSeek-V4 | NPU 融合算子 + DistMuon/AdamW NovaSwap、EP2 + FSDP2、2 steps | 2 | - | 否 | 两步训练 smoke；未生成 swap 数值 golden，也未单独断言 swap action |
| `dsv4_lora_ep2_fsdp2` | DeepSeek-V4 LoRA | EP2 + FSDP2，固定基座，训练 LoRA A/B，2 steps | 2 | - | 否 | 检查冻结参数与 routing bias 不变、adapter 更新及最终 PEFT 导出；不做中间保存或恢复 |
| `dsv4_checkpoint_resume_ep2_fsdp2` | DeepSeek-V4 | Golden recipe + AdamW NovaSwap、EP2 + FSDP2，step 2 恢复到 step 4 | 2 | - | 否，动态精确比较 loss/grad_norm | 与本次连续训练的 step 3、4 精确比较；不读取仓内 golden loss |
| `dsv4_smla_1rank_aot_eager` | DeepSeek-V4 | 1 Rank | 1 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_cp2_ep2_fsdp2` | DeepSeek-V4 | CP2 + EP2 + FSDP2 | 4 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_mtp_smla_cp2_headtail` | DeepSeek-V4 MTP | CP2 + headtail | 2 | - | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv3_2_dsa_1rank` | DeepSeek-V3.2 | 1 Rank，DSA | 1 | - | 是 | - |
| `dsv3_2_dsa_ep2_fsdp2` | DeepSeek-V3.2 | DSA + EP2/FSDP2 | 2 | - | 是 | - |
| `dsv3_2_dsa_cp2` | DeepSeek-V3.2 | DSA + CP2 | 2 | - | 否 | ST 仅验证训练触发；CPU metadata oracle 单独覆盖，暂未生成 CP2 golden loss |
| `dsv4_ema_ep2_fsdp2` | DeepSeek-V4 | Golden + EP2/FSDP2 + EMA CPU offload | 2 | - | 否 | 校验完整 DCP metadata 包含 `ema_optimizer.*` |

`use_golden` 与 `check_loss` 是两个独立维度：`use_golden` 仅决定使用 Golden 参考算子
还是 SMLA/NPU override；`check_loss` 决定是否启用 deterministic、读取参考 loss 并执行
精确数值比较。

当前两个 Golden case（均为 V4）设置 `check_loss=True`，使用固定随机种子和 deterministic 模式，
比较 TensorBoard 标量 `loss_metrics/global_avg_loss`，要求 step 集合和每个浮点值均精确相等。
DeepSeek-V4.1 的常规训练验证使用 [A3/A5 示例入口](../../examples/deepseek_v4_1/readme.md)。Engram HF 验证使用下述独立 suite，不在默认 `models` 或 CI smoke 中执行。

两个 DeepSeek-V3.2 case 同样设置 `check_loss=True`，使用 RoPE workaround、Ascend DSA
metadata/attention override，并分别对 1-rank 和 EP2/FSDP2 的 100-step loss 做精确比较。

`dsv4_checkpoint_resume_ep2_fsdp2` 覆盖 AdamW NovaSwap 的 checkpoint 保存、恢复和精度对齐，已注册到
独立的 `deepseek_v4_checkpoint` suite，不在默认 `models` suite 中执行。它使用两卡 EP2 + FSDP2、Golden recipe、`swap_optimizer` override 和
`--optimizer.name=AdamW`，设置 `use_golden=True` 与 `check_resume=True`，
固定 seed=42 并开启 deterministic。第一阶段连续训练 4 步，保留 step 2 的完整 checkpoint；
第二阶段在新进程中通过 `--checkpoint.load-step=2` 恢复，再训练第 3、4 步。
两阶段均设置 `--training.steps=4`，确保学习率调度一致，共用 checkpoint 目录，
分别写入 `tb_phase_0` 和 `tb_phase_1`。检查 TensorBoard 的
`loss_metrics/global_avg_loss` 和 `grad_norm`：步骤集合必须分别为 `(1, 2, 3, 4)` 和 `(3, 4)`，
续训两步的两个标量必须与连续训练逐值精确相等，不舍入、不使用容差；缺失、重复步骤或
非有限值均失败。本次连续训练是动态基准，不读取或更新仓内 golden loss 文件。

单独执行此用例：

```bash
python -m tests.integration_tests.run_tests /tmp/checkpoint_resume_output \
  --test_suite deepseek_v4_checkpoint --test_name dsv4_checkpoint_resume_ep2_fsdp2 --ngpu 2
```

`dsv4_lora_ep2_fsdp2` 在实际模型并行化后冻结基座、训练 LoRA A/B。
连续训练 2 步，检查冻结参数与 routing bias 不变、可训练 adapter 更新。
最后一步通过实际 checkpointer 导出 PEFT，检查 adapter 文件的 key、shape、数值和配置中的 rank、alpha、target。
该用例不做中间 checkpoint 保存或恢复。冻结 LoRA A、仅训练 B 的差异由 CPU 单元测试覆盖。

四个 SMLA case 都设置 `check_loss=False`，因此不会启用 `--debug.deterministic`，也不会
读取 golden loss。它们用于覆盖 SMLA/NPU override 在单卡、EP+FSDP、CP+EP+FSDP 以及
MTP+CP 场景下的实际构图、编译和训练执行路径；单卡、EP2 和 CP2+EP2 场景均使用
`aot_eager`，并默认覆盖 fused MoE token dispatcher。MTP+CP 用例固定使用
`deepseek_v4_debugmodel`、CP2 和 headtail，在 C4 packed sequence 上执行完整的
MTP forward、chunked loss 和 backward。

`dsv4_muon_swap_ep2_fsdp2` 保留 NPU 融合算子 recipe：Ascend RMSNorm、complex RoPE、sparse
attention、MHC 和 MoE token dispatcher；两卡 EP2/FSDP2，显式选择 `--optimizer.name=Muon`
并启用 `swap_optimizer` override。它运行两步，覆盖 DistMuon momentum state 与其 AdamW fallback
state 的 NovaSwap 路径。该 case 同样只检查训练完成，不读取 golden loss，也不声称数值等价。

这里的 integration recipe 聚焦 sparse-attention / MHC 回归边界。端到端 example 脚本
额外启用 Virtual Optimizer；checkpoint 保存兼容由 extension `CheckpointManager` 提供。
这些 optimizer state/checkpoint 路径不属于当前 integration loss regression 的覆盖范围。

## Nightly All Models（A3 / A5 CI 用例）

GitHub 的正式 `*-lite-actions.yml` Workflow 配合独立的 [lite-actions](https://github.com/depeng1994/lite-actions) 调度仓；每条模型测试由本仓的 `tests/integration_tests/nightly_all_models_test/` 中的 `OverrideDefinitions/build_test_list()` 定义，入口固定为 `tests.integration_tests.tools.lite_actions.entrypoint`。无需 `ci_registry.json`、为单个模型新增调度分支或复制 Workflow。

| Suite / 测试用例 | 训练内容 | 当前验收状态 |
| --- | --- | --- |
| `a3_8p_tests` / `dsv4_flash_a3_8p_example` | A3 单机 8P、Muon、Eager、5 steps | V2 [Run 38054910957](https://github.com/depeng1994/torchtitan-npu/actions/runs/38054910957) 单 Case 5 steps/TB/rc0 PASS；同 Run AdamW 曾外部 SIGKILL，Run 整体失败，未隐去 |
| `a3_8p_tests` / `dsv4_flash_a3_8p_adamw` | A3 单机 8P、AdamW + Virtual Optimizer、Eager、5 steps | V2 [Run 38059258896](https://github.com/depeng1994/torchtitan-npu/actions/runs/38059258896)，SHA `99cd7e3`，4096 seq、5 steps/TB、rc0，独立 GitHub Job PASS |
| `a3_16p_tests` / `dsv4_flash_a3_16p_example` | A3 双机 16P、AdamW、EP16、Eager、5 steps | V2 动态选卡在 [Run 38054913344](https://github.com/depeng1994/torchtitan-npu/actions/runs/38054913344) 基于 `99cd7e3` **PASS**：A3-3 0–7 + A3-4 8–15，双节点 rc=0、TensorBoard 5 steps、GitHub Job Success |
| `a5_64p_tests` / `dsv4_pro_a5_64p` | A5 八机 64P、DeepSeek-V4 Pro | 禁用：真实 CANN/HF/Checkpoint 资产及 HCCL 网络未配置、未实机验收 |

`workflow_dispatch.inputs.test_cases` 只支持真实 test ID 或可信 suite；不接受自定义 Python 路径、CLI、`STEPS` 或 `params`：

```bash
# 一次性依次执行 Muon、AdamW 两条 8P case
gh workflow run a3-8p-lite-actions.yml -R depeng1994/torchtitan-npu --ref master \
  -f 'test_cases=[{"suite":"a3_8p_tests"}]'

# 仅执行一个 16P case
gh workflow run a3-16p-lite-actions.yml -R depeng1994/torchtitan-npu --ref master \
  -f 'test_cases=[{"test_id":"dsv4_flash_a3_16p_example"}]'
```

`OverrideDefinitions.env_vars` 由模型用例分别声明 `ASCEND_SET_ENV_PATH`、`HF_ASSETS_PATH`、`CKPT_INIT_LOAD_PATH`、优化器开关、`TORCHINDUCTOR_NPU_BACKEND` 和模型特定端口；调度机从**请求绑定的 Commit SHA** 的源码解析它们，source 对应 CANN、export 环境后启动。物理 SSH 地址、HCCL 小网 IP、NPU ID、资源锁及动态输出 `CKPT_SAVE_LOAD_PATH` 由 Lite Actions 管理。换模型或升级 CANN 只修改本仓测试定义，不修改 Lite Actions 的模型配置。

8P AdamW 保持原始 4096 序列长度，并通过 `env_vars["OPTIMIZER_OVERRIDES"]="torchtitan_npu.override.common.optimizer.virtual"` 选择 Virtual Optimizer，替换（而不是叠加）Shell 默认的 `swap_optimizer`，保持全部 NPU 算子 imports，不修改 Muon 用例。Virtual Optimizer 将 AdamW moments 使用 Host-backed swap memory，以降低 HBM 占用；当前 8P Muon/AdamW 双用例已在上述 Run 38034002241 中完整 PASS；该结果覆盖 Eager smoke，不覆盖 Inductor、golden 或长期稳定性。

**历史 V1** 同一个 8P Suite 在单 Job 中汇总结果；**V2 Phase 1** 拆为两个 GitHub Matrix Jobs，每个 Case 独立 PASS/FAIL，Dispatcher 全局最多一个训练 Case。V2 的 [Run 38050639931](https://github.com/depeng1994/torchtitan-npu/actions/runs/38050639931) 两个 8P Case 分别 5 steps、rc=0、GitHub 独立 PASS；新 SHA `99cd7e3` 的回归另见计划台账。Eager PASS 不代表 Inductor、数值 golden 或 A5 64P 实机通过。

**GitCode 同步注意：** `Sync Upstream` 每日镜像 GitCode 同名分支；发布应将包含 GitCode/GitHub 双方历史的相同 Release SHA 以快进方式提交到两侧 `master`，否则可能覆盖 Workflow。正式发布仅允许 `master`，特性分支临时验收白名单必须撤除。

## 并行调度（单机 Integration Runner）

`python -m tests.integration_tests.run_tests` 默认复用 TorchTitan 的 `GPUPool` 机制并发执行用例；每个用例通过 `ASCEND_RT_VISIBLE_DEVICES` 绑定**互不重叠**的 NPU，池内同时使用的 NPU 数不超过可用数量。需要串行运行时使用 `--no-parallel`。

- **设备池来源**：如果环境已设置 `ASCEND_RT_VISIBLE_DEVICES`，只使用显式分配的设备；`--ngpu` 超出该集合时直接报错。否则通过 `torch.npu.device_count()` 获取运行时可见数量；`--ngpu` 超出时告警并截断，不凭参数虚构不存在的设备 ID。
- **调度与资源不足**：用例按 `ngpu` 从大到小提交；超过实际设备池容量的用例明确跳过，避免在 `acquire()` 中永久等待。
- **日志与超时**：各用例的输出按 case 名汇总，避免并发日志交错。可通过 `OverrideDefinitions.timeout` 设置超时；超时后先向训练进程组发送 `SIGTERM`，必要时再发送 `SIGKILL`，并将该用例判为失败，防止遗留 `torchrun` rank 继续占卡。
- **并发诊断**：执行结束输出 `[parallel] pool`（设备池利用率、忙碌时长及分配统计）与 `[parallel] overlap`（并发重叠时长及相对串行运行的节省时间），用于判断测试是否实际并发运行。

**同机多任务注意**：如果手动在同一台机器上并发启动多个独立 HCCL 训练 run，应为各 run 分配**不同的** `HCCL_NPU_SOCKET_PORT_RANGE`，避免端口绑定冲突。

此节仅描述模型仓的**单机用例并发**，不等于 Lite Actions 的**跨机器 SSH 调度和资源锁**；后者的设备分配、外部占用检查与任务排障详见 [Lite Actions README](https://github.com/depeng1994/lite-actions/blob/main/README.md)。

## LoRA training and resume

The default `models` suite includes `dsv4_lora_ep2_fsdp2` for training and PEFT export. The distributed DCP resume case, `dsv4_lora_resume_ep2_fsdp2`, runs separately in `deepseek_v4_checkpoint` (two NPUs each). The resume case compares optimizer state exactly before the first resumed update and checks resumed loss/gradient norm. The block-FP8 case requires `torchao==0.17.0` and the optional `experiments/torchao-npu` package (install with `pip install ./experiments/torchao-npu` from the repository root); it runs real forward/backward/optimizer updates with a quantized frozen base and trainable floating-point adapters. Quantization unit tests follow the repository convention and skip when the optional package is unavailable; explicitly running this NPU case requires the package.

On Ascend 950 devices with the CANN 9.2.0 release runtime (2026-09-09 packages) and block-FP8 support, run the registered quantized case through the same runner:

```bash
python -m tests.integration_tests.run_tests ./test_reports/lora-fp8 \
  --test_suite deepseek_v4_quantized --test_name dsv4_lora_block_fp8_ep2_fsdp2 --ngpu 2
```

The A3 CI `models` suite runs floating-point LoRA training and PEFT export. Distributed DCP resume stays in the dedicated checkpoint suite because the A3 CI runtime fails to load `libscatter_aicpu_kernel.so` during checkpoint planning. Run resume on a runtime that supports this distributed checkpoint path:

```bash
python -m tests.integration_tests.run_tests ./test_reports/lora-resume \
  --test_suite deepseek_v4_checkpoint --test_name dsv4_lora_resume_ep2_fsdp2 --ngpu 2
```

The quantized suite requires the block-FP8 hardware/runtime.

## Engram HF 专项验证（手动）

在支持 Engram MXFP8 Host 通信的超节点上激活 CANN、torch extension 和训练环境，准备 V4.1 tokenizer assets 后执行：

```bash
HF_ASSETS_PATH=/path/to/v41-tokenizer \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
python -m tests.integration_tests.run_tests /tmp/engram-hf-output \
  --test_suite deepseek_v4_1_engram_hf --ngpu 4 --no-parallel
```

输出目录使用新目录。该用例为 V4.1 debug text、四卡 EP4/FSDP4、eager、seq512、GBS4，启用 Engram MXFP8 override，关闭普通模型 FP8 和 optimizer CPU offload。使用仓内 C4 数据及自动生成的微型 HF fixture，不需要正式模型权重。

先训练两步并保存同一模型的原生 DCP 和 FP32 HF 权重，再各自通过真实 CheckpointManager 初始化模型，使用相同的新优化器和数据种子训练三步。每个 rank 校验加载后的 Engram 参数及 MXFP8 缓存/scale 有效行字节，随后对比 TensorBoard loss/grad_norm。失败会使测试返回非零，成功输出 `HF_ROUNDTRIP PASS`。不读取固定 golden，不验证优化器状态续训，也不覆盖正式整包非 Engram 量化权重的导入。
