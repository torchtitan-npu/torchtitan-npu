<div align="center">

# torchtitan-npu

<h4>基于 torchtitan 的昇腾全流程大模型训练适配框架</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](#特性支持概览)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![contributing](https://img.shields.io/badge/contributing-blue)](./CONTRIBUTING.md)

<div align="left">

## Latest News
 - [2026/03] torchtitan-npu 正式开源：在 NPU 上支持 4D 并行等 torchtitan 原生特性，并引入 Swap Optimizer 等 NPU 亲和优化。

## 简介

torchtitan-npu 基于 torchtitan 的训练流程，在继承 torchtitan 原生易用性优势的同时，在昇腾平台上提供稳定、可复现且可分析的训练框架。

目前，torchtitan-npu 已实现对 DeepSeek-V3 / V3.2 及 Llama 系列模型在昇腾 NPU 上的端到端预训练支持。见 [性能基准](#性能基准)。

## 特性支持概览

### 并行能力

| 功能                                                                                | 原生支持 | NPU支持 |
| ----------------------------------------------------------------------------------- | -------- | ------- |
| 4D 并行 (FSDP2/TP/CP/PP)                                                            | ✅       | ✅      |
| 专家并行 (EP/ETP)                                                                   | ✅       | ✅      |
| [自定义 CP (DeepSeek V3.2 CP/SDPA Ulysses CP)](docs/feature_guides/parallelism/custom_cp.md) | ✅       | ✅      |

### torch.compile

| 功能            | 原生支持 | NPU支持                                    |
| --------------- | -------- | ------------------------------------------ |
| `torch.compile` | ✅       | [部分支持](docs/feature_guides/compile.md) |

### 训练精度

| 功能                                                       | 原生支持 | NPU支持 |
| ---------------------------------------------------------- | -------- | ------- |
| MxFP8 量化                                                 | ✅       | ✅      |
| [HiF8 量化](docs/feature_guides/low_precision_training.md) | ✅       | ✅      |

### 训练调试与监控

| 功能                                                     | 原生支持 | NPU支持 |
| -------------------------------------------------------- | -------- | ------- |
| 分布式 Checkpoint                                        | ✅       | ✅      |
| [调试工具](docs/feature_guides/metrics_and_debugging.md) | ✅       | ✅      |

### 性能优化

| 功能                                                     | 原生支持 | NPU支持 |
| -------------------------------------------------------- | -------- | ------- |
| [Swap Optimizer](docs/feature_guides/swap_optimizer.md)  | ❌       | ✅      |
| [NPU 融合算子适配](docs/feature_guides/npu_fused_ops.md) | ❌       | ✅      |

## 项目结构
torchtitan-npu 充分利用了 torchtitan 提供的 ModelConverter 插件化机制。该机制介入模型定义之后、并行策略（如 TP/FSDP）应用之前，支持以非侵入式的方式，通过注册机制对特定模块进行替换或重写。基于此方案，我们实现了融合算子优化、量化支持以及优化器增强等功能。见以下项目结构：
```
torchtitan-npu/
├── torchtitan_npu/     # torchtitan_npu核心源代码
│   ├── config/         # 对Config的补丁
│   ├── converter/      # 基于torchtitan ModelConverter机制的补丁
│   ├── distributed/    # 自定义分布式代码
│   ├── models/         # 基于torchtitan-npu的模型 (如Deepseek-V3.2)
│   ├── patches/        # 其他补丁
│   ├── entry.py        # 启动训练
│   └── __init__.py     # torchtitan-npu 插件修改注入点
├── docs/               # 文档
└── run_train.sh        # 训练启动脚本

```

## 快速开始

### 环境准备

* 硬件：Atlas A3 系列
* 软件版本：
  * CANN==8.5.0（HDK配套版本见 [Ascend开发者文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/releasenote/releasenote_0000.html)）
  * Python>=3.10
  * Python 依赖见 `requirements.txt`。

### 安装 torchtitan-npu

```bash
git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
# 安装依赖
pip install -r requirements.txt
```

### tokenizer 下载

```bash
# 从huggingface下载 DeepSeek V3.2 tokenizer https://huggingface.co/settings/tokens

python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.2 --assets tokenizer
```

### 开始训练

在单卡 NPU 上启动 Deepseek v3.2 debug 模型预训练任务
```bash
# NGPU: 训练任务卡数
# CONFIG_FILE: 训练配置文件路径
NGPU=1 CONFIG_FILE=./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_debug.toml bash run_train.sh
```


## 性能基准

### 2026.02

System: Atlas 800T A3
| Model              | Number of NPUs | Precision | GBS | Local BS | Sequence Length | FSDP | TP  | PP  | CP  | EP  | Tokens / sec | MFU%     |
| :----------------- | :------------- | :-------- | :-- | :------- | :-------------- | :--- | :-- | :-- | :-- | :-- | :----------- | :------- |
| Deepseek V3.2-671B | 64             | FP16      | 16  | 16       | 32768           | 1    | 4   | 2   | 16  | 64  | 30           | 1.88%    |
| Deepseek V3.2-671B | 64             | FP16      | 64  | 4        | 4096            | 16   | 4   | 2   | 1   | 64  | 47           | 暂时缺失 |
| Deepseek V3-671B   | 16             | FP16      | 256 | 16       | 4096            | 16   | 4   | 2   | 1   | 64  | 217          | 暂时缺失 |
| Llama3-70B         | 16             | FP16      | 16  | 8        | 8192            | 2    | 8   | 1   | 1   | 1   | 1059         | 19.66%   |
| Llama3-8B          | 8              | FP16      | 4   | 1        | 8192            | 8    | 1   | 2   | 1   | 1   | 2125         | 39.45%   |

## 免责声明
致torchtitan-npu使用者

1. torchtitan-npu提供的所有内容仅供您用于非商业目的。
2. 对于torchtitan-npu测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用torchtitan-npu过程中，发现任何问题（包括但不限于功能问题、合规问题），请在GitCode提交issue，我们将及时审视并解决。
torchtitan-npu功能依赖的Pytorch等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，torchtitan-npu仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。