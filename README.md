# torchtitan-npu
## 简介

torchtitan 是 PyTorch 生态的并行训练框架，它在 PyTorch 原生并行能力的基础上，实现了模型定义与并行策略的解耦、4D 并行的有机整合，并集成了多种调试与检查点工具，为研究与实践提供了高度易用、可扩展的基础设施。

torchtitan-npu 基于 torchtitan 的训练流程，在昇腾平台上提供稳定、可复现且可分析的训练框架，用于支撑 LLM 训练任务。它在继承 torchtitan 原生易用性优势的同时，围绕训练阶段的关键系统能力，协同运行时与编译栈，实现内存管理、分布式并行、执行调度、算子融合与图优化路径等技术的工程化落地，并提供面向训练性能的分析与调优能力。

目前，torchtitan-npu 已实现对 DeepSeek-V3 / V3.2 及 Llama 系列模型在昇腾 NPU 上的端到端预训练支持。

### 项目结构
torchtitan-npu 充分利用了 torchtitan 提供的 ModelConverter 插件化机制。该机制介入模型定义之后、并行策略（如 TP/FSDP）应用之前，支持以非侵入式的方式，通过注册机制对特定模块进行替换或重写。基于此方案，我们实现了融合算子优化、量化支持以及优化器增强等功能。见以下项目结构：
```
torchtitan-npu/
├── torchtitan_npu/     # torchtitan_npu核心源代码
│   ├── config/         # 对Config的补丁
│   ├── converter/      # 基于torchtitan ModelConverter机制的补丁
│   ├── distributed/    # 自定义分布式代码
│   ├── models/         # 基于torchtitan-npu的模型 (e.g., Deepseek-V3.2)
│   ├── patches/        # 其他补丁
│   ├── entry.py        # 启动训练
│   └── __init__.py     # torchtitan-npu 插件修改注入点
├── docs/               # 文档
└── run_train.sh        # 训练启动脚本
```

## 快速开始
### 环境准备
 - 硬件：Atlas A3 系列
 - 软件版本：
    - CANN==8.5.0（HDK配套版本见[Ascend开发者文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/releasenote/releasenote_0000.html)）
    - Python>=3.10

### 安装 torchtitan-npu
#### 从源代码安装
```
git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
# 安装依赖
pip install -r requirements.txt
```

### tokenizer下载
```
# 从huggingface下载 DeepSeek V3.2 tokenizer https://huggingface.co/settings/tokens

python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.2 --assets tokenizer
```

### 开始训练
使用 Deepseek v3.2 debug 模型启动2卡训练任务。

```shell
NGPU=1 bash run_train.sh
```

### 特性支持

| 类别 | 功能特性 | 状态 |
| :--- | :--- | :---: |
| **分布式并行策略** | MoE的TP策略优化 | ✅ |
| | [自定义CP策略](docs/feature_tutorials/custom_cp.md) | ✅ |
| | DTensor计算的Sharding策略优化 | ✅ |
| **融合算子适配** | LI/SFA | ✅ |
| | SparseLightningIndexerGradKLLoss | ✅ |
| **内存优化** | Swap Optimizer | ✅ |
| **量化** | MxFP8 | ✅ |

## 性能基准
### 2026.02
System: Atlas 800T A3
| Model                                 | #NPU | Precision | GBS | MBS | Sequence Length | FSDP | TP  | PP  | CP  | EP  | Tokens / sec / NPU | TFLOP / sec / NPU |
| :------------------------------------ | :--- | :-------- | :-- | :-- | :-------------- | :--- | :-- | :-- | :-- | :-- | :----------------- | :---------------- |
| Deepseek V3.2 671B                    | 64   | FP16      | 4   | 4   | 65536           | 1    | 4   | 2   | 16  | 64  | 22.00              |                   |
| Deepseek V3.2 671B (torchtitan 0.2.0) | 64   | FP16      | 16  | 16  | 32768           | 1    | 4   | 2   | 16  | 64  | 30.00              |                   |

