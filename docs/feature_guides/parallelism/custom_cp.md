# 自定义Context Parallel特性

在大规模语言模型的分布式训练中，上下文并行（Context Parallelism, CP）是突破单卡内存瓶颈、支持超长序列训练的核心技术。在基于 NPU 硬件生态推进 torchtitan 框架适配时，现有技术方案暴露出显著的局限性：
1. PyTorch 原生 CP 设计强绑定于标准的 SDPA 算子，仅提供 RingAttention 或原生 AllGatherKV 的特定实现，无法支持采用复杂稀疏注意力机制的模型如 DeepSeek-V3.2 模型的DSA。
2. 框架需要允许开发者灵活扩展新的 CP 范式，如 UlyssesCP。

## 实现原理
torchtitan_npu在`torchtitan_npu/patches/distributed/custom_context_parallel.py`对原生CP context 进行替换，从而使能自定义的CP context，增强CP的可扩展性。目前已提供如下两个自定义的CP context：
### SDPA Ulysses CP
定义在`torchtitan_npu/distributed/context_parallel/ulysses_cp.py`，为常用的SDPA实现Ulysses风格的CP。通过自定义Context Parallel Context，我们在基于torch_npu提供的融合算子的Attention计算前后插入All-to-All通信算子，将数据从“序列并行分布”转换为“多头维度分布”，使单计算节点获得完整上下文。

### DSA CP
定义在`torchtitan_npu/distributed/context_parallel/dsa_cp.py`，为DeepSeek Sparse Attention提供AllGatherKV风格的CP。本项目在遵循torch原生CP设计逻辑的基础上，通过自定义Context Parallel Context，将注意力部分的forward函数进行替换，对KV相关激活做CP域的AllGather，以此确保注意力计算在CP场景下的正确性。

关于DSA CP的更多原理介绍，参考[技术文档](https://gitcode.com/cann/cann-recipes-train/blob/master/docs/llm_pretrain/deepseekv32_pre_train_optimization.md#自定义CP策略)。

## 配置选项

在训练任务的 TOML 配置文件（例如 `torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_debug.toml`，或实际启动训练时 `--job.config_file` 所指向的路径）中，找到对应的 `[parallelism]` 节，并添加以下配置以启用 Custom CP：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `context_parallel_degree` | int | 1 | Context并行度 |
| `enable_custom_context_parallel` | bool | false | 是否启用自定义CP |
| `custom_context_parallel_path` | str | "" | 自定义CP**类**的完整导入路径 |


### 配置示例：SDPA Ulysses CP（DeepSeek-V3）

首先在配置文件中使能本代码仓的自定义配置，随后在`[parallelism]`节中添加以下配置：
```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"    # 使能本代码仓的自定义配置

[parallelism]
context_parallel_degree = 2
enable_custom_context_parallel = true
custom_context_parallel_path = "torchtitan_npu.distributed.context_parallel.ulysses_cp.UlyssesContextParallelContext"
```

### 配置示例：DSA Context Parallel（DeepSeek-V3.2）

```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"    # 使能本代码仓的自定义配置

[parallelism]
context_parallel_degree = 2
enable_custom_context_parallel = true
custom_context_parallel_path = "torchtitan_npu.distributed.context_parallel.dsa_cp.AscendDSAContextParallelContext"
```

