# torchtitan-npu

### 执行命令


```shell
# 安装torchtitan以及torchtitan-npu
pip install torchtitan==0.2.0
pip install -e /path/to/torchtitan-npu

# 示例执行，2层（1moe，1dense）裁剪模型（请提前配置toml文件中相关地址）
chmod +x ./torchtitan-npu/run_train.sh
NGPU=4 CONFIG_FILE="./torchtitan-npu/torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_debug.toml" ./torchtitan-npu/run_train.sh

```

### 特性

#### 融合算子替换

- 配置文件的 [model] 中配置 `converters` 使能，支持基础算子`npu_rms_norm`、`npu_rope`、`npu_permute`、`npu_gmm`、`npu_fusion_attention` 和 DeepSeekV3.2 `npu_dsa`
- 可以配置多种替换 如: `converters = ["npu_rms_norm", "npu_rope", "npu_permute"]`
- 注：当前的npu_fusion_attention替换仅为配置sync参数，解决inductor后端使用reduce_overhead的attention算子的多流问题，性能精度均与之前一致。

#### 权重加载：当前支持deepseek_v32

- 支持离线权重转换，根据配置项`use_grouped_mm`进行转换
  - 如果`use_grouped_mm=True` 那么则将普通HF权重转化为gmm titan权重
  - 反之`use_grouped_mm=False` 转化为普通权重
  - ```
    python torchtitan/scripts/checkpoint_conversion/convert_from_hf.py \
    /path/to/input/ \
    /path/to/output/step-0/ \
    --model_name deepseek_v32
    --model_flavor debugmodel
    ```
- 支持直接使用HF权重（自动适配gmm），也支持直接使用titan dcp权重
- 支持自定义设置导出权重
  - ```
    save_format = "dcp", 保存文件类型（"dcp"/"hf"）
    save_expert_format = "standard", 保存expert类型("gmm"/"standard")
    hf_save_dir = "/path/to/output/",
    save_patch_enabled = True, (如果=False，则正常输出权重)
    ```

#### Swap optimizer
该特性将在模型前反向计算时将优化器卸载至host侧节省device内存，优化器更新时分片执行 “load -> update -> offload” 以降低优化器更新时的内存峰值。详细信息可参考[该文档](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/swap-optimizer.md)。
- 配置文件的 [optimizer] 中配置`swap_optimizer = true`使能该特性。
- 配置文件的 [optimizer] 中配置`swap_optimizer_times = 16`可设定分块swap的次数，更精细控制优化器更新时的内存峰值。

#### 自定义 Context Parallel
同时修改以下两个配置，可使用自定义的 Context Parallel 上下文环境，执行自定义的CP逻辑。
- 配置文件的 [parallelism] 中配置`enable_custom_context_parallel = true`使能自定义CP。
- 配置文件的 [parallelism] 中配置`custom_context_parallel_path`为自定义的CP上下文环境类的路径以真正使能自定义CP。例如：`custom_context_parallel_path = "torchtitan_npu.distributed.context_parallel.dsa_cp.AscendDSAContextParallelContext"`。