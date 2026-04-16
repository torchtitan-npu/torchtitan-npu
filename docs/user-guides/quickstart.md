# 快速上手

参考 [软件安装](./installation.md) 准备环境后，按照如下步骤操作，在 NPU 平台上运行 torchtitan-npu。

## 数据准备

1. 下载 Tokenizer [（以 DeepSeekV3.2 网络为例）](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main)。

新建 "deepseekv3.2-tokenizer" 目录，将 `tokenizer.json` 和 `tokenizer_config.json` 文件下载至该目录。

也可以通过以下方式下载 tokenizer：

```bash
# 从huggingface下载 DeepSeek V3.2 tokenizer https://huggingface.co/settings/tokens

python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.2 --assets tokenizer
```

2. 下载数据集 (以 [enwiki 数据集](https://huggingface.co/datasets/lsb/enwiki20230101) 为例)。

通过 huggingface 下载 [enwiki 的 parquet 数据](https://huggingface.co/datasets/lsb/enwiki20230101) 到 `./tests/assets`。
```bash
cd ./tests/assets
hf download lsb/enwiki20230101 --repo-type=dataset --local-dir .
cd ../..
```

## 配置 CANN 环境变量
```bash
source /usr/local/Ascend/cann/set_env.sh
```

## 启动训练任务

启动 torchtitan-npu 训练任务时，推荐使用以下脚本：单机环境使用 `scripts/run_train.sh`，多机环境使用 `scripts/run_train_multinodes.sh`。以下展示了一些常见任务的启动方式。

### 单机训练任务

默认配置，以 8 NPU 启动 DeepSeek-V3.2 debug 模型训练任务：
```bash
bash scripts/run_train.sh
```

自定义配置，以 16 NPU 启动 DeepSeek-V3.2 4 层模型训练任务：
```bash
NGPU=16 CONFIG_FILE=./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_4layers_debug.toml \
bash scripts/run_train.sh \
  --training.steps=100 \
  --training.global_batch_size=32
```

> [!NOTE]
> * `CONFIG_FILE`: 指定模型 TOML 配置文件路径，需在该文件中预先配置好相关特性的使能。
> * `NGPU`: 指定参与训练的 NPU 数量（默认为 8）。
> * `--training.steps` 与 `--training.global_batch_size`：动态覆盖 toml 配置中 `[training]` 部分的 `steps` 与 `global_batch_size`。

### 多机训练任务

在执行启动命令前，按照集群的实际情况编辑 `scripts/run_train_multinodes.sh` 文件中的网络与节点配置：

```toml
# TODO change to your network interface
Network_Interface=enp23s0f3 # 填入 ifconfig 的驱动名
...
# TODO change to your device ips
IPs=('192.168.xxx.xxx' '192.168.xxx.xxx') # 填入集群的所有IP
# TODO change 192.168 to your local IP
LOCAL_HOST=`ifconfig|grep "inet 192.168"| awk '{print $2}'` # 将 "192.168" 替换为当前 IP
```

在所有参与训练的节点上同时执行 `run_train_multinodes.sh`，以启动多机预训练任务。以 DeepSeek-V3.2 完整模型为例：
```bash
CONFIG_FILE=./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_61layers_4k_128die.toml \
bash scripts/run_train_multinodes.sh \
  --training.steps=500
```

> [!NOTE]
> * 脚本会自动通过 `LOCAL_HOST` 匹配 `IPs` 数组以推导当前机器的 `NODE_RANK`。若提取规则错误导致未匹配成功，脚本将报错退出。
> * 多机通信依赖相应的端口开放，请确保 `MASTER_PORT` (默认 6300) 以及 HCCL 通信基础端口 (默认 30000) 不被防火墙拦截。

### torchtitan 仓库内置训练任务

除了 torchtitan-npu 已经适配的模型外，还可以直接下载 torchtitan 代码，使用原生配置启动训练任务：

1. 拉取 torchtitan 代码。

```bash
cd ..
git clone -b v0.2.2 https://github.com/pytorch/torchtitan.git
```
2. 将 torchtitan 源代码移动至 torchtitan-npu 项目中。
```bash
cp ./torchtitan/torchtitan ./torchtitan-npu/ -r
```

3. 在 torchtitan-npu 项目中，使用 torchtitan 原生 toml 配置文件，启动训练。以 llama3 的 debug_model 配置为例:
```bash
cd torchtitan-npu
CONFIG_FILE="../torchtitan/models/llama3/train_configs/debug_model.toml" \
bash scripts/run_train.sh
```
