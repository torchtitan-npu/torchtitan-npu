# 快速上手

请先参考[软件安装](./installation.md)进行环境准备，环境准备后按照如下步骤操作，即可实现torchtitan-npu在昇腾设备上的高效运行，且无缝集成并充分发挥torchtitan-npu所提供的丰富加速与优化技术。


### 1. 数据准备

a. 下载Tokenizer[(以DeepSeekV3.2网络为例)](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main)。

新建“deepseekv3.2-tokenizer”目录，并将 `tokenizer.json` 和 `tokenizer_config.json` 文件下载至该目录。

也可以通过以下方式下载tokenizer：

```bash
# 从huggingface下载 DeepSeek V3.2 tokenizer https://huggingface.co/settings/tokens

python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.2 --assets tokenizer
```

b. 下载数据集(以[enwiki数据集](https://huggingface.co/datasets/lsb/enwiki20230101)为例)。

首先创建数据集路径。

```shell
mkdir -p ./tests/assets/enwiki
```

下载[enwiki 的parquet数据](https://huggingface.co/datasets/lsb/enwiki20230101)到`./tests/assets`路径下面。

可以使用下面的命令下载数据集。

```bash
cd ./tests/assets
hf download lsb/enwiki20230101 --repo-type=dataset --local-dir .
cd ../..
```

> [!NOTE]
> 用户需要自行设置代理，以便访问或下载数据集。


### 2. 配置环境变量

当前以 root 用户安装后的默认路径为例，请用户根据 `set_env.sh` 的实际路径执行如下命令。

```shell
source /usr/local/Ascend/cann/set_env.sh
```

### 3. 启动预训练

本项目提供统一的预训练脚本：单机环境使用 `scripts/run_train.sh`，多机环境使用 `scripts/run_train_multinodes.sh`。用户可以通过**修改环境变量来指定计算资源与模型配置**，并通过**命令行参数动态重载（Override）TOML 配置文件**的默认设定。

#### 3.1 单机预训练

**3.1.1 默认参数启动**

默认拉起单机 8 卡 DeepSeek 训练任务：

```shell
bash scripts/run_train.sh
```

**3.1.2 自定义参数启动**

以单机 8 卡拉起 DeepSeekV3.2 模型为例，动态覆盖配置中的训练步数与全局 Batch Size：

```shell
NGPU=8 CONFIG_FILE=./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_debug.toml \
bash scripts/run_train.sh \
  --training.steps=100 \
  --training.global_batch_size=32
```

> [!NOTE]
> * `CONFIG_FILE`: 指定模型 TOML 配置文件路径，需在该文件中预先配置好相关特性的使能。
> * `NGPU`: 指定参与训练的 NPU 数量（默认为 8）。


#### 3.2 多机预训练

**3.2.1 准备多机训练脚本**

在执行启动命令前，需按集群实际情况编辑 `scripts/run_train_multinodes.sh` 文件中的网络与节点配置：
> [!IMPORTANT]
> * **节点 IP 配置**：修改 `IPs` 数组，填入所有参与训练的机器 IP（各机器需处于同一网段）。
> * **网卡与本机 IP 配置**：修改 `Network_Interface` 为实际通信网卡名称；按需调整 `LOCAL_HOST` 的提取规则（如将 `ifconfig|grep "inet 192.168"` 修改为对应的机器网段地址）。
> * **加载真实权重（可选）**：若需加载预训练权重，请在所选 TOML 文件的 `[checkpoint]` 模块下将 `enable` 设为 `true`，并填写 `initial_load_path` 为真实权重路径。

**3.2.2 启动多机任务**

在**所有参与训练的节点**上同时执行以下命令拉起训练，此处以 DeepSeek-V3.2 完整模型为例：

```shell
CONFIG_FILE=./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_61layers_4k_128die.toml \
bash scripts/run_train_multinodes.sh \
  --training.steps=500
```

> [!NOTE]
> * 脚本会自动通过 `LOCAL_HOST` 匹配 `IPs` 数组以推导当前机器的 `NODE_RANK`。若提取规则错误导致未匹配成功，脚本将报错退出。
> * 多机通信依赖相应的端口开放，请确保 `MASTER_PORT` (默认 6300) 以及 HCCL 通信基础端口 (默认 30000) 不被防火墙拦截。

### 4. 原仓模型训练

除了 torchtitan-npu 已适配的模型外，用户还可以直接使用 torchtitan 原仓代码，利用其自带的 TOML 配置文件启动训练。

1. 在本项目**外**克隆 torchtitan 原仓，并在本项目根目录创建软链接指向其内层包目录：

   ```shell
   # 在 torchtitan-npu 的上级目录执行
   git clone -b v0.2.2 https://github.com/pytorch/torchtitan.git
   # 回到 torchtitan-npu 项目根目录
   cd torchtitan-npu
   ln -s ../torchtitan/torchtitan ./torchtitan
   ```

2. 根据需要训练的模型，在 `./torchtitan/` 目录中选择对应的 TOML 配置文件，并按实际情况修改其中的 `tokenizer_path` 及权重加载路径（`initial_load_path`）。

3. 在 torchtitan-npu 项目根目录执行以下命令拉起训练，以 LLaMA3 为例：

   ```shell
   CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml \
   bash scripts/run_train.sh
   ```
