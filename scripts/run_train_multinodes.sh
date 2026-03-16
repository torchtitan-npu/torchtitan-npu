# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ps -ef |grep -i python |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i torchrun |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i ray |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i vllm |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}

# TODO change to your environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

# TODO change to your network interface
Network_Interface=enp23s0f3
export GLOO_SOCKET_IFNAME=${Network_Interface}
export HCCL_SOCKET_IFNAME=${Network_Interface}
export HCCL_IF_BASE_PORT=30000

export LOG_RANK=${LOG_RANK:-0}  # rank to show log
export PYTHONUNBUFFERED=1

# TODO change to your device ips
IPs=('192.168.xxx.xxx' '192.168.xxx.xxx')
# TODO change 192.168 to your local IP 
LOCAL_HOST=`ifconfig|grep "inet 192.168"| awk '{print $2}'`
# if you have not ifconfig , use the following command
# LOCAL_HOST=$(ip addr show ${Network_Interface} | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d'/' -f1)
echo $LOCAL_HOST
NPUS_PER_NODE=16
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6300
NNODES=${#IPs[@]}
NODE_RANK=""
for i in "${!IPs[@]}";
do
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
    then
        echo "Node Rank : ${i}"
        NODE_RANK=$i
        break
    fi
done
if [[ $NODE_RANK == "" ]];then
    echo "[Error] Variable \"NODE_RANK\" must be configured"
    exit 1
fi
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

set -ex


NGPU=${NGPU:-"16"}
RDZV_ID="dsv32_train_$(date +%Y%m%d)"
CONFIG_FILE=${CONFIG_FILE:-"torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_61layers_128die.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan_npu.entry"}
time=$(date +%Y%m%d%H%M)
logfile=dsv32_128die_${time}_node${NODE_RANK}_${LOCAL_HOST//./_}.log
mkdir -p logs


TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${NGPU} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@" 2>&1 | tee -a logs/${logfile}