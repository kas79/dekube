#!/bin/bash -x



[[ -z ${MAX_STEPS} ]] && export MAX_STEPS=22
[[ -z ${CUDA_VISIBLE_DEVICES} ]] && export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

mkdir -p /datasets
cp -r /mnt/s3fs/datasets/databricks-dolly-15k /datasets

export DEKUBE_DATASET_PATH=/datasets/databricks-dolly-15k
export DEKUBE_MODEL_PATH=/mnt/s3fs/models/LLAMA2/metaAi/Llama-2-7b-chat-hf


# Get accelerate default config location
idcfg="${HOME}/.cache/huggingface/accelerate/default_config.yaml"


# Get ip/hostname of the main process machine
env | grep TF_CONFIG 2>&1 > /dev/null
if [ $? -eq 1 ];then
  # if there is no TF_CONFIG variable it means that there is only one machine
  IRANK=0
  IPSHOST=127.0.0.1
  NUM_NODES=1
else
  cluster_len=$(echo ${TF_CONFIG} | jq '.cluster | length')
  if [ ${cluster_len} -eq 1 ];then
    # If there is only one type (PS or WORKER) - we can use index as is 
    IRANK=$(echo ${TF_CONFIG} | jq .task.index)
    IPSHOST=$(echo ${TF_CONFIG} | jq -r '.cluster | .[] | .[0]'  |  cut -f 1 -d : )
    NUM_NODES=$(echo ${TF_CONFIG} | jq '.cluster | .[] | length')
  elif [ ${cluster_len} -eq 2 ];then
    if [ $(echo ${TF_CONFIG} | jq -r .task.type) == "ps" ];then
      # For PS we'll use index as is
      IRANK=$(echo ${TF_CONFIG} | jq .task.index)
    else
      # For WORKER we'll use it's own index increased by quantity of PS
      psnum=$(echo ${TF_CONFIG} | jq '.cluster.ps | length')
      IRANK=$(( $(echo ${TF_CONFIG} | jq .task.index) + ${psnum} ))
    fi
    IPSHOST=$(echo ${TF_CONFIG} | jq -r '.cluster.ps[0]'  |  cut -f 1 -d : )
    NUM_NODES=$(( $(echo ${TF_CONFIG} | jq '.cluster.ps | length') + $(echo ${TF_CONFIG} | jq '.cluster.worker | length') ))
  else
    echo "Abnormal cluster length"
    echo "Exiting..."
    exit 1
  fi
fi


### debug
echo "TF_CONFIG = $TF_CONFIG"
echo "IRANK = $IRANK"
echo "IPSHOST = $IPSHOST"
echo "NUM_NODES = $NUM_NODES"
###


iRESULTwrk_=${iRESULTwrk}
[ $IRANK -eq 0 ] || iRESULTwrk=/mnt/wrkSlave
mkdir -p ${iRESULTwrk}

##################################################
envsubst <<EOF > ${idcfg}
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: ${IRANK}
main_process_ip: ${IPSHOST}
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: $NUM_NODES
num_processes: $NUM_NODES
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
##################################################


cp /mnt/s3fs/scripts/C-02/llama2ft-02.py ${iRESULTwrk}
cd ${iRESULTwrk}

accelerate launch llama2ft-02.py | tee ${iRESULTwrk}/llama2ft-02.py_$(hostname).log
[ $IRANK -eq 0 ] || cp ${iRESULTwrk}/llama2ft-02.py_$(hostname).log ${iRESULTwrk_}
