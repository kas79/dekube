#!/bin/bash -x

mkdir -p ${iRESULTwrk}

[[ -z ${MAX_STEPS} ]] && export MAX_STEPS=22
[[ -z ${CUDA_VISIBLE_DEVICES} ]] && export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

mkdir -p /datasets
cp -r /mnt/s3fs/datasets/databricks-dolly-15k /datasets
export DEKUBE_DATASET_PATH=/datasets/databricks-dolly-15k
export DEKUBE_MODEL_PATH=/mnt/s3fs/models/LLAMA2/metaAi/Llama-2-7b-chat-hf

idcfg="${HOME}/.cache/huggingface/accelerate/default_config.yaml"


##################################################
envsubst <<EOF > ${idcfg}
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_process_ip: 127.0.0.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
##################################################


cp /mnt/s3fs/scripts/C-01/llama2ft-01.py ${iRESULTwrk}
cd ${iRESULTwrk}
sed -i '/sharded_ddp=.*/d' llama2ft-01.py

accelerate launch llama2ft-01.py | tee ${iRESULTwrk}/llama2ft-01.py_$(hostname).log
