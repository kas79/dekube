#!/bin/bash -x


mkdir -p ${iRESULTwrk}

[[ -z ${MAX_STEPS} ]] && export MAX_STEPS=5
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export NCCL_DEBUG=INFO


cp /mnt/dekube/scripts/A-03/fine-tune.py ${iRESULTwrk}
cp /mnt/dekube/scripts/A-03/ds_config.json ${iRESULTwrk}
cd ${iRESULTwrk}


hostfile=""
deepspeed --hostfile=$hostfile fine-tune.py  \
    --report_to "none" \
    --data_path "/mnt/dekube/datasets/databricks-dolly-15k-modified.json" \
    --model_name_or_path "/mnt/dekube/models/Baichuan/Baichuan2-7B-Base" \
    --output_dir "output" \
    --model_max_length 512 \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --use_lora True \
    | tee ${iRESULTwrk}/fine-tune.py_$(hostname).log
