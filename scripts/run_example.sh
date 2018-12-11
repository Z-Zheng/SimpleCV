#!/usr/bin/env bash


export NUM_GPUS=2
config_path=''
model_dir=''
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir}