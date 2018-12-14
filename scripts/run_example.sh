#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=${PYTHONPATH}:`pwd`
export NUM_GPUS=2
config_path='retinanet_R_50_FPN_1x'
model_dir='./log/ret_R_50_FPN_1x'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir}