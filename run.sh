#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

config="${WORKING_DIR}/configs/base_config.yaml"
output="/scratch/t.tovi/models/coseg/"

 python ${WORKING_DIR}/train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 $opts
