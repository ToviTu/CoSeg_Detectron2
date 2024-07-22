#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

config="${WORKING_DIR}/configs/eval_config.yaml"
#output="/storage1/chenguangwang/Active/t.tovi/models/coseg/"
output="/scratch/t.tovi/models/coseg/"

 python ${WORKING_DIR}/train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 --eval-only \
 OUTPUT_DIR $output \
 $opts
