#!/usr/bin/env bash
set +e

# YS: not able to run locally; cpu training not supported

# train_data_dir=/Users/mac/Desktop/syt/Deep-Learning/Dataset/TaBERT_datasets/train_data/vanilla_tabert_sample
train_data_dir=/vault/TaBERT_datasets/train_data/vanilla_tabert

# run_dir=/Users/mac/Desktop/syt/Deep-Learning/repos/TaBERT/pretrain/runs/vanilla_tabert_sample
run_dir=/vault/TaBERT/pretrain/runs/vanilla_tabert
mkdir -p ${run_dir}

python train.py \
    --task vanilla \
    --data-dir ${train_data_dir} \
    --output-dir ${run_dir} \
    --table-bert-extra-config '{}' \
    --train-batch-size 8 \
    --gradient-accumulation-steps 32 \
    --learning-rate 2e-5 \
    --max-epoch 10 \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128