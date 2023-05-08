#!/usr/bin/env bash
# Required environment variables:
# batch_size (recommendation: 8 / 16)
# lr: learning rate (recommendation: 3e-5 / 5e-5)
# seed: random seed, default is 1234
# BERT_NAME: pre-trained text model name ( bert-*)
# max_seq: max sequence length
# sample_ratio: few-shot learning, default is 1.0
# save_path: model saved path

DATASET_CHOOSE="umgf" # umt umgf hvpnet
DATASET_NAME="twitter17"
BERT_NAME="bert-base-uncased"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --dataset_choose=${DATASET_CHOOSE} \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=24 \
        --batch_size=16 \
        --bert_lr=3e-5 \
        --crf_lr=2e-1 \
        --other_lr=2e-3 \
        --warmup_ratio=0.1 \
        --eval_begin_epoch=3 \
        --seed=2022 \
        --do_train \
        --ignore_idx=0 \
        --max_seq=84 \
        --save_path=your_ckpt_path \
        --dropout=0.3 \
        --final_dropout=0.5 \
        --negative_slope1=0.01 \
	--negative_slope2=0.01

