#!/bin/bash

# Baselines
python train.py --conf KWT_configs/kwt1_baseline_mean_config.yaml
python train.py --conf KWT_configs/kwt2_baseline_mean_config.yaml
python train.py --conf KWT_configs/kwt3_baseline_mean_config.yaml

#python train.py --conf KWT_configs/kwt1_finetune_mean_config.yaml --ckpt runs/kwt1_data2vec/best_encoder.pth
#python train.py --conf KWT_configs/kwt2_finetune_mean_config.yaml --ckpt runs/kwt2_data2vec/best_encoder.pth
#python train.py --conf KWT_configs/kwt3_finetune_mean_config.yaml --ckpt runs/kwt3_data2vec/best_encoder.pth
