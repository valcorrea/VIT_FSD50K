#!/bin/bash

#SBATCH --output="ssformer.log"
#SBATCH --job-name="ssformer"
#SBATCH --gres=gpu:3
#SBATCH --time=4:00:00 # walltime

srun --gres=gpu:3 singularity exec -nv ~/containers/pytorch-24.02 python light_train.py ~/repositories/avs8-850-deep-learning-mini-project/configs/ssformer.cfg --tr_manifest_path ~/datasets/covid19_cough/manifests/train_chunk.csv --val_manifest_path ~/datasets/covid19_cough/manifests/val_chunk.csv --labels_map ~/datasets/covid19_cough/manifests/lbl_map.json --id _test_lightning