#!/bin/bash

#SBATCH --output="SGD_train_light_100_epochs.log"
#SBATCH --job-name="100_epoch_fsd50k_train"
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00 # walltime
#SBATCH --nodelist=i256-a10-09
#SBATCH --ntasks-per-node=4

echo "The config file used is small_ViT_train_config.cfg"
echo "Using 4 GPUs"
srun singularity exec --nv ~/pytorch=24.01 python train_light.py "/home/student.aau.dk/saales20/local-global-Mul-Head-Attention/VIT_FSD50K/configs/small_ViT_train_config.cfg"
