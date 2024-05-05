#!/bin/bash
#SBATCH --output="train_FNet_Speech_Commands_preNorm.log"
#SBATCH --job-name="SpeechCommands_FNet2"
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00 # walltime
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=nv-ai-03

srun singularity exec --nv ~/pytorch-24.03 python train_speechcommands.py --config configs/FNet_train_config_mdalal_preNorm.cfg --useFNet True