#!/bin/bash
#SBATCH --output="train_3_speech_vcb.log"
#SBATCH --job-name="SpeechCommands_vcb_3"
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00 # walltime
#SBATCH --nodelist=i256-a10-08

srun --gres=gpu:4 --ntasks-per-node=4 singularity exec --nv ~/pytorch-24.01 python train_speechcommands.py --config configs/small_ViT_train_config_speech_commands_vcb.cfg