#!/bin/bash
#SBATCH --output="train_FNet_Speech_Commands.log"
#SBATCH --job-name="SpeechCommands_FNet"
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00 # walltime
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=i256-a10-07

srun singularity exec --nv ~/pytorch-24.03 python train_speechcommands.py --config configs/FNet_train_config_mdalal.cfg --useFNet True