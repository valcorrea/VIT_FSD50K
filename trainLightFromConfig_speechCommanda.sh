#!/bin/bash
#SBATCH --output="train_FNet_Speech_Commands.log"
#SBATCH --job-name="SpeechCommands_FNet"
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00 # walltime
#SBATCH --ntasks-per-node=4
# #SBATCH --nodelist=a256-t4-02
# #SBATCH --exclude=nv-ai-03

srun singularity exec --nv ~/pytorch-24.03 python train_speechcommands.py --config configs/FNet_train_config_mdalal.cfg --useFNet True
# srun singularity exec --nv ~/pytorch-24.03 python train_speechcommands.py --config configs/large_KWT.cfg