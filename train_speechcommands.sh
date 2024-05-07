#!/bin/bash
#SBATCH --output="time_test.log"
#SBATCH --job-name="SpeechCommands_sweep"
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00 # walltime
#SBATCH --nodelist=i256-a10-07

srun --gres=gpu:1 --ntasks-per-node=1 singularity exec --nv ~/containers/pytorch-24.02 python time_tests.py --seq-len '100' --embed-len '1024' --batch-size '256'