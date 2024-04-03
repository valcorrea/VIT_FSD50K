#!/bin/bash

# Source: https://stackoverflow.com/a/44168719
# Run using the following: bash trainLightFromConfig.sh <Num_GPUs> <Config_Name>

sbatch <<EOT
#!/bin/bash
#SBATCH --output="trainLight_${2}.log"
#SBATCH --job-name="CoughViT_${2}"
#SBATCH --gres=gpu:${1}
#SBATCH --time=16:00:00 # walltime
# #SBATCH --nodelist=a256-t4-03

echo "The config file used is KWT_configs/${2}.cfg"
echo "Using ${1} GPUs"
srun --gres=gpu:${1} singularity exec --nv ~/pytorch-24.01 python train_light.py "KWT_configs/${2}.cfg"
EOT
