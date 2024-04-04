#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-30:00
#SBATCH --mem=60G

module load python/3.10.9-fasrc01
source activate vaccine_abc
python -u cpu_train_nn.py
source deactivate

