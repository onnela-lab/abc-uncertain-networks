#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p sapphire
#SBATCH -t 0-02:30
#SBATCH --mem 600G


module load python/3.10.9-fasrc01
source activate vaccine_abc
python abc_framework.py $1
source deactivate

