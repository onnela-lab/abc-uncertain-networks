#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-01:00
#SBATCH --mem 100G


module load python/3.10.9-fasrc01
source activate vaccine_abc
python abc_framework.py $1
source deactivate

