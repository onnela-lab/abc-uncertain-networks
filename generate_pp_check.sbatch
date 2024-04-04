#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-05:00
#SBATCH --mem 10G


module load python/3.10.9-fasrc01
source activate vaccine_abc
python -u pp_check.py $1
source deactivate

