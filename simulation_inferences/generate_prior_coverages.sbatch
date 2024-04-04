#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-20:00
#SBATCH --mem 100G


module load python/3.10.9-fasrc01
source activate vaccine_abc
python -u prior_calculate_coverages.py $1 $2
source deactivate

