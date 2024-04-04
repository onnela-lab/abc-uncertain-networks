#!/bin/bash
#
#SBATCH -c 1
#SBATCH -J vaccine_abc
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH --mem 250G

module load python/3.10.9-fasrc01
source activate vaccine_abc
python -u unify_pp.py $1 $2
source deactivate
