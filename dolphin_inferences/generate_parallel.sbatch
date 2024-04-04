#!/bin/bash
#
#SBATCH -c 1
#SBATCH -J vaccine_abc
#SBATCH -p serial_requeue
#SBATCH -t 0-01:50
#SBATCH --mem 10G

export XDG_CACHE_HOME=/n/holyscratch01/onnela_lab/maw4494/
module load python/3.10.9-fasrc01
source activate vaccine_abc
python -u simple_generate.py ${SLURM_ARRAY_TASK_ID} $1 $2 -s /dev/null 2>&1
source deactivate
