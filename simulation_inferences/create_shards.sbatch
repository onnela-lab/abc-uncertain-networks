#!/bin/bash
#
#SBATCH -c 1
#SBATCH -J vaccine_abc
#SBATCH -p serial_requeue
#SBATCH -t 0-01:00
#SBATCH --mem 30G

export XDG_CACHE_HOME=/n/holyscratch01/onnela_lab/maw4494/
module load python/3.10.9-fasrc01
source activate vaccine_abc
python -u create_shards.py ${SLURM_ARRAY_TASK_ID} $1 $2
source deactivate
