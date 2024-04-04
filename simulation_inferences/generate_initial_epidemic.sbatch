#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-00:20
#SBATCH --mem 500M


module load python/3.10.9-fasrc01
source activate vaccine_abc
python run_initial_epidemic.py
source deactivate

