#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH --mem 100G


module load python/3.10.9-fasrc01
source activate vaccine_abc
python network_troubleshoot.py
source deactivate

