Code for dolphin inferences.

Framework:
- This code is meant to be run using a SLURM job scheduler; due to the large number of simulations required for Approximate Bayesian Computation, it is necessary to parallelize these jobs.
SLURM code is included in the .sbatch shell scripts.
- This code uses a datafile called dolphin_data.pkl. This data was preprocessed using data obtained from: https://datadryad.org/stash/dataset/doi:10.5061/dryad.rbnzs7h76 . Additional data
was provided by the Mann Lab at Georgetown University.

MAIN WORKFLOW:

1) Initial preprocessing
clean_dolphin_data.py
  - Reads data on dolphins and stores it in dolphin_data.pkl.
run_initial_epidemic.py
  - Creates dictionaries for necessary parameters, such as priors on contagion parameters.

2) Simulation generation
simple_generate.py
  - Generates individual simulation jobs.
  - Run in parallel using: sbatch --array=1-<number_of_jobs> generate_parallel.sbatch <mode> <simulations_per_job>
  - For example, sbatch --array=1-1000 generate_parallel.sbatch training 5000 will generate 1000x5000=5000000 simulations for our training set.
  - Must run for a "training" set and a "validation" set.
  - Uses auxiliary file epidemic_utils.py for implementation of SIR epidemic.
create_shards.py
  - Shard the large numbers of simulations.
  - Run using sbatch --array=1-<number_of_shards> create_shards.sbatch <mode> <jobs_per_shard>
  - For above example, can create 10 shards by running --array=1-10 create_shards.sbatch training 100
unify_shards.py
  - Unify the shards into one numpy array.
  - Executed by using sbatch unify_shards.sbatch <mode> <num_shards>

3) MDN Training
cpu_train_nn.py
  - Execute with cpu_train_nn.sbatch
  - Uses auxiliary file nn_utils.py

4) ABC generation
abc_framework.py
  - Run using sbatch generate_abc.sbatch <threshold>
  - <threshold> is a percent value. To accept the closest 5% of samples, use generate_abc.sbatch 5

NETWORK DIAGNOSTICS

1) Evaluating Mixing
single_network_generation.py
  - Is run in parallel using generate_parallel_networks.sbatch
  - Generates a single chain of samples using HMC.
network_troubleshoot.py
  - Unifies these chains.

2) Discrepancy Calculations
diagnose_ordered_nbin.py
  - Following the work by Young, 2020, generate discrepancy measures that evaluate the goodness of fit for our network model.

POSTERIOR PREDICTIVE CHECKS

1) simple_generate.py
- Identical to the generation of the initial training/validation samples, but run in "storing" mode.
- This will store parameters and networks that lead to simulated epidemics that fulfill a precalculated ABC threshold.

2) unify_pp.py
- Unifies the posterior draws into one file.

2) pp_check.py
- Using the posterior predictive samples generated from the parameters and networks stored previously, simulate new epidemics.
- These epidemics are stored to be used for our posterior predictive diagnostics.







  
