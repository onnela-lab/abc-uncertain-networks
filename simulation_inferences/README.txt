Code for simulation study.
Due to large volume of simulations required for MDN training and rejection ABC, it is recommended to run simulations via a parallelized framework.
Code here employs the SLURM job scheduler, and shell scripts reflect that framework.
ABC results are included, so plots can be directly generated.

MAIN WORKFLOW
1) run_initial_epidemic.py
  - Creates simulated "original" epidemic. All inferences treat this as the ground truth that generates the observed data.
2) simple_generate.py (parallel).
  - Generates a single strand of simulated epidemics. 
  - Should be run in parallel:
    sbatch --array=1-<num_parallel_jobs> generate_parallel.sbatch <mode> <num_simulations_per_job>
  - Generate datasets for both "training" and "validation" mode.
3) cpu_train_nn.py 
  - Trains an MDN
  - Auxiliary file nn_utils.py includes additional code.
4) abc_framework.py
  - Runs rejection ABC by using the summary statistics from the MDN.
5) create_plots.py
  - Creates plots seen in paper.

COVERAGE PROPERTY EVALUATION
1) prior_calculate_coverages.py
  - Sampling theta values from the prior and true network from the posterior distribution of A and Phi conditional on X, we can calculate coverage properties.

