
import os
import sys
import epidemic_utils
import pickle
from tqdm import tqdm
import argparse
import numpy as np
import time
import itertools as it
import stan
import networkx as nx
import torch as th
import copy
import scipy

"""
For network diagnostics, we can generate multiple chains of
HMC samples via STAN.

Use these to evaluate the mixing properties of the network.
job_id is the parallelized SLURM job ID, which is then used to recollect the generated files.
num_samples is the number of samples to draw
burn_in is number of iterations for burn_in
file_name is the name that we will store our files under.
"""


job_id = sys.argv[1]
num_samples = int(sys.argv[2])
burn_in = int(sys.argv[3])
file_name = str(sys.argv[4])

data_file = open("dolphin_data.pkl", "rb")
dolphin_data = pickle.load(data_file)
data_file.close()
print("Start date: " + str(dolphin_data["first_date"]))
print("Last date: " + str(dolphin_data["last_date"]))
print("Max entry: " + str(np.max(dolphin_data["X_dynamic"])))
observed_network_adj_matrix = dolphin_data["X_dynamic"]
num_nodes = observed_network_adj_matrix.shape[1]
times = observed_network_adj_matrix.shape[0]

def flatten(X):
    num_nodes = X.shape[1]
    times = X.shape[0]
    flattened_X = []
    for t in range(times):
        curr_vec = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                curr_vec.append(X[t,i,j])
        flattened_X.append(copy.deepcopy(curr_vec))
    return np.array(flattened_X)
flattened_X = flatten(observed_network_adj_matrix).astype(int)


def get_zero(flat_X):
    n_zero = 0
    for i in flat_X:
        if i == 0:
            n_zero += 1
    print("zero elements: " + str(n_zero))
    return n_zero

def get_non_zero_X(flat_X):
    ret_list = []
    for i in flat_X:
        if i >0:
            ret_list.append(i)
    return np.array(ret_list).astype(int)
def get_zero_per_time(X):
    ret_matrix = []
    for t in range(times):
        z_count = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if X[t,i,j] == 0:
                    z_count += 1
        ret_matrix.append(z_count)
    return np.array(ret_matrix).astype(int)

# Next, use STAN to sample the networks.
f = open("ordered_hetero_nbin.stan")
stan_code = f.read()
f.close()
# Build a model.
model = stan.build(stan_code, data = {"N": flattened_X.shape[1], "times": times, "X_flat": flattened_X, "N_zero":get_zero_per_time(observed_network_adj_matrix)}, random_seed = int(job_id))
start_time = time.time()
fit = model.sample(num_chains = 1, num_warmup = burn_in, num_samples = num_samples) # Sample enough times to cover both burn in and the number of samples we want.
ttm = (time.time() - start_time)/60
print("Minutes needed for network sampling: " + str(ttm))
samples = {}

output_dir = "chains/"
names = ["n_0", "n_1", "n_2", "n_3", "n_4", "p", "rho"]
for name in names:
    samples[name] = np.flipud(np.rot90(np.array(fit[name])))
    network_params_file = output_dir + "/" + str(file_name) + "_" + name + "_" + str(job_id) + ".npy"
    print(samples[name].shape)
    np.save(network_params_file, samples[name])
