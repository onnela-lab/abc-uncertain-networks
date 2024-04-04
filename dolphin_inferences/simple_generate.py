# Code that generates the simulated data we require.
# Reads in the vaccinated lists (for now, don't necessarily start the epidemic with the same seeds)
# For now, don't allow for command line arguments, but this will be implemented in the future.


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
This code is meant to be run through a SLURM scheduler.
This allows for ease of parallelization.
Each job_id is associated with a series of <num_samples> simulations,
and the jobs are then sharded and unified into a single file once
simulations are completed.
"""
job_id = sys.argv[1]
mode = sys.argv[2]
num_samples = int(sys.argv[3])

epidemic_file = open("true_epidemic/true_epidemic.pkl", "rb")
true_output_len = len(pickle.load(epidemic_file))
epidemic_file.close()

# base_directory is the location that all simulated data files will be stored.
base_directory = "/n/maw44494/base_directory/"
output_dir = base_directory + str(mode) + "/"

"""
By default, we do NOT store the drawn networks, due to storage concerns.
However, if we are evaluating the posterior predictive (or later wish to do inference
on the network), we can utilize the "storing" command.
In "storing" mode, ONLY samples of A,theta,phi that are under a previously-defined
discrepancy threshold are accepted and stored.
To run "storing" mode, a full run of MDN-ABC must have already been completed, such
that the MDN has been trained, and an ABC discrepancy threshold is defined.
"""
store_networks = False
if str(mode) == "storing":
    store_networks = True
if store_networks:
    # There's an additional option to allow for sampling and STORING the networks.
    # Note that if this option is activated, we assume that the ABC has already been run.
    threshold_file = open("abc/threshold.pkl", "rb")
    threshold = pickle.load(threshold_file)
    threshold_file.close()

    threshold = threshold
    store_networks = True
    print("Storing networks... Assuming ABC has ALREADY run")
    print("Loading compressor...")
    compressor = th.load("models/compressor.pt")
    compressor.eval()
    true_results_file = open("true_epidemic/true_epidemic.pkl", "rb")
    true_results = pickle.load(true_results_file)
    true_results_file.close()
    orig_features = compressor(th.Tensor(true_results))
else:
    # Unless we have already run NA-MDN-ABC once, we do NOT store networks.
    store_networks = False



batch_size = 20
print("Generating " + str(num_samples) + " trials for " + mode)
path_to_original_epidemic = "true_epidemic/"

"""
Load in the test times (times when individuals are infected.
"""
test_times_file = open(path_to_original_epidemic + "test_times.pkl", "rb")
test_times = pickle.load(test_times_file)
test_times_file.close()
# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()

r_file = open(path_to_original_epidemic + "true_recovered_list.pkl", "rb")
r_list = pickle.load(r_file)
r_file.close()
"""
Load in various parameters from original dictionary.
"""
params_file = open(path_to_original_epidemic + "params.pkl", "rb")
params_dic = pickle.load(params_file)
time_steps = params_dic["time_steps"]
prior_params = params_dic["prior_params"]
prior_dist = params_dic["prior_dist"]
delta = params_dic["delta"]
demo_info = params_dic["demo_info"]
candidates = params_dic["candidates"]
i_times = params_dic["i_times"]
params_file.close()
samples = {}


"""
Load in network data.
"""
data_file = open("dolphin_data.pkl", "rb")
dolphin_data = pickle.load(data_file)
data_file.close()
print("Start date: " + str(dolphin_data["first_date"]))
print("Last date: " + str(dolphin_data["last_date"]))
print("Max entry: " + str(np.max(dolphin_data["X_dynamic"])))
observed_network_adj_matrix = dolphin_data["X_dynamic"]
num_nodes = observed_network_adj_matrix.shape[1]
times = observed_network_adj_matrix.shape[0]

network_transition_times = dolphin_data["network_transition_times"]
"""
For preprocessing the network data.
"""

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



# Note that using opt_zinbin_data, we actually just directly call the priors in the script.
# Next, use STAN to sample the networks.
# ordered_hetero_nbin.stan is STAN code that models a temporal network with heterogeneous parameters for negative-binomial data model.
f = open("ordered_hetero_nbin.stan")
stan_code = f.read()
f.close()
# Build a model.
burn_in_period = 1500 # Number of samples to discard for burn-in.
model = stan.build(stan_code, data = {"N": flattened_X.shape[1], "times": times, "X_flat": flattened_X, "N_zero":get_zero_per_time(observed_network_adj_matrix)}, random_seed = int(job_id))
start_time = time.time()
fit = model.sample(num_chains = 1, num_warmup = burn_in_period, num_samples = num_samples) # Sample enough times to cover both burn in and the number of samples we want.
ttm = (time.time() - start_time)/60
print("Minutes needed for network sampling: " + str(ttm))


def generate_flattened_Q_nbin(n,p,rho,X_flat):
    # Generate a flatted Q-vector, which can be used to draw a single instance of A from the posterior A,phi|X
    # Q is a flat vector with an element that describes the probability that a corresponding entry in A is 1.
    # Do this for a single pairing of n, p, t
    log_mu_ij_0 = np.zeros(X_flat.shape)
    log_mu_ij_1 = np.zeros(X_flat.shape)

    Q = np.zeros((times,X_flat.shape[1]))
    for t in range(times):
        ns = copy.deepcopy(n[t,:])
        ps = copy.deepcopy(p[t])
        p_trans = [1/(1+1), ps/(ps+1)] # Transform from the STAN form to python.

        log_mu_ij_0[t,X_flat[t,:] < 1] = scipy.stats.nbinom.logpmf(0,ns[0],p_trans[0])
        log_mu_ij_1[t,X_flat[t,:] < 1] = scipy.stats.nbinom.logpmf(0,ns[1],p_trans[1])
        log_mu_ij_0[t,X_flat[t,:] >= 1] = scipy.stats.nbinom.logpmf(X_flat[t,:][X_flat[t,:]>=1],ns[0],p_trans[0])
        log_mu_ij_1[t,X_flat[t,:] >= 1] = scipy.stats.nbinom.logpmf(X_flat[t,:][X_flat[t,:]>=1],ns[1],p_trans[1])

        log_nu_ij_0 = np.log(1-rho[t])
        log_nu_ij_1 = np.log(rho[t])
        z_ij_0 = log_mu_ij_0[t,:] + log_nu_ij_0
        z_ij_1 = log_mu_ij_1[t,:] + log_nu_ij_1
        Q[t,:] = 1/(1+np.exp(z_ij_0 - z_ij_1))
    return Q # This Q is flattened to 2D.


def network_list_from_flattened(Q_flat, times,num_nodes):
    # Generate a flattened adjacency matrix
    network_list = []
    for t in range(times):
        G = nx.Graph()
        counter = 0
        for i in range(num_nodes):
            G.add_node(i)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                prob =  Q_flat[t,counter]
                counter += 1
                if np.random.rand() < prob:
                    G.add_edge(i,j)
        network_list.append(copy.deepcopy(G))
    return network_list


curr_samples = 0
found = False
accepted_thetas = []
accepted_networks = []
all_n_matrix = np.zeros((num_samples, times, 2))
while curr_samples < num_samples:
    # We generate args.batch_size per iteration
    curr_batch_size = min(batch_size, num_samples - curr_samples)
    # Need to then sample some epidemics.
    sampled_network_lists = []
    for i in range(curr_samples, curr_samples + curr_batch_size):
        new_n = np.vstack((fit["n_0"][:,i],fit["n_1"][:,i],fit["n_2"][:,i],fit["n_3"][:,i],fit["n_4"][:,i]))
        all_n_matrix[i,:,:] = copy.deepcopy(new_n)
        flattened_Q = generate_flattened_Q_nbin(new_n, fit["p"][:,i], fit["rho"][:,i], flattened_X)
        sampled_network_lists.append(network_list_from_flattened(flattened_Q, times, num_nodes))
    sample = epidemic_utils.sample_nm(curr_batch_size, true_output_len, sampled_network_lists, network_transition_times, i_list, i_times, r_list, test_times, time_steps, delta, prior_params, prior_dist, demo_info, num_nodes)
    for key, value in sample.items(): # sample comes out with a 'theta' and an 'output' and a 'time'
        samples.setdefault(key, []).append(value)

    curr_samples += curr_batch_size
    print(curr_samples)

    # If we are storing networks, evaluate the new num_samples
    # according to the predetermined ABC threshold.
    if store_networks:
        print("Filtering networks for storage.")
        training_features = compressor(th.as_tensor(sample["output"], dtype = th.float))
        distances = (training_features - orig_features).pow(2).sum(axis=1).sqrt().detach().numpy()
        print(len(distances))
        for i in range(len(distances)):
            if distances[i] <= threshold:
                accepted_thetas.append(sample["theta"][i])
                accepted_networks.append(sampled_network_lists[i])

# Bring everything into one big list: thetas will be one giant list, and outputs will be a list of lists of lists
# The output for a trial is a list of lists (each interior list is the newly infected at that timestep)
samples = {key: np.concatenate(value, axis=0) for key, value in samples.items()}
samples["n"] = all_n_matrix
samples["p"] = np.expand_dims(np.flipud(np.rot90(np.array(fit["p"]))),axis = 2)
samples["rho"] = np.expand_dims(np.flipud(np.rot90(np.array(fit["rho"]))),axis = 2)
print(samples["n"].shape)
print(samples["p"].shape)
print(samples["rho"].shape)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for additional in ["/output/", "/theta/", "/network_params/", "/summaries/"]:
    if not os.path.exists(output_dir + additional):
        os.makedirs(output_dir + additional)
output_file = open(output_dir + "/output/" + str(mode) + "_output_data_" + str(job_id) + ".npy","wb")
np.save(output_file, samples["output"])
print(samples["output"].shape)
output_file.close()
theta_file = open(output_dir + "/theta/" + str(mode) + "_theta_data_" + str(job_id) + ".npy","wb")
np.save(theta_file, samples["theta"])
print(samples["theta"].shape)
theta_file.close()
network_params_file = open(output_dir + "/network_params/" + str(mode) + "_network_params_data_" + str(job_id) + ".npy","wb")
np.save(network_params_file, np.concatenate((samples["n"], samples["p"], samples["rho"]), axis = 2))
network_params_file.close()
s_file = open(output_dir + "/summaries/" + str(mode) + "_summaries_data_" + str(job_id) + ".npy","wb")
np.save(s_file, samples["summaries"])
print(samples["summaries"].shape)
s_file.close()
print("Data generation finished for " + mode)

"""
If "storing" mode (we are getting parameters in preparation for posterior predictive
simulations), then store the networks and theta.
Note that we do not need to store network_params, since those parameters are
unnecessary if the network is known.
"""
if store_networks:
    print(accepted_thetas)
    if len(accepted_thetas) > 0:
        accepted_network_matrices = []
        for n in accepted_networks:
            n_l = []
            for t in range(times):
                matrix = nx.adjacency_matrix(n[t])
                n_l.append(matrix)
            accepted_network_matrices.append(n_l)
        accepted_network_matrices = np.array(accepted_network_matrices)
        for additional in ["/pp_networks/", "/pp_theta/"]:
            if not os.path.exists(output_dir + additional):
                os.makedirs(output_dir + additional)
        accepted_network_matrices = np.array(accepted_network_matrices)
        theta_file = open(output_dir + "/pp_theta/" + str(mode) + "_theta_data_" + str(job_id) + ".npy", "wb")
        np.save(theta_file, np.array(accepted_thetas))
        network_file = open(output_dir + "/pp_networks/" + str(mode) + "_network_data_" + str(job_id) + ".npy", "wb")
        np.save(network_file, accepted_network_matrices)
        network_file.close()
        theta_file.close()
        print("Number of acceptable samples found: " + str(len(accepted_networks)))
        print("Shape of accepted thetas: " + str(np.array(accepted_thetas).shape))
        print("Shape of accepted networks: " + str(accepted_network_matrices.shape))
    else:
        print("No acceptable samples found.")
