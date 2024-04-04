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
# Generate the data in batches...




job_id = sys.argv[1]
mode = sys.argv[2]
num_samples = int(sys.argv[3])

if str(mode) == "storing":
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
    orig_features = compressor(th.Tensor(true_results["results"]))
else:
    store_networks = False

burn_in_period = 2000 # Number of samples to discard for burn-in.
# Note that STAN typically has 1000 burn in samples regardless.

batch_size = 100
	
print("Generating " + str(num_samples) + " trials for " + mode)
	
# Define our paths.
path_to_output = r"data/" # Where we'll store our data.
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
output_name = path_to_output + mode + "_data_" + str(job_id)
output_file = path_to_output + mode + "_data_" + str(job_id) + ".pkl"
path_to_original_epidemic = r"true_epidemic/" # Where we'll get our network and vaccinated list.
	

# Load in the original "true" network
true_network_file = open(path_to_original_epidemic + "true_network.pkl", "rb")
true_network = pickle.load(true_network_file)
true_network_file.close()

# Load in the observed network
obs_network_file = open(path_to_original_epidemic + "observed_network.pkl", "rb")
observed_network_data = pickle.load(obs_network_file)
observed_network_adj_matrix = observed_network_data["adj_matrix"]

test_times_file = open(path_to_original_epidemic + "test_times.pkl", "rb")
test_times = pickle.load(test_times_file)
test_times_file.close()
	
# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()
	
# Load in beta and time_steps from the original epidemic.
params_file = open(path_to_original_epidemic + "params.pkl", "rb")
params_dic = pickle.load(params_file)
time_steps = params_dic["time_steps"]
prior_params = params_dic["prior_params"]
prior_dist = params_dic["prior_dist"]

params_file.close()
samples = {}
times = []


# First, load in the prior information for network reconstruction
n_params_file = open(path_to_original_epidemic + "network_reconstruction_params.pkl", "rb")
n_params_dic = pickle.load(n_params_file)
n_params_file.close()
print(n_params_dic)
rate_0_prior = n_params_dic["rate_0_prior"]
rate_1_prior = n_params_dic["rate_1_prior"]
rho_prior = n_params_dic["rho_prior"]
# Next, use STAN to sample the networks.
f = open("poisson_data_ER_prior.stan")
stan_code = f.read()
f.close()

# Build a STAN model and sample proposal networks..
model = stan.build(stan_code, data = {"n": observed_network_adj_matrix.shape[0], "X": observed_network_adj_matrix, "rate_0_prior": rate_0_prior, "rate_1_prior": rate_1_prior, "rho_prior": rho_prior})
start_time = time.time()
fit = model.sample(num_chains = 1, num_samples = burn_in_period + num_samples) # Sample enough times to cover both burn in and the number of samples we want.
ttm = (time.time() - start_time)/60
print("Minutes needed for network sampling: " + str(ttm))

# Finally, infer all the networks, based on the probabilities drawn from the model.
sampled_network_matrices = []
usable_Q = fit["Q"][:,:,burn_in_period:] # Only take samples after burn-in.
num_nodes = fit["Q"].shape[0]
for k in range(usable_Q.shape[2]):
    curr_Q = usable_Q[:,:,k]
    adj_matrix = np.zeros(shape = (curr_Q.shape[0], curr_Q.shape[0]))
  # rate of seeing positives when the edge DOES exist.
    for (i,j) in it.combinations(range(num_nodes),2):
        if np.random.rand() < curr_Q[i,j]:
            adj_matrix[i,j] = 1
            adj_matrix[j,i] = 1
    sampled_network_matrices.append(adj_matrix)

def generate_network_from_matrix(adj_matrix):
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        G.add_node(i)
    for (i,j) in it.combinations(range(num_nodes),2):
        if adj_matrix[i,j] > 0:
            G.add_edge(i,j)
    return G

curr_samples = 0
found = False
while curr_samples < num_samples:
    # We generate args.batch_size per iteration
    # but on the last iteration we might not have that many samples left to generate.
    curr_batch_size = min(batch_size, num_samples - curr_samples)
    # Need to then sample some epidemics.
    sampled_networks = []
    for i in range(curr_samples, curr_samples + curr_batch_size):
        sampled_networks.append(generate_network_from_matrix(sampled_network_matrices[i]))
    sample = epidemic_utils.sample_nm(curr_batch_size, sampled_networks, i_list, test_times, time_steps, prior_params, prior_dist)		
    if store_networks: # If we're storing networks, first make sure our samples fall into the correct threshold
        sample_results = sample["output"]
        op_features = compressor(th.Tensor(sample_results))
        dist = (op_features - orig_features).pow(2).sum(axis = 1).sqrt().detach().numpy()
        euclidean_distances = list(dist)
        accepted = []
        for i in range(len(euclidean_distances)):
            if euclidean_distances[i] < threshold:
                accepted.append(i+curr_samples)
        if sum(accepted) == 0:
            continue
        else:
            found = True
            for key, value in sample.items():
                accepted_values = [sample[key][i-curr_samples] for i in accepted]
                samples.setdefault(key, []).append(accepted_values)
            # And also append the parts we need for network sampling parameters.
            accepted_rho = [fit["rho"][:,burn_in_period:][0,i] for i in accepted]        
            accepted_rates = [fit["rates"][:,burn_in_period:][:,i] for i in accepted]
            samples.setdefault("rho",[]).append(accepted_rho)
            samples.setdefault("rates",[]).append(accepted_rates)        
            accepted_networks = [sampled_network_matrices[i] for i in accepted]
            samples.setdefault("networks", []).append(accepted_networks)
    else: # If we're not storing networks, just grab everything (the default option)
        for key, value in sample.items(): # sample comes out with a 'theta' and an 'output' and a 'time'
            samples.setdefault(key, []).append(value)
    # This above operation basically makes it so samples has keys 'theta' and 'output', in a list of lists.

    curr_samples += curr_batch_size
# Bring everything into one big list: thetas will be one giant list, and outputs will be a list of lists of lists
# The output for a trial is a list of lists (each interior list is the newly infected at that timestep)
if not store_networks:
    samples = {key: np.concatenate(value, axis=0) for key, value in samples.items()}
    samples["rates"] = np.flipud(np.rot90(np.array(fit["rates"][:,burn_in_period:])))
    samples["rho"] = np.array(fit["rho"][:,burn_in_period:]).reshape((num_samples,1))
else:
    if found: 
        samples = {key: np.concatenate(value, axis=0) for key, value in samples.items()}
        print(samples["rates"])
        print(samples["networks"])
        print(samples["rho"])

output_dir = "/n/sim_directory/" + str(mode) + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for additional in ["/output/", "/theta/", "/network_params/"]:
    if not os.path.exists(output_dir + additional):
        os.makedirs(output_dir + additional)
output_file = open(output_dir + "/output/" + str(mode) + "_output_data_" + str(job_id) + ".npy","wb")
np.save(output_file, samples["output"])
output_file.close()
theta_file = open(output_dir + "/theta/" + str(mode) + "_theta_data_" + str(job_id) + ".npy","wb")
np.save(theta_file, samples["theta"])
theta_file.close()
network_params_file = open(output_dir + "/network_params/" + str(mode) + "_network_params_data_" + str(job_id) + ".npy","wb")
np.save(network_params_file, np.concatenate((samples["rates"], samples["rho"]), axis = 1))
network_params_file.close()

print("Data generation finished for " + mode)


