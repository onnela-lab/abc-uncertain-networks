import matplotlib.pyplot as plt
import pickle
import torch as th
import numpy as np
import epidemic_utils
from tqdm import tqdm
import scipy.stats as st
import seaborn as sns
import os
import time
import argparse
import sys
import statistics
import scipy
import copy
import random
import pandas
import networkx as nx

"""
This script draws posterior predictive epidemics for model evaluation.
Note that to run the posterior predictive check, you should have already run generate_parallel.sbatch
in "storing" mode, in order to store the posterior draws of the networks A.
If not, it is still possible to conduct an approximate posterior predictive check by forward-generating
instances of A from posterior draws of Phi. In this case, set the indicator full_pp to False.
"""

# How many repetitions to do.
repetitions = int(sys.argv[1])

# Load up our thetas and our network params

root = "/n/base_directory/storing/"

theta_file = open("abc/abc_thetas.npy", "rb")
abc_thetas = np.load(theta_file)
theta_file.close()

n_params_file = open("abc/abc_n_params.npy", "rb")
abc_n_params = np.load(n_params_file)
n_params_file.close()

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


params_file = open("true_epidemic/params.pkl", "rb")
params_dic = pickle.load(params_file)
num_nodes = params_dic["num_nodes"]
time_steps = params_dic["time_steps"]
prior_params = params_dic["prior_params"]
delta = params_dic["delta"]
demo_info = params_dic["demo_info"]
i_times = params_dic["i_times"]
candidates = params_dic["candidates"]

params_file.close()
network_reconstruction_params_file = open("true_epidemic/network_reconstruction_params.pkl", "rb")
network_params_dic = pickle.load(network_reconstruction_params_file)
network_reconstruction_params_file.close()
path_to_original_epidemic = "true_epidemic/"
# Load in original test times.
true_test_times_file = open(path_to_original_epidemic + "test_times.pkl", "rb")
true_test_times = pickle.load(true_test_times_file)
true_test_times_file.close()
results_file = open(path_to_original_epidemic + "true_epidemic.pkl", "rb")
true_results = pickle.load(results_file)
results_file.close()
# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()
r_file = open(path_to_original_epidemic + "true_recovered_list.pkl", "rb")
r_list = pickle.load(r_file)
r_file.close()

# Load in original network observation.
obs_network_file = open(path_to_original_epidemic + "observed_network.pkl", "rb")
observed_network_data = pickle.load(obs_network_file)
X = observed_network_data["adj_matrix"]

results_dict = {"results": [], "i_times": [], "r_times": [], "tot_i": [], "tot_r": [], "positive_results": [], "negative_results":[]}

def generate_flattened_Q_nbin(n,p,rho,X_flat):
    # Do this for a single pairing of n, p, t
    log_mu_ij_0 = np.zeros(X_flat.shape)
    log_mu_ij_1 = np.zeros(X_flat.shape)

    Q = np.zeros((times,X_flat.shape[1]))
    # Deal with label-switching.
    for t in range(times):
        ns = copy.deepcopy(n[t,:])
        ps = copy.deepcopy(p[t])
        p_trans = [1/(1+1), ps/(ps+1)] # Transform from the STAN form to python.

        log_mu_ij_0[t,X_flat[t,:] < 1] = scipy.stats.nbinom.logpmf(0,ns[0],p_trans[0])
        log_mu_ij_1[t,X_flat[t,:] < 1] = scipy.stats.nbinom.logpmf(0,ns[1],p_trans[1])

        log_mu_ij_0[t,X_flat[t,:] >= 1] = scipy.stats.nbinom.logpmf(X_flat[t,:][X_flat[t,:]>=1],ns[0],p_trans[0])
        log_mu_ij_1[t,X_flat[t,:] >= 1] = scipy.stats.nbinom.logpmf(X_flat[t,:][X_flat[t,:]>=1],ns[1],p_trans[1])

        log_nu_ij_0 = np.log(1-rho[t,:])
        log_nu_ij_1 = np.log(rho[t,:])
        z_ij_0 = log_mu_ij_0[t,:] + log_nu_ij_0
        z_ij_1 = log_mu_ij_1[t,:] + log_nu_ij_1
        Q[t,:] = 1/(1+np.exp(z_ij_0 - z_ij_1))
    return Q # This Q is flattened to 2D.

def network_list_from_flattened(Q_flat, times,num_nodes):
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


"""
This section is not a "true" posterior predictive check, since it forward-generates
the network A from Phi and X instead of directly using the posterior draws of A.
"""

for rep in range(repetitions):
    print("Repetition " + str(rep) + " out of " + str(repetitions))
    # Sample from the posterior.
    random_index = random.randint(0,abc_n_params.shape[0] - 1)
    sampled_calf_beta = abc_thetas[random_index, 0]
    sampled_juvenile_beta = abc_thetas[random_index, 1]
    sampled_adult_beta = abc_thetas[random_index, 2]
    sampled_e_rate = abc_thetas[random_index,3]
    sampled_r_shape = abc_thetas[random_index, 4]
    sampled_r_scale = abc_thetas[random_index, 5]
    ns = abc_n_params[random_index,:,0:2]
    ps = abc_n_params[random_index, :,2:3]
    rhos = abc_n_params[random_index,:, 3:4]
    # First, use our network parameters to get our network back.
    flat_Q = generate_flattened_Q_nbin(ns, ps, rhos, flattened_X)
    network_list = network_list_from_flattened(flat_Q, times, num_nodes)
    results = epidemic_utils.simulate_SIR(network_list, network_transition_times, sampled_calf_beta, sampled_juvenile_beta, sampled_adult_beta, sampled_e_rate, sampled_r_shape, sampled_r_scale, i_list, i_times, r_list, true_test_times, time_steps, delta, demo_info)
    for k in list(results_dict.keys()):
        results_dict[k].append(results[k])

# Dump the results.
output_file = open("abc/pp_check.pkl", "wb")
pickle.dump(results_dict, output_file)
output_file.close()

# Analyse the positive results.
pos_low = np.percentile(np.array(results_dict["positive_results"]), 2.5)
pos_high = np.percentile(np.array(results_dict["positive_results"]), 97.5)
pos_mean = np.mean(np.array(results_dict["positive_results"]))
true_pos = np.sum(true_results)
print("In pp check, positive results had mean " + str(pos_mean) + ", median of "+ str(np.median(results_dict["positive_results"])) + " and 95% CI at " + str(pos_low) + ", " + str(pos_high))
print("In true epidemic, had " + str(true_pos))

fig, ax = plt.subplots()
plt.hist(results_dict["positive_results"], density = True)
plt.axvline(true_pos)
plt.savefig("abc/pp_check_positive_results.pdf")

"""
This section is the "true" posterior predictive check.
"""
full_pp = True
if full_pp:
    fh1 = open(root + "storing_unified_pp_networks.npy", "rb")
    pp_networks = np.load(fh1, allow_pickle=True)
    fh1.close()

    fh2 = open(root + "storing_unified_pp_theta.npy", "rb")
    pp_theta = np.load(fh2, allow_pickle=True)
    fh2.close()
    full_results_dict = {"results": [], "i_times": [], "r_times": [], "tot_i": [], "tot_r": [], "positive_results": [], "negative_results":[]}
    # Now, load in the full indices.
    for rep in range(repetitions):
        print("Full repetition " + str(rep) + " out of " + str(repetitions))
        random_index = random.randint(0,pp_theta.shape[0] - 1)
        sampled_network_matrices = pp_networks[random_index,:]
        sampled_network_list = []
        for t in range(times):
            sampled_network = nx.from_numpy_matrix(sampled_network_matrices[t])
            sampled_network_list.append(sampled_network)
        sampled_thetas = pp_theta[random_index,:]
        results = epidemic_utils.simulate_SIR(sampled_network_list, network_transition_times, sampled_thetas[0], sampled_thetas[1], sampled_thetas[2], sampled_thetas[3], sampled_thetas[4], sampled_thetas[5], i_list, i_times, r_list, true_test_times, time_steps, delta, demo_info)
        for k in list(full_results_dict.keys()):
            full_results_dict[k].append(results[k])
    output_file = open("abc/full_pp_check.pkl", "wb")
    pickle.dump(full_results_dict, output_file)
    output_file.close()

    pos_low = np.percentile(np.array(full_results_dict["positive_results"]), 2.5)
    pos_high = np.percentile(np.array(full_results_dict["positive_results"]), 97.5)
    pos_mean = np.mean(np.array(full_results_dict["positive_results"]))
    true_pos = np.sum(true_results)
    print("In pp check, positive results had mean " + str(pos_mean) + ", median of "+ str(np.median(full_results_dict["positive_results"])) + " and 95% CI at " + str(pos_low) + ", " + str(pos_high))
    print("In true epidemic, had " + str(true_pos))

    fig, ax = plt.subplots()
    plt.hist(full_results_dict["positive_results"], density = True)
    plt.axvline(true_pos)
    plt.savefig("abc/full_pp_check_positive_results.pdf")
