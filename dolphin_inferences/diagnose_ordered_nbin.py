

import os
import sys
import pickle
import argparse
import numpy as np
import time
import itertools as it
import stan
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import random
import copy

num_samples = int(sys.argv[1])

print("Generating " + str(num_samples) + " trials for network diagnostics")
print("Heterogeneous ordered p0-Fixed Negative Binomial Diagnostics")
print("----------------------------------")	
path_to_output = r"ordered_nbin_network_diagnostics/" # Where we'll store our data.
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)


data_file = open("../second_triple_aged_hetero_12_network_dolphin_data.pkl", "rb")
dolphin_data = pickle.load(data_file)
data_file.close()
print("Start date: " + str(dolphin_data["first_date"]))
print("Last date: " + str(dolphin_data["last_date"]))
print("Max entry: " + str(np.max(dolphin_data["X_dynamic"])))
observed_network_adj_matrix = dolphin_data["X_dynamic"]
num_nodes = observed_network_adj_matrix.shape[1]
times = observed_network_adj_matrix.shape[0]

# First, load in the prior information for network reconstruction
# Next, use STAN to sample the networks.
f = open("ordered_hetero_nbin.stan")
stan_code = f.read()
f.close()

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
# Flattened X is still times x (num_nodes^2)
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
# Build a model.
model = stan.build(stan_code, data = {"N": flattened_X.shape[1], "times": times, "X_flat": flattened_X, "N_zero": get_zero_per_time(observed_network_adj_matrix)})
start_time = time.time()
burn_in = 1000
fit = model.sample(num_chains = 1, num_warmup = burn_in, num_samples = num_samples) # Sample enough times to cover both burn in and the number of samples we want.
ttm = (time.time() - start_time)/60
print("Minutes needed for network sampling: " + str(ttm))



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

        log_nu_ij_0 = np.log(1-rho[t])
        log_nu_ij_1 = np.log(rho[t])
        z_ij_0 = log_mu_ij_0[t,:] + log_nu_ij_0
        z_ij_1 = log_mu_ij_1[t,:] + log_nu_ij_1
        Q[t,:] = 1/(1+np.exp(z_ij_0 - z_ij_1))
    return Q # This Q is flattened to 2D.


def matrix_from_flattened(Q_flat,times,num_nodes):
    Q = np.zeros((times,num_nodes, num_nodes))
    for t in range(times):
        counter = 0
        for i in range(num_nodes):
            Q[t,i,i] = 0
            for j in range(i+1, num_nodes):
                Q[t,i,j] = Q_flat[t,counter]
                counter += 1
                Q[t,j,i] = Q[t,i,j]
    return Q

def get_num_nonzero(adj_matrix):
    num_nodes = adj_matrix.shape[1]
    times = adj_matrix.shape[0]
    num_nonzero = 0
    for t in range(times):
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj_matrix[t,i,j] > 0:
                    num_nonzero += 1
    return num_nonzero

def get_avg_encounters(adj_matrix):
    times = adj_matrix.shape[0]
    num_nodes = adj_matrix.shape[1]
    summed = 0
    entries = 0
    for t in range(times):
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                summed += adj_matrix[t,i,j]
                entries += 1
    return summed/entries

def get_std(adj_matrix):
    times = adj_matrix.shape[0]
    num_nodes = adj_matrix.shape[1]
    all_encounters = []
    for t in range(times):
        for i in range(num_nodes):
            for j in range(i+1,num_nodes):
                all_encounters.append(adj_matrix[t,i,j])
    return np.std(all_encounters)

def get_max_encounters(adj_matrix):
    return np.max(np.array(adj_matrix))

def nb_mean(n,p):
    return n/p


# Finally, infer all the networks, based on the probabilities drawn from the model.
print("Num nodes: " + str(num_nodes))
metrics = {"nonzero": [], "avg": [], "std":[], "max": []}


def nbmean(n,p):
    return n/p # This is the STAN formulation.

def discrepancy(X,Q,n,p):
    times = Q.shape[0]
    num_nodes = Q.shape[1]
    # X_tilde is the average synthetic data when parameters are fixed (rates).
    # Since we've drawn Q (prob of seeing each edge), we can draw X_tilde by then saying the average edge of X is going to be
    # the elements of 1 in Q times the true positive rate, and then we have all the 0s in Q times the false positive rate.
    idx = np.triu_indices(num_nodes, k=1) # Only consider upper triangular indices.
    tot_sum = 0
    timewise_discrepancies = []
    for t in range(times):
        X_tilde_mat = Q[t,:,:] * nb_mean(n[t,1],p[t]) + (1-Q[t,:,:]) * nb_mean(n[t,0],1)
        idx2 = np.where(X[t][idx]>0)[0] # Only consider matrix coordinates if they're greater than zero.
        new_term = np.sum(X[t][idx][idx2] * np.log(X[t][idx][idx2]/X_tilde_mat[idx][idx2]))
        tot_sum += new_term
        timewise_discrepancies.append(new_term)
    return {"overall": tot_sum, "timewise": timewise_discrepancies}

p_val_samps = 1000
d_data = np.zeros(p_val_samps)
d_artificial = np.zeros(p_val_samps)
B = np.random.rand(num_nodes, num_nodes)
print("check 4")
A_sum = 0
t_d_data = np.zeros((p_val_samps, times))
t_d_artificial = np.zeros((p_val_samps, times))
degree_dist = np.zeros(shape = (p_val_samps, times, num_nodes))

draws_to_store = 25
summed_draws = np.zeros((times, num_nodes, num_nodes))
stored_draws = np.zeros((draws_to_store, times, num_nodes, num_nodes))
stored_A = np.zeros((draws_to_store, times, num_nodes, num_nodes))
print("Shape of n_0: " + str(fit["n_0"].shape)) 
print("Shape of p: " + str(fit["p"].shape)) 
print("Shape of rho: " + str(fit["rho"].shape)) 
for sample_id in range(p_val_samps):
    new_n = np.vstack((fit["n_0"][:,sample_id],fit["n_1"][:,sample_id],fit["n_2"][:,sample_id],fit["n_3"][:,sample_id],fit["n_4"][:,sample_id]))
    # Generate an artificial datasetflat_Q = generate_flattened_Q_nbin(fit["n"][:,sample_id], fit["p"][:,sample_id], fit["rho"][:,sample_id], flattened_X)
    flat_Q = generate_flattened_Q_nbin(new_n, fit["p"][:,sample_id], fit["rho"][:,sample_id], flattened_X)
    drawn_Q = matrix_from_flattened(flat_Q, times, num_nodes)
    A = np.random.rand(times,num_nodes,num_nodes) < drawn_Q # Grab a random adjacency matrix.
    X_tilde = np.zeros((times,num_nodes,num_nodes))
    for t in range(times):
        A[t][np.triu_indices(num_nodes, k=1)] = A[t].T[np.triu_indices(num_nodes, k=1)] 
        X_tilde_0 = scipy.stats.nbinom.rvs(new_n[t,0], 1/(1+1), size = drawn_Q[t,:,:].shape)
        X_tilde_1 = scipy.stats.nbinom.rvs(new_n[t,1], fit["p"][t,sample_id]/(fit["p"][t,sample_id] + 1), size = drawn_Q[t,:,:].shape)
        X_tilde[t,:,:] = (1-A[t,:,:]) * X_tilde_0 + A[t,:,:]*X_tilde_1 # An adjacency matrix created when we use sampled rates and Q to generate X.
    
    d_external = discrepancy(observed_network_adj_matrix, drawn_Q, new_n, fit["p"][:,sample_id])
    d_internal = discrepancy(X_tilde, drawn_Q, new_n, fit["p"][:,sample_id])

    d_data[sample_id] = d_external["overall"]
    d_artificial[sample_id] = d_internal["overall"]
    t_d_data[sample_id,:] = np.array(d_external["timewise"])
    t_d_artificial[sample_id,:] = np.array(d_internal["timewise"])


    metrics["nonzero"].append(get_num_nonzero(X_tilde))
    metrics["avg"].append(get_avg_encounters(X_tilde))
    metrics["std"].append(get_std(X_tilde))
    metrics["max"].append(get_max_encounters(X_tilde))

    for t in range(times):
        degree_dist[sample_id,t,:] = np.sum(X_tilde[t], axis = 1)

    if sample_id in list(range(draws_to_store)):
        stored_draws[sample_id, :, :, :] = X_tilde
        stored_A[sample_id, :, :, :] = A
    summed_draws = summed_draws + (A/p_val_samps)





print("A sum mean: " + str(A_sum/p_val_samps))
def pp_p_val(sampled_vals, obs_val, name):
    fig,ax = plt.subplots()
    plt.hist(sampled_vals, color = "grey", alpha = 0.2)
    plt.axvline(x=obs_val, color = "blue")
    p_val = sum(np.array(sampled_vals)>obs_val)/len(sampled_vals)
    print("For the metric " + str(name) + ", p-val was: " + str(p_val))
    plt.title(name + " p-value evalulation: p-val = " + str(p_val))
    print("The median sampled value was: " + str(np.mean(sampled_vals)) + " and observed value was: " + str(obs_val))
    plt.savefig(path_to_output + name + "_p_val.png")
pp_p_val(metrics["nonzero"], get_num_nonzero(observed_network_adj_matrix), "num_nonzero")
pp_p_val(metrics["avg"], get_avg_encounters(observed_network_adj_matrix), "average_val")
pp_p_val(metrics["std"], get_std(observed_network_adj_matrix), "std_val")
pp_p_val(metrics["max"], get_max_encounters(observed_network_adj_matrix), "max")

fh = open(path_to_output + "degree_dist.npy", "wb")
np.save(fh, degree_dist)
fh.close()

fh = open(path_to_output + "stored_draws.npy", "wb")
np.save(fh,stored_draws)
fh.close()

fh = open(path_to_output + "stored_summed_A.npy", "wb")
np.save(fh, summed_draws)
fh.close()


fig,ax = plt.subplots()
plt.scatter(d_data[d_data<d_artificial], d_artificial[d_data<d_artificial], s=15, edgecolor='#333333', linewidth=1, color='blue', alpha=1)
plt.scatter(d_data[d_data>d_artificial], d_artificial[d_data>d_artificial], s=15, edgecolor='#333333', linewidth=1, color='w', alpha=1)
min_val = np.min(np.array([np.min(d_data), np.min(d_artificial)]))
max_val = np.max(np.array([np.max(d_data), np.max(d_artificial)]))
plt.plot([min_val, max_val], [min_val, max_val], c='k', ls='--', lw=1)
plt.xlim(min_val,max_val)
plt.ylim(min_val,max_val)
plt.xlabel(r'$D(X;\theta)$')
plt.ylabel(r'$D(\tilde{X};\theta)$')
plt.savefig(path_to_output + "overall_discrepancy_comparison.png")
print("Discrepancy plots done.")
plt.text(30, 280, "P-value=" + str(float(len(d_artificial[d_data<d_artificial])) / float(p_val_samps)))
print("Finished. Overall p-value is : " + str(float(len(d_artificial[d_data<d_artificial])) / float(p_val_samps)))


fig,ax = plt.subplots()
plt.scatter(t_d_data[t_d_data<t_d_artificial], t_d_artificial[t_d_data<t_d_artificial], s=15, edgecolor='#333333', linewidth=1, color='green', alpha=1)
plt.scatter(t_d_data[t_d_data>t_d_artificial], t_d_artificial[t_d_data>t_d_artificial], s=15, edgecolor='#333333', linewidth=1, color='w', alpha=1)
min_val = np.min(np.array([np.min(t_d_data), np.min(t_d_artificial)]))
max_val = np.max(np.array([np.max(t_d_data), np.max(t_d_artificial)]))
plt.plot([min_val, max_val], [min_val, max_val], c='k', ls='--', lw=1)
plt.xlim(min_val,max_val)
plt.ylim(min_val,max_val)
plt.xlabel(r'$D(X;\theta)$')
plt.ylabel(r'$D(\tilde{X};\theta)$')
plt.savefig(path_to_output + "timewise_discrepancy_comparison.png")
print("Timewise Discrepancy plots done.")
plt.text(30, 280, "P-value=" + str(float(len(t_d_artificial[t_d_data<t_d_artificial])) / float(p_val_samps*times)))
print("Finished. Timewise p-value is : " + str(float(len(t_d_artificial[t_d_data<t_d_artificial])) / float(p_val_samps*times)))


fh = open(path_to_output + "/discrepancies.pkl", "wb")
pickle.dump({"t_d_data": t_d_data, "t_d_artificial": t_d_artificial, "d_data": d_data, "d_artificial": d_artificial}, fh)
fh.close()
