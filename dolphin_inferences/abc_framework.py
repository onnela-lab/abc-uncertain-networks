"""
Simple script for doing the NA-MDN-compressed ABC, once the models have been trained.
Generates data-files for the accepted values of theta and phi.
"""

import matplotlib.pyplot as plt
import pickle
import epidemic_utils
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

"""
We can run a "check" mode to use the "temporary" neural network files
for ABC (used if training fails unexpectedly but we want to check results).
"""
temp = False
if len(sys.argv) > 2:
    if str(sys.argv[2]) == "check":
        temp = True
        print("Using TEMP neural networks for ABC.")    
    else:
        temp = False

# How much of our training set to accept as ABC samples.
percentile_accepted = float(sys.argv[1])
print("Accepting " + str(percentile_accepted) + " percent.")

"""
Load the compressor
"""

path_to_compressor = "models/compressor.pt"
if temp:
    path_to_compressor = "models/compressor_temp.pt"
compressor = th.load(path_to_compressor)
compressor.eval()

path_to_mdn = "models/mdn.pt"
if temp: 
    path_to_mdn = "models/mdn_temp.pt"
mdn = th.load(path_to_mdn)
mdn.eval()	

"""
Load our original epidemic information
"""
params_file = open("true_epidemic/params.pkl", "rb")
params_dic = pickle.load(params_file)
num_nodes = params_dic["num_nodes"]
time_steps = params_dic["time_steps"]
prior_params = params_dic["prior_params"]
params_file.close()

	
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
	
# Lastly, let's get the original features, from our compressor.
orig_features = compressor(th.Tensor(np.array([true_results])))
orig_mdn = mdn(th.Tensor(np.array([true_results])))
		

"""
Now, conduct a basic rejection ABC algirithm.
First, draw a bunch of samples, do the simulation, and record the euclidean distances of features from original run.
"""
	
sampled_betas = []
sampled_gammas = []
sampled_times = []
euclidean_distances = []

accepted_times = [] 
	
print("Beginning rejection ABC.")	

base_directory = "/n/base_directory/"
location = base_directory + "training/training_unified_output.npy"
fh = open(location, "rb")
training_features = compressor(th.as_tensor(np.load(fh), dtype = th.float))
distances = (training_features - orig_features).pow(2).sum(axis=1).sqrt().detach().numpy()
fh.close()

params_location = base_directory + "training/training_unified_theta.npy"
fh = open(params_location, "rb")
sampled_parameters = np.load(fh)
fh.close()


network_params_location = base_directory + "training/training_unified_network_params.npy"
fh = open(network_params_location, "rb")
sampled_network_parameters = np.load(fh)
fh.close()
print("Loaded in values")
"""
Now, extract the best % of the runs and plot ABC results.
"""
if not os.path.exists("abc"):
	os.makedirs("abc")

# percentile_accepted came in as an argument.
accepted_thetas = []
# Grab the first percentile
percentile_value = np.percentile(distances, float(percentile_accepted))
print("Cutoff: " + str(percentile_value))
tf = open("abc/threshold.pkl", "wb")
pickle.dump(percentile_value, tf)
tf.close()



accepted_network_params = []
print(distances)
print("Shape of distance: " + str(distances.shape))
accepted_thetas = sampled_parameters[np.array(distances) <= percentile_value,:]
accepted_network_params = sampled_network_parameters[np.array(distances) <= percentile_value, :]
print("accepted_thetas shape: " + str(accepted_thetas.shape))
print("Distances computed and accepted values drawn.")
print(accepted_thetas)

"""
Here, we can generate some basic diagnostic plots, though 
these are not the ones created for the paper.
"""
def generate_analysis(accepted_vector, name):
    fig, ax = plt.subplots()
    var_kde = sns.kdeplot(list(accepted_vector))
    x,y = var_kde.get_lines()[0].get_data()
    var_cdf = scipy.integrate.cumtrapz(y,x,initial = 0)
    var_median = x[np.abs(var_cdf - 0.5).argmin()]
    plt.vlines(var_median,0,y[np.abs(var_cdf-0.5).argmin()], color = "tab:cyan")
    plt.title("MDN-Compressed ABC,\n " + str(name))
    plt.ylabel("Posterior Density")
    plt.xlabel(name)
    plt.savefig('abc/' + str(name) + '.png')
    fig,ax = plt.subplots()
    plt.hist(accepted_vector)
    plt.title(str(name))
    plt.savefig('abc/' + str(name) + '_hist.png')
    
    var_095 = np.percentile(accepted_vector, 97.5) # Upper
    var_05 = np.percentile(accepted_vector, 2.5) # Lower
    var_median_calculated = statistics.median(accepted_vector)
    print("Median for accepted " + str(name) + " is " + str(var_median_calculated) + ". 95% CI is " + str(var_05) + ", " + str(var_095))

generate_analysis(accepted_thetas[:,0], "beta_calf")
generate_analysis(accepted_thetas[:,1], "beta_juvenile")
generate_analysis(accepted_thetas[:,2], "beta_adult")
generate_analysis(accepted_thetas[:,3], "epsilon")
generate_analysis(accepted_thetas[:,4], "r_shape")
generate_analysis(accepted_thetas[:,5], "r_scale")

path_to_output = r"abc/" # Where we'll store our data.
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
output_file = open(path_to_output + "abc_thetas.npy", "wb")
np.save(output_file, accepted_thetas)
output_file.close()

distance_file = open(path_to_output + "abc_distances.npy", "wb")
np.save(distance_file, np.array(distances))
distance_file.close()

i_period_means = []
# Now draw the recovery times.
for i in range(accepted_thetas.shape[0]):
    r_shape = accepted_thetas[i,4]
    r_scale = accepted_thetas[i,5]
    i_period_means.append(r_scale * scipy.special.gamma(1 + 1/r_shape))
generate_analysis(i_period_means, "mean_infectious_period")


print("Doing network parameters.")
print("-------------------------------------------")
generate_analysis(accepted_network_params[:,0,0], "n_0_t_0")
generate_analysis(accepted_network_params[:,0,1], "n_1_t_0")
generate_analysis(accepted_network_params[:,1,0], "n_0_t_1")
generate_analysis(accepted_network_params[:,1,1], "n_1_t_1")
generate_analysis(accepted_network_params[:,2,0], "n_0_t_2")
generate_analysis(accepted_network_params[:,2,1], "n_1_t_2")
generate_analysis(accepted_network_params[:,3,0], "n_0_t_3")
generate_analysis(accepted_network_params[:,3,1], "n_1_t_3")
generate_analysis(accepted_network_params[:,4,0], "n_0_t_4")
generate_analysis(accepted_network_params[:,4,1], "n_1_t_4")
generate_analysis(accepted_network_params[:,0,2], "p_t_0")
generate_analysis(accepted_network_params[:,1,2], "p_t_1")
generate_analysis(accepted_network_params[:,2,2], "p_t_2")
generate_analysis(accepted_network_params[:,3,2], "p_t_3")
generate_analysis(accepted_network_params[:,4,2], "p_t_4")
generate_analysis(accepted_network_params[:,0,3], "rho_t_0")
generate_analysis(accepted_network_params[:,1,3], "rho_t_1")
generate_analysis(accepted_network_params[:,2,3], "rho_t_2")
generate_analysis(accepted_network_params[:,3,3], "rho_t_3")
generate_analysis(accepted_network_params[:,4,3], "rho_t_4")
n_params_file = open(path_to_output + "abc_n_params.npy", "wb")
np.save(n_params_file, accepted_network_params)
n_params_file.close()


