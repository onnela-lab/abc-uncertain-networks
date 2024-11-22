# Code that generates the initial, "true" epidemic. Only need to run once, to generate the appropriate files.

import epidemic_utils
import networkx
import os
import pickle
import numpy as np
import argparse
import sys
import network_reconstruction

# Generate a file to keep the data from the initial epidemic.
# Defines the priors
path_to_output = r"true_epidemic/"
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)


data_file = open("dolphin_data.pkl", "rb")
dolphin_dict = pickle.load(data_file)
data_file.close()

num_nodes = len(dolphin_dict["node_keys"])

# Prior parameters.
prior_param = [0,1,0,0.01,0,0.005,0,0.001,0.2,5,10,160]
prior_dist = "uniform" # Distribution family for our rate parameters beta and gamma.
delta = 1



observed_network = {"adj_matrix": dolphin_dict["X"]}
observed_network_file = open(path_to_output + "observed_network.pkl", "wb")
pickle.dump(observed_network, observed_network_file)
observed_network_file.close()
print(observed_network["adj_matrix"])


# Pick the seed nodes for infection.
initial_file = open(path_to_output + "true_initial_list.pkl", "wb")
pickle.dump(dolphin_dict["i_list"], initial_file)
initial_file.close()

r_file = open(path_to_output + "true_recovered_list.pkl", "wb")
pickle.dump(dolphin_dict["r_list"], r_file)
r_file.close()
# Select the times of testing
test_times = dolphin_dict["test_times"]
test_times_file = open(path_to_output + "test_times.pkl", "wb")
pickle.dump(test_times, test_times_file)
test_times_file.close()
print(test_times)

# Next, generate and store the "true" epidemic process. Retry until it fits "typical" values
true_epidemic_output = epidemic_utils.dict_to_list(dolphin_dict["results"])
epidemic_file = open(path_to_output + "true_epidemic.pkl", "wb")
pickle.dump(true_epidemic_output, epidemic_file)
epidemic_file.close()


param_dic = {"num_nodes": num_nodes, "demo_info": dolphin_dict["demo_info"], "time_steps": dolphin_dict["time_steps"], "i_times": dolphin_dict["i_times"], "prior_dist": prior_dist, "prior_params": prior_param, "delta": delta, "candidates": dolphin_dict["candidates"], "network_transition_times": dolphin_dict["network_transition_times"]}
param_file = open(path_to_output + "params.pkl", "wb")
pickle.dump(param_dic, param_file)
param_file.close()
