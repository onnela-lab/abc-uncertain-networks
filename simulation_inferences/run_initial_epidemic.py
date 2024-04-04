# Code that generates the initial, "true" epidemic. Only need to run once, to generate the appropriate files.
# Also sets the parameters for network sampling, and creates an "observed" network data matrix (X) that is used for sampling.

import epidemic_utils
import networkx 
import os
import pickle
import numpy as np
import argparse
import sys
import network_reconstruction

# Generate a file to keep the data from the initial epidemic.
path_to_output = r"true_epidemic/"
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)


# "True" values of parameters. that we start with.
num_nodes = 100
mean_degree = 4 # 
network_type = "ER"

initial_infected_amount = 0.05 # Percentage of population to be initially infected.

beta = 0.15
gamma = 0.1

time_steps = 50 # Total time-steps of epidemic.
cadences = [7]*num_nodes # All nodes have weekly testing cadence.
adherences = [1.0]*num_nodes # Percentage of times each node actually adhereces to scheduled tests.
check_typical = True
prior_param = [2,8,2,8] 
prior_dist = "gamma" # Distribution family for our rate parameters beta and gamma.


# True values and priors for network reconstruction
true_rate_0 = 1
true_rate_1 = 8
rate_0_prior_param = [2,4] # This is the prior for the FALSE postivie rateq
rate_1_prior_param = [24,2] # This is the parameters of a beta distribution that governs TRUE positive (e.g. rate of seeing positives when edge truly exists).
rho_prior_param = [0.5,30]
network_reconstruction_file = open(path_to_output + "network_reconstruction_params.pkl", "wb")
pickle.dump({"rate_0_prior": rate_0_prior_param, "rate_1_prior": rate_1_prior_param, "rho_prior": rho_prior_param, "rate_1": true_rate_1, "rate_0": true_rate_0, "rho": mean_degree/num_nodes}, network_reconstruction_file)
network_reconstruction_file.close()

"""
	num_nodes and avg_deg are the number of nodes and the average degree of the generated network.
	network_type is a string that tells us what kind of network to generate: "ER" or "BA"
	initial_infected_amount is a value 0-1 that is the amount of population infected at time 0
	beta is the probability of transmission on contact
"""
		
if network_type == "ER":
	prob = mean_degree/num_nodes
	true_network = networkx.erdos_renyi_graph(num_nodes,prob)
elif network_type == "LN":
	true_network = network_reconstruction.create_log_normal_graph(num_nodes, mean_degree)
network_file = open(path_to_output + "true_network.pkl", "wb")
pickle.dump(true_network,network_file)
network_file.close()
print("Generated and stored true network, based on reconstruction")


# Next, observe the network with some error.
observed_network = network_reconstruction.observe_poisson(true_network, true_rate_0, true_rate_1)
observed_network_file = open(path_to_output + "observed_network.pkl", "wb")
pickle.dump(observed_network, observed_network_file)
observed_network_file.close()
print("Observed a network with missingness.")
print("Number of false negatives: " + str(observed_network["false_negatives"]))
print("Number of false positives: " + str(observed_network["false_positives"]))
print("Adjacency matrix: ")
print(observed_network["adj_matrix"])


# Pick the seed nodes for infection.	
initial_number = int(initial_infected_amount * len(list(true_network.nodes())))
i_list = list(np.random.choice(np.array(list(true_network.nodes())), initial_number, replace = False))
print("Seeded " + str(initial_number) + " infected individuals")
initial_file = open(path_to_output + "true_initial_list.pkl", "wb")
pickle.dump(i_list, initial_file)
initial_file.close()

# Select the times of testing
test_times = epidemic_utils.generate_test_times(true_network, cadences,adherences, i_list, time_steps)
test_times_file = open(path_to_output + "test_times.pkl", "wb")
pickle.dump(test_times, test_times_file)
test_times_file.close()
print(test_times)

# Next, generate and store the "true" epidemic process. Retry until it fits "typical" values
true_epidemic_output = epidemic_utils.simulate_SIR_gillespie(true_network, beta, gamma, i_list, test_times, time_steps)
if check_typical:
	# Avoid extreme epidemics that are far beyond what is typical for these parameters.
	true_epidemic_prevalences = true_epidemic_output["tot_i"]
	num_checks = 0
	max_checks = 100
	typical_vals = epidemic_utils.get_typical_values(true_network, beta, gamma, i_list, test_times, time_steps)
	while epidemic_utils.is_typical(true_epidemic_prevalences, typical_vals) == False:
		# If our current trial is not typical, rerun it.
		true_epidemic_output = epidemic_utils.simulate_SIR_gillespie(true_network, beta, gamma, i_list, test_times, time_steps)
		true_epidemic_prevalences = true_epidemic_output["tot_i"]	
		num_checks += 1
		if num_checks > max_checks:
			print("Failed to find typical epidemic after " + str(max_checks) + " iterations.")
			break	   
print("Took " + str(num_checks) + " trials to get a suitable run.")

total_infected = true_epidemic_output["tot_i"]
print("Total number of people infected by the end: " + str(total_infected) + " out of " + str(num_nodes))

total_recovered = true_epidemic_output["tot_r"]
print("Total number of people recovered: " + str(total_recovered) + " out of " + str(num_nodes))

total_positive = true_epidemic_output["positive_results"]
print("Total positive results: " + str(total_positive))

print("'True' Epidemic finished, storing results")
epidemic_file = open(path_to_output + "true_epidemic.pkl", "wb")
pickle.dump(true_epidemic_output, epidemic_file)
epidemic_file.close()
	
param_dic = {"num_nodes": num_nodes, "mean_degree": mean_degree, "network_type": network_type, 
			 "initial_infected_amount": initial_infected_amount,
			 "beta": beta, "gamma": gamma, "time_steps": time_steps, "prior_dist": prior_dist, "prior_params": prior_param}
param_file = open(path_to_output + "params.pkl", "wb")
pickle.dump(param_dic, param_file)
param_file.close()
	
