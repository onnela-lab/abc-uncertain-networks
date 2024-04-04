import numpy as np
import torch as th
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import epidemic_utils
import os
import argparse
import sys
import statistics
import scipy
import random
import copy
from scipy import stats
import network_reconstruction as nr

coverage_iterations = int(sys.argv[1])
percentile_accepted = float(sys.argv[2])


path_to_original_epidemic = r"true_epidemic/"

# Load in the observed data
obs_network_file = open(path_to_original_epidemic + "observed_network.pkl","rb")
obs_network = pickle.load(obs_network_file)
obs_network_file.close()
obs_data = obs_network["adj_matrix"]

# Load in beta, recovery coefficient, and time_steps from the original epidemic.
params_file = open(path_to_original_epidemic + "params.pkl", "rb")
params_dic = pickle.load(params_file)
true_beta = params_dic["beta"]
true_gamma = params_dic["gamma"]
time_steps = params_dic["time_steps"]
num_nodes = params_dic["num_nodes"]
prior_params = params_dic["prior_params"]
params_file.close()

test_times_file = open(path_to_original_epidemic + "test_times.pkl", "rb")
test_times = pickle.load(test_times_file)
test_times_file.close()

network_params_file = open(path_to_original_epidemic + "network_reconstruction_params.pkl", "rb")
network_params_dic = pickle.load(network_params_file)
true_rate_0 = network_params_dic["rate_0"]
true_rate_1 = network_params_dic["rate_1"]
true_rho = network_params_dic["rho"]
network_params_file.close() 

# Load in the initial list and test times
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()

# Load in the models
path_to_compressor = "models/compressor.pt"
compressor = th.load(path_to_compressor)
compressor.eval()

path_to_mdn = "models/mdn.pt"
mdn = th.load(path_to_mdn)
mdn.eval()

# Load in the ABC data
print("Drawing all samples from training set")
data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
training_samples = epidemic_utils.decompress_pickle(train_data_path)
num_training_samples = len(training_samples["output"])
theta = training_samples["theta"]
training_betas = theta[:,0]
training_gammas = theta[:,1]
rates = training_samples["rates"].swapaxes(0,1)
training_rates_0 = rates[0,:]
training_rates_1 = rates[1,:]
training_rho = training_samples["rho"]

training_features = compressor(th.Tensor(training_samples["output"]))

percentiles_of_interest = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
beta_coverages = {}
gamma_coverages = {}
rate_0_coverages = {}
rate_1_coverages = {}
rho_coverages = {}
for percentile in percentiles_of_interest:
    beta_coverages[percentile] = 0
    gamma_coverages[percentile] = 0
    rate_0_coverages[percentile] = 0
    rate_1_coverages[percentile] = 0
    rho_coverages[percentile] = 0
# Which parameter values will we test out?
beta_draws = np.random.gamma(shape = prior_params[0], scale = 1/prior_params[1], size = coverage_iterations)
gamma_draws = np.random.gamma(shape = prior_params[2], scale = 1/prior_params[3], size = coverage_iterations)
rate_0_draws = np.random.gamma(shape = network_params_dic["rate_0_prior"][0], scale = 1/network_params_dic["rate_0_prior"][1], size = coverage_iterations)
rate_1_draws = np.random.gamma(shape = network_params_dic["rate_1_prior"][0], scale = 1/network_params_dic["rate_1_prior"][1], size = coverage_iterations)
rho_draws = np.random.beta(network_params_dic["rho_prior"][0], network_params_dic["rho_prior"][1], size = coverage_iterations)

results = {"beta": {"draws":beta_draws, "means": [], "medians": [], "ranks": [], "percentiles": [], "bounds": [], "coverages": beta_coverages}, "gamma": {"draws": gamma_draws, "means": [], "medians": [], "ranks":[], "percentiles":[], "bounds": [], "coverages": gamma_coverages}, "rate_0": {"draws": rate_0_draws, "means": [], "medians": [], "ranks": [], "percentiles":[], "bounds": [], "coverages": rate_0_coverages}, "rate_1": {"draws":rate_1_draws, "means": [], "medians": [], "ranks": [], "percentiles": [], "bounds": [], "coverages": rate_1_coverages}, "rho": {"draws": rho_draws, "means": [], "medians": [], "ranks": [], "percentiles": [], "bounds":[], "coverages": rho_coverages}}
print("Drawing from prior and generating frequentist coverages.")
print(training_betas.shape)
print(training_gammas.shape)
print(training_rates_0.shape)
print(training_rates_1.shape)
print(training_rho.shape)
print(rates.shape)
for i in range(coverage_iterations):
	print("Iteration: " + str(i) + " out of " + str(coverage_iterations))
	generated_network = nr.generate_graph(obs_data, rate_0_draws[i], rate_1_draws[i], rho_draws[i])        
	samp = epidemic_utils.simulate_SIR_gillespie(generated_network, beta_draws[i], gamma_draws[i], i_list, test_times, time_steps)
	trial_features = compressor(th.Tensor(samp["results"]))
	dist = (training_features - trial_features).pow(2).sum(axis=1).sqrt().detach().numpy()
	euclidean_distances = list(dist)
	
	accepted_thetas = []
	accepted_rates_0 = []
	accepted_rates_1 = []
	accepted_rhos = []
	percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
	for j in range(num_training_samples):
		if euclidean_distances[j] <= percentile_value:
			accepted_thetas.append([training_betas[j],training_gammas[j]])
			accepted_rates_0.append(training_rates_0[j])
			accepted_rates_1.append(training_rates_1[j])
			accepted_rhos.append(training_rho[j])
	accepted_thetas = np.array(accepted_thetas)
	accepted_rates_0 = np.array(accepted_rates_0)
	accepted_rates_1 = np.array(accepted_rates_1)
	accepted_rhos = np.array(accepted_rhos)
	accepted_betas = accepted_thetas[:,0]
	accepted_gammas = accepted_thetas[:,1]
	results["beta"]["means"].append(np.mean(accepted_betas))
	results["beta"]["medians"].append(np.median(accepted_betas))
	results["gamma"]["means"].append(np.mean(accepted_gammas))
	results["gamma"]["medians"].append(np.median(accepted_gammas))
	results["rate_0"]["means"].append(np.mean(accepted_rates_0))
	results["rate_0"]["medians"].append(np.median(accepted_rates_0))
	results["rate_1"]["means"].append(np.mean(accepted_rates_1))
	results["rate_1"]["medians"].append(np.median(accepted_rates_1))
	results["rho"]["means"].append(np.mean(accepted_rhos))
	results["rho"]["medians"].append(np.median(accepted_rhos))
	for percentile in percentiles_of_interest:
		beta_lower_bound = np.percentile(accepted_betas, (100-percentile)/2)
		beta_upper_bound = np.percentile(accepted_betas, percentile + (100-percentile)/2)
		if beta_draws[i] < beta_upper_bound and beta_draws[i] > beta_lower_bound:
			results["beta"]["coverages"][percentile] += 1/coverage_iterations
		results["beta"]["bounds"].append([beta_lower_bound, beta_upper_bound])

		gamma_lower_bound = np.percentile(accepted_gammas, (100-percentile)/2)
		gamma_upper_bound = np.percentile(accepted_gammas, percentile + (100-percentile)/2)
		if gamma_draws[i] < gamma_upper_bound and gamma_draws[i] > gamma_lower_bound:
			results["gamma"]["coverages"][percentile] += 1/coverage_iterations
		results["gamma"]["bounds"].append([gamma_lower_bound, gamma_upper_bound])
	
		rate_0_lower_bound = np.percentile(accepted_rates_0, (100-percentile)/2)
		rate_0_upper_bound = np.percentile(accepted_rates_0, percentile + (100-percentile)/2)
		if rate_0_draws[i] < rate_0_upper_bound and rate_0_draws[i] > rate_0_lower_bound:
			results["rate_0"]["coverages"][percentile] += 1/coverage_iterations
		results["rate_0"]["bounds"].append([rate_0_lower_bound, rate_0_upper_bound])
		
		rate_1_lower_bound = np.percentile(accepted_rates_1, (100-percentile)/2)
		rate_1_upper_bound = np.percentile(accepted_rates_1, percentile + (100-percentile)/2)
		if rate_1_draws[i] < rate_1_upper_bound and rate_1_draws[i] > rate_1_lower_bound:
			results["rate_1"]["coverages"][percentile] += 1/coverage_iterations
		results["rate_1"]["bounds"].append([rate_1_lower_bound, rate_1_upper_bound])
		
		rho_lower_bound = np.percentile(accepted_rhos, (100-percentile)/2)
		rho_upper_bound = np.percentile(accepted_rhos, percentile + (100-percentile)/2)
		if rho_draws[i] < rho_upper_bound and rho_draws[i] > rho_lower_bound:
			results["rho"]["coverages"][percentile] += 1/coverage_iterations
		results["rho"]["bounds"].append([rho_lower_bound, rho_upper_bound])
	
	beta_rank = sum(accepted_betas < beta_draws[i])
	results["beta"]["ranks"].append(beta_rank)
	gamma_rank = sum(accepted_gammas < gamma_draws[i])
	results["gamma"]["ranks"].append(gamma_rank)
	results["rate_0"]["ranks"].append(sum(accepted_rates_0 < rate_0_draws[i]))
	results["rate_1"]["ranks"].append(sum(accepted_rates_1 < rate_1_draws[i]))
	results["rho"]["ranks"].append(sum(accepted_rhos < rho_draws[i]))
	
	beta_percentile = stats.percentileofscore(accepted_betas, beta_draws[i])
	results["beta"]["percentiles"].append(beta_percentile)
	gamma_percentile = stats.percentileofscore(accepted_gammas, gamma_draws[i])
	results["gamma"]["percentiles"].append(gamma_percentile)
	results["rate_0"]["percentiles"].append(stats.percentileofscore(accepted_rates_0, rate_0_draws[i]))
	results["rate_1"]["percentiles"].append(stats.percentileofscore(accepted_rates_1, rate_1_draws[i]))
	results["rho"]["percentiles"].append(stats.percentileofscore(accepted_rhos, rho_draws[i]))

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(results["beta"]["coverages"].values()))
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.title("Nominal vs Empirical Coverage, Beta")
plt.savefig("abc/prior_ci_coverage_beta.png")
print("95 percent coverage for beta for MDN-ABC is: " + str(results["beta"]["coverages"][95]))

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(results["gamma"]["coverages"].values()))
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.title("Nominal vs Empirical Coverage, Gamma")
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.savefig("abc/prior_ci_coverage_gamma.png")
print("95 percent coverage for gamma for MDN-ABC is: " + str(results["gamma"]["coverages"][95]))

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(results["rate_0"]["coverages"].values()))
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.title("Nominal vs Empirical Coverage, Rate_0")
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.savefig("abc/prior_ci_coverage_rate_0.png")
print("95 percent coverage for rate_0 for MDN-ABC is: " + str(results["rate_0"]["coverages"][95]))

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(results["rate_1"]["coverages"].values()))
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.title("Nominal vs Empirical Coverage, Rate_1")
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.savefig("abc/prior_ci_coverage_rate_1.png")
print("95 percent coverage for rate_1 for MDN-ABC is: " + str(results["rate_1"]["coverages"][95]))

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(results["rho"]["coverages"].values()))
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.title("Nominal vs Empirical Coverage, Rho")
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.savefig("abc/prior_ci_coverage_rho.png")
print("95 percent coverage for rho for MDN-ABC is: " + str(results["rho"]["coverages"][95]))

results_file = open("abc/prior_coverage_mean_medians.pkl", "wb")
pickle.dump(results, results_file)
results_file.close()


