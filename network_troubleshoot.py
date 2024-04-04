"""
Simple script for doing the MDN-compressed ABC, once the models have been trained
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
Unify the parallel HMC chains created via single_network_generation.py
Purpose is to create trace plots to find potential issues with mixing.
"""


dir_name = "hetero_ordered_nbin"
names = ["n_0", "n_1", "n_2", "n_3", "n_4", "p", "rho"]
chains_to_sample = 100
chains = {i: [] for i in names}
found_chains = 0
for i in range(chains_to_sample):
    for name in names:
        file_name = "chains/" + str(dir_name) + "_" + name + "_" + str(i+1) + ".npy"
        if not os.path.exists(file_name):
            continue
        file_params = np.load(file_name)
        chains[name].append(file_params)
        found_chains += 1
print("Number of chains found: " + str(found_chains/len(names)))
tot_chains = int(found_chains/len(names))

if not os.path.exists("chains/" + dir_name):
    os.makedirs("chains/" + dir_name)

for n in names:
    print("Shape of " + n + " is " + str(chains[n][0].shape))

def get_chain_threads(chains, name, time_slot, full_name, index = 0):
    fig,ax = plt.subplots()

    chain_length = chains[name][0].shape[-1] # Take the last dimension.
    colors = ["dodgerblue", "green", "red", "violet", "orange", "black", "pink", "gold", "brown", "indigo", "hotpink", "darkolivegreen", "deepskyblue","turquoise","slategrey","magenta"]
    two_d = False
    if len(chains[name][0].shape) == 2:
        two_d = True
        chain_length = chains[name][0].shape[0]
    curr_array = np.zeros((len(chains[name]),chain_length))
    for c in range(len(chains[name])):
        if two_d:
            chain_data = chains[name][c][:,time_slot]
        else:
            chain_data = chains[name][c][time_slot, index ,:]
        color = list(np.random.choice(range(256), size=3))
        plt.scatter(range(chain_length), chain_data, alpha = 0.4, s = 0.5, color = colors[int(c%len(colors))])
        curr_array[c,:] = chain_data
    plt.savefig("chains/" + dir_name + "/diagnostic_chains_" + str(dir_name) + "_" + str(full_name) + ".png")
    np.save("chains/" + dir_name + "/diagnostic_chains_" + str(dir_name) + "_" + str(full_name) + "_array.npy", curr_array)

def get_chain_threads_n(chains, name, index):
    fig, ax = plt.subplots()
    colors = ["dodgerblue", "green", "red", "violet", "orange", "black", "pink", "gold", "brown", "indigo", "hotpink", "darkolivegreen", "deepskyblue","turquoise","slategrey","magenta"]
    chain_length = chains[name][0].shape[0]
    curr_array = np.zeros((len(chains[name]), chain_length))
    for c in range(len(chains[name])):
        chain_data = chains[name][c][:,index]
        color = list(np.random.choice(range(256), size=3))
        plt.scatter(range(chain_length), chain_data, alpha = 0.4, s = 0.5, color = colors[int(c%len(colors))])
        curr_array[c,:] = chain_data
    plt.savefig("chains/" + dir_name + "/diagnostic_chains_" + str(dir_name) + "_" + str(name) + "_" + str(index) + ".png")
    np.save("chains/" + dir_name + "/diagnostic_chains_" + str(dir_name) + "_" + str(name) + "_" + str(index) + "_array.npy", curr_array)

get_chain_threads_n(chains, "n_0", 0)
get_chain_threads_n(chains, "n_0", 1)
get_chain_threads_n(chains, "n_1", 0)
get_chain_threads_n(chains, "n_1", 1)
get_chain_threads_n(chains, "n_2", 0)
get_chain_threads_n(chains, "n_2", 1)
get_chain_threads_n(chains, "n_3", 0)
get_chain_threads_n(chains, "n_3", 1)
get_chain_threads_n(chains, "n_4", 0)
get_chain_threads_n(chains, "n_4", 1)
get_chain_threads(chains, "p", 0, "p_t_0")
get_chain_threads(chains, "p", 1, "p_t_1")
get_chain_threads(chains, "p", 2, "p_t_2")
get_chain_threads(chains, "p", 3, "p_t_3")
get_chain_threads(chains, "p", 4, "p_t_4")
get_chain_threads(chains, "rho", 0, "rho_t_0")
get_chain_threads(chains, "rho", 1, "rho_t_1")
get_chain_threads(chains, "rho", 2, "rho_t_2")
get_chain_threads(chains, "rho", 3, "rho_t_3")
get_chain_threads(chains, "rho", 4, "rho_t_4")
