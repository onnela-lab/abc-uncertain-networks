import sys
import pickle
import os
import numpy as np
import time
import epidemic_utils

mode = str(sys.argv[1])
num_dirs = int(sys.argv[2])
# Now, gather the samples
# Note that samples are stored as a dictionary. 
# For each key in the dictionary, we just keep appending.

all_samples = "undefined"

start_time = time.time()
root = "/n/sim_directory/" + str(mode) + "/"
# First, find all the output files
output_array = "NULL"
theta_array = "NULL"
network_params_array = "NULL"
print("Unifying at " + str(root))
num_found = 0
for i in range(num_dirs):
    directory_num = i+1
    output_file_path = root + str(mode) + "_unified_output_shard_" + str(directory_num) + ".npy"
    theta_file_path = root + str(mode) + "_unified_theta_shard_" + str(directory_num) + ".npy"
    network_params_file_path = root + str(mode) + "_unified_network_params_shard_" + str(directory_num) + ".npy"
    if os.path.isfile(output_file_path) and os.path.isfile(theta_file_path) and os.path.isfile(network_params_file_path): 
        output_fh = open(output_file_path, "rb")        
        theta_fh = open(theta_file_path, "rb")
        n_fh = open(network_params_file_path, "rb")        
        if output_array == "NULL" and theta_array == "NULL" and network_params_array == "NULL":
            output_array = np.load(output_fh)
            theta_array = np.load(theta_fh)
            network_params_array = np.load(n_fh)
        else:
            new_output_array = np.load(output_fh)
            new_theta_array = np.load(theta_fh)
            new_network_params_array = np.load(n_fh)
            output_array = np.concatenate((output_array, new_output_array), axis = 0)
            theta_array = np.concatenate((theta_array, new_theta_array), axis = 0)
            network_params_array = np.concatenate((network_params_array, new_network_params_array), axis = 0)
        output_fh.close()
        theta_fh.close()
        n_fh.close()
        num_found += 1
        print("Found " + str(directory_num))

if num_found > 0:
    print("Shape of output: " + str(output_array.shape))
    print("Shape of parameters: " + str(theta_array.shape))
    print("Shape of network parameters: " + str(network_params_array.shape))
    print("Converting type of output array")
    output_array = output_array.astype(bool)
    # Finally, dump our new file out.
    print("Size of output: " + str(output_array.size * output_array.itemsize))
    print("Size of thetas: " + str(theta_array.size * theta_array.itemsize))

    unified_output_path = root + str(mode) + "_unified_output.npy"
    unified_theta_path = root + str(mode) + "_unified_theta.npy"
    unified_network_params_path = root + str(mode) + "_unified_network_params.npy"
    unified_output_fh = open(unified_output_path, "wb")
    unified_theta_fh = open(unified_theta_path, "wb")
    unified_network_params_fh = open(unified_network_params_path, "wb")
    np.save(unified_output_fh, output_array)
    np.save(unified_theta_fh, theta_array)
    np.save(unified_network_params_fh, network_params_array)
    unified_output_fh.close()
    unified_theta_fh.close()
    unified_network_params_fh.close()

    delete = True
    if delete:
        print("Samples gathered, deleting")
        for i in range(num_dirs):
            directory_num = i+1
            output_file_path = root + str(mode) + "_unified_output_shard_" + str(directory_num) + ".npy"
            theta_file_path = root + str(mode) + "_unified_theta_shard_" + str(directory_num) + ".npy"
            network_params_file_path = root + str(mode) + "_unified_network_params_shard_" + str(directory_num) + ".npy"
            if os.path.isfile(output_file_path) and os.path.isfile(theta_file_path) and os.path.isfile(network_params_file_path): 
                os.remove(output_file_path) # Delte the file once we've processed it.
                os.remove(theta_file_path) # Delte the file once we've processed it.
                os.remove(network_params_file_path) # Delte the file once we've processed it.
 
