import sys
import pickle
import os
import numpy as np
import time
import epidemic_utils

"""
Gather samples generated from storing mode.
In this case, mode should be storing. num_strands is the number of jobs you ran.
Note that if you ran 1000 jobs, you may not actually observe that many strands,
as a stringent ABC threshold can lead to many jobs where NO epidemics are accepted.
"""

mode = str(sys.argv[1])
num_strands = int(sys.argv[2])
# Now, gather the samples
# Note that samples are stored as a dictionary.
# For each key in the dictionary, we just keep appending.

all_samples = "undefined"

start_time = time.time()
root = "/n/nase_directory/" + str(mode) + "/"
# First, find all the output files
theta_array = "NULL"
network_matrix_array = "NULL"
print("Unifying at " + str(root))
num_found = 0
names = ["networks", "theta"]
data_arrays = {i: "NULL" for i in names}
for i in range(num_strands):
    directory_num = i+1
    file_paths = []
    file_missing = False
    if os.path.isfile(root + "pp_networks/" +  str(mode) + "_network_data_" + str(directory_num) + ".npy") and os.path.isfile(root + "pp_theta/" + str(mode) + "_theta_data_" + str(directory_num) + ".npy"):
        file_missing = False
        print("Found files for strand " + str(directory_num))
    else:
        file_missing = True
        print("File missing: " + str(directory_num))
    if not file_missing:
        fh_network  = open(root + "pp_networks/" +  str(mode) + "_network_data_" + str(directory_num) + ".npy", "rb")
        if data_arrays["networks"] == "NULL":
            data_arrays["networks"] = np.array(np.load(fh_network, allow_pickle = True))
        else:
            new_array = np.array(np.load(fh_network, allow_pickle=True))
            print(new_array)
            data_arrays["networks"] = np.concatenate((data_arrays["networks"],new_array),axis=0)
            fh_network.close()

    if not file_missing:
        fh_theta  = open(root + "pp_theta/" +  str(mode) + "_theta_data_" + str(directory_num) + ".npy", "rb")
        if data_arrays["theta"] == "NULL":
            data_arrays["theta"] = np.load(fh_theta, allow_pickle = True)
        else:
            new_array = np.load(fh_theta, allow_pickle = True)
            data_arrays["theta"] = np.concatenate((data_arrays["theta"], new_array), axis = 0)
            fh_theta.close()
        num_found += 1
        print("Found " + str(directory_num))
print("Strands found: " + str(num_found))
if num_found > 0:
    data_arrays["networks"] = np.array(data_arrays["networks"])
    print(data_arrays["networks"])
    print("Shape of networks: " + str(data_arrays["networks"].shape))
    print("Shape of parameters: " + str(data_arrays["theta"].shape))
    print("Converting type of output array")

    unified_network_path = root + str(mode) + "_unified_pp_networks.npy"
    fh = open(unified_network_path, "wb")
    np.save(fh, data_arrays["networks"])
    fh.close()
    unified_theta_path = root + str(mode) + "_unified_pp_theta.npy"
    fh = open(unified_theta_path, "wb")
    np.save(fh, data_arrays["theta"])
    fh.close()

    delete = False
    if delete:
        print("Samples gathered, deleting")
        for i in range(num_strands):
            directory_num = i+1
            if os.path.isfile(root + "pp_networks/" +  str(mode) + "_network_data_" + str(directory_num) + ".npy") and os.path.isfile(root + "pp_theta/" + str(mode) + "_theta_data_" + str(directory_num) + ".npy"):
                network_file_path = root + "pp_networks/" +  str(mode) + "_network_data_" + str(directory_num) + ".npy"
                if os.path.isfile(network_file_path):
                    os.remove(network_file_path) # Delte the file once we've processed it.
                theta_file_path = root + "pp_theta/" + str(mode) + "_theta_data_" + str(directory_num) + ".npy"
                if os.path.isfile(theta_file_path):
                    os.remove(theta_file_path)
