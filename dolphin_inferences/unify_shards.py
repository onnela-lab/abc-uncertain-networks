import sys
import pickle
import os
import numpy as np
import time
import epidemic_utils


"""
unify_shards.py unifies the shards created by create_shards.sbatch.
This creates one big numpy file.
The <num_dirs> parameter is the number of shards you have of your mode <mode>.
"""
mode = str(sys.argv[1])
num_dirs = int(sys.argv[2])

all_samples = "undefined"

start_time = time.time()
root = "/n//base_directory/" + str(mode) + "/"
# First, find all the output files
output_array = "NULL"
theta_array = "NULL"
network_params_array = "NULL"
print("Unifying at " + str(root))
num_found = 0
names = ["output", "theta", "network_params", "summaries"]
data_arrays = {i: "NULL" for i in names}
for i in range(num_dirs):
    directory_num = i+1
    file_paths = []
    file_missing = False
    for name in names:
        file_paths.append(root + str(mode) + "_unified_" + str(name) + "_shard_" + str(directory_num) + ".npy")
    for path in file_paths:
        if os.path.isfile(path):
            continue
        else:
            file_missing = True
            print("File missing: " + str(path))
    if not file_missing:
        for j in range(len(names)):
            fh  = open(file_paths[j], "rb")
            if data_arrays[names[j]] == "NULL":
                data_arrays[names[j]] = np.load(fh)
            else:
                new_array = np.load(fh)
                data_arrays[names[j]] = np.concatenate((data_arrays[names[j]], new_array), axis = 0)
            fh.close()
        num_found += 1
        print("Found " + str(directory_num))

if num_found > 0:
    print("Shape of output: " + str(data_arrays["output"].shape))
    print("Shape of parameters: " + str(data_arrays["theta"].shape))
    print("Shape of network parameters: " + str(data_arrays["network_params"].shape))
    print("Converting type of output array")
    data_arrays["output"] = data_arrays["output"].astype(bool)
    # Finally, dump our new file out.
    print("Size of output: " + str(data_arrays["output"].size * data_arrays["output"].itemsize))

    for name in names:
        unified_path = root + str(mode) + "_unified_" + name + ".npy"
        fh = open(unified_path, "wb")
        np.save(fh, data_arrays[name])
        fh.close()
    delete = False
    if delete:
        print("Samples gathered, deleting")
        for i in range(num_dirs):
            directory_num = i+1
            for name in names:
                file_path = root + str(mode) + "_unified_" + name + "_shard_" + str(directory_num) + ".npy"
                if os.path.isfile(file_path):
                    os.remove(file_path) # Delte the file once we've processed it.
