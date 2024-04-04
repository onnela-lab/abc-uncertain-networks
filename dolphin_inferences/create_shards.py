import sys
import pickle
import os
import numpy as np
import time
import epidemic_utils


"""
create_shards.py unifies single simulation threads of a mode <mode>.
The number of threads unified into a single shard is <num_files>.
These threads were generated via generate_parallel.sbatch, and as such,
have a job_id that we can use to capture.
create_shards.sbatch should also be run in parallel in order to create
multiple shards.
"""



job_id = int(sys.argv[1])
mode = str(sys.argv[2])
num_files = int(sys.argv[3])

all_samples = "undefined"

start_time = time.time()
root = "/n/base_directory/" + str(mode) + "/"
# First, find all the output files
print("Unifying at " + str(root))
num_found = 0
lower = int((job_id - 1) * num_files) + 1
upper = int(job_id * num_files) + 1

names = ["output", "theta", "network_params", "summaries"]
data_arrays = {i: "NULL" for i in names}

for i in range(lower, upper):
    directory_num = i
    file_missing = False
    file_paths = []
    for name in names:
        file_paths.append(root + name + "/" + str(mode) + "_" + name + "_data_" + str(directory_num) + ".npy")
    for path in file_paths:
        if os.path.isfile(path):
            continue
        else:
            file_missing = True
            print("File missing: " + str(path))
    if not file_missing:
        for j in range(len(names)):
            fh = open(file_paths[j], "rb")
            if data_arrays[names[j]] == "NULL":
                data_arrays[names[j]] = np.load(fh)
            else:
                new_array = np.load(fh)
                data_arrays[names[j]] = np.concatenate((data_arrays[names[j]], new_array), axis = 0)
            fh.close()
        num_found += 1
        print("Found sample " + str(i))


if num_found > 0:
    print("Shape of output: " + str(data_arrays["output"].shape))
    print("Shape of parameters: " + str(data_arrays["theta"].shape))
    print("Shape of network parameters: " + str(data_arrays["network_params"].shape))
    print("Converting type of output array")
    data_arrays["output"] = data_arrays["output"].astype(bool)
    # Finally, dump our new file out.
    print("Size of output: " + str(data_arrays["output"].size * data_arrays["output"].itemsize))
    unified_path = []
    for name in names:
        unified_path = root + str(mode) + "_unified_" + name + "_shard_" + str(job_id) + ".npy"
        unified_fh = open(unified_path, "wb")
        np.save(unified_fh, data_arrays[name])
        unified_fh.close()

    delete = False
    if delete:
        print("Samples gathered, deleting")
        for i in range(lower, upper):
            directory_num = i
            file_paths = []
            for name in names:
                file_paths.append(root + name + "/" + str(mode) + "_" + name + "_data_" + str(directory_num) + ".npy")
            for path in file_paths:
                if os.path.isfile(path):
                    os.remove(path) # Delte the file once we've processed it.
