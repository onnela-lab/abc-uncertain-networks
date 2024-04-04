"""
Script for training the neural nets for vaccine inference
"""

import torch as th
import epidemic_utils
import pickle
import nn_utils
import os
import argparse
import numpy as np
import time
start_time = time.time()
# Read the number of nodes from our original epidemic parameter dictionary.
device = "cpu"
print("Using device: " + str(device))

data_path = "/n/base_directory/"

params_file = open("true_epidemic/params.pkl", "rb")
params_dic = pickle.load(params_file)
num_nodes = params_dic["num_nodes"]
params_file.close()

single_trial_file = open(data_path + "/training/output/training_output_data_1.npy", "rb")
single_trial = np.load(single_trial_file)
single_trial_file.close()
len_output = single_trial.shape[1] # When doing time inference, there's actually two elements to the output (the output and the times)
print("Length of output is " + str(len_output))

num_components = 5
batch_size = 512
num_features = 20
num_param = 5
compressor_layers = [len_output, 3000, 1000, 500, 200, 100, 50, num_features]
mdn_layers = [num_features, 20, 20, num_components]
learning_rate = 5*10e-5
patience = 10
max_epochs = 40

print("Training Neural Net")
print("Batch size: " + str(batch_size))
print("Number of features: " + str(num_features))
print("Number of beta components for joint posterior: " + str(num_components))
print("Number of parameters of inference: " + str(num_param))
print("Compressor layers: " + str(compressor_layers))
print("MDN layers: " + str(mdn_layers))
print("Learning rate: " + str(learning_rate))
print("Patience: " + str(patience))

"""
Loading Data
"""
datasets = {}
train_output_path = data_path + "training/training_unified_output.npy"
train_theta_path = data_path + "training/training_unified_theta.npy"
validation_output_path = data_path + "validation/validation_unified_output.npy"
validation_theta_path = data_path + "validation/validation_unified_theta.npy"

output_fh = open(train_output_path, "rb")
training_data = th.as_tensor(np.load(output_fh), dtype = th.bool)
training_data.to(device)
output_fh.close()
theta_fh = open(train_theta_path, "rb")
training_params = th.as_tensor(np.load(theta_fh)).float()
training_params.to(device)
theta_fh.close()
datasets["train"] = th.utils.data.TensorDataset(training_data, training_params)

output_fh = open(validation_output_path, "rb")
validation_data = th.as_tensor(np.load(output_fh), dtype = th.bool)
output_fh.close()
theta_fh = open(validation_theta_path, "rb")
validation_params = th.as_tensor(np.load(theta_fh)).float()
theta_fh.close()
datasets["validation"] = th.utils.data.TensorDataset(validation_data, validation_params)

# Define paths to our training, test, and validation sets.
print("Datasets initated.")




# Put it into a dataloader.
data_loaders = {key: th.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers = 1, pin_memory = True)
                for key, dataset in datasets.items()}

"""
Defining loss function
"""

def evaluate_negative_log_likelihood(theta: th.Tensor, dist: th.distributions.Distribution) \
    -> th.Tensor:
    """
        Evaluate the negative log likelihood of a minibatch to minimize the expected posterior entropy
        directly (just taking mean for the Monte Carlo Estimate)
    """
    loss = - dist.log_prob(theta)
    return loss.mean()

loss_function = evaluate_negative_log_likelihood

"""
Defining Neural Nets
"""
# Number of features and number of components are passed in through the argument.

print("Defining neural networks")

# The relevant neural nets are defined in nn_utils.
# But since our dimension is now high, let's get ourselves 50 nodes in the two hidden layers for now.
compressor = nn_utils.DenseStack(compressor_layers, th.nn.Tanh())
module = nn_utils.MixtureDensityNetwork_multi_param(compressor, mdn_layers, num_param, th.nn.Tanh())

"""
Optimization and Training
"""

# Define an optimizer and a scheduler.
optimizer = th.optim.AdamW(module.parameters(), learning_rate)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)

print("Beginning training")

# Run the training.
epoch = 0
best_validation_loss = float('inf')
num_bad_epochs = 0
while num_bad_epochs < patience and (max_epochs is None or epoch < max_epochs):
    # Run one epoch using minibatches.
    start_time = time.time()
    train_loss = 0
    for step, (x, theta) in enumerate(data_loaders['train']):
        # M: We enumerate and get a step, an x, and a theta for each 10 x 3 datapoint.

        # Get the output of module, which is a distribution from the MDN fitting.
        y = module(x.float())
        # Needed to apply the float() function here since we "expect object of type float

        # The loss function is obtained from evaluate_negative_log_likelihood.
        loss: th.Tensor = loss_function(theta, y)
        # There's an error here where we're trying to plug in something of our batch size, but we expect something of size 2.
        assert not loss.isnan()

        optimizer.zero_grad() # Clear the gradients from last step.
        loss.backward() # Compute derivative of loss wrt to parameters.
        optimizer.step() # Take a step based on the gradients.

        # Extract the loss value as a float, to keep a running sum for this epoch.
        train_loss += loss.item()
        if int(step)%1000 == 0:
            print(step)
    # Get the average training loss.
    train_loss /= step + 1

    # Evaluate the validation loss and update the learning rate if required.
    # The validation loss is calculated by sticking the training set (x's and corresponding thetas)
    # into the module (that has gone one step of optimization), and then getting the EPE.
    validation_loss = sum(loss_function(theta, module(x.float())).mean().item() for x, theta
                              in data_loaders['validation']) / len(data_loaders['validation'])
    scheduler.step(validation_loss)

    # Update the best validation loss.
    # M: An epoch is "bad" if our validation loss did not improve (get lower).
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        num_bad_epochs = 0
    else:
        num_bad_epochs += 1

    epoch += 1
    print("epoch %3d: train loss = %.3f; validation loss = %.3f \nbest validation loss = %.3f; number bad epochs = %d / %d" % (epoch, train_loss, validation_loss,
                  best_validation_loss, num_bad_epochs, patience))
    end_time = time.time()
    tot_time = (end_time - start_time)/60
    print("Minutes for iteration: " + str(tot_time))
    # Save temporary models, hust in case the script times out or has other error.
    path_to_models = "models/"
    if not os.path.exists(path_to_models):
        os.makedirs(path_to_models)
    th.save(compressor, path_to_models + "compressor_temp.pt")
    th.save(module, path_to_models + "mdn_temp.pt")
print("Training complete, saving results")

"""
Save results
"""
path_to_models = r"models/" # Where we'll store our data.
if not os.path.exists(path_to_models):
    os.makedirs(path_to_models)

compressor_path = path_to_models + "compressor.pt"
th.save(compressor, compressor_path)
mdn_path = path_to_models + "mdn.pt"
th.save(module, mdn_path)

if os.path.exists(path_to_models + "compressor_temp.pt"):
    os.remove(path_to_models + "compressor_temp.pt")
if os.path.exists(path_to_models + "mdn_temp.pt"):
    os.remove(path_to_models + "mdn_temp.pt")

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))
