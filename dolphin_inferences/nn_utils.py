"""
Using code adapted from Till Hoffmann's work for application in epidemic inferences.
"""

import torch as th
import typing
import pickle
import os
import epidemic_utils
import random
import _pickle as cPickle
import numpy as np

class DolphinDataset(th.utils.data.Dataset):
    def __init__(self, root, mode, num_samples, device):
        self.root = root
        self.num_samples = int(num_samples)
        self.mode = mode
        self.device = device
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        idx_temp = idx
        dir_name = self.mode + "_dir_" + str(idx_temp//1000) + "/"
        path = str(self.root) + dir_name + str(self.mode) + "_data_" + str(idx) + ".pkl"
        while not os.path.exists(path) or not os.path.getsize(path) > 0:
            # If we actually have a file that doesn't exist, just grab a random different one.
            idx_temp = random.randint(1,self.num_samples)
            dir_name = self.mode + "_dir_" + str(idx_temp//1000) + "/"
            path = str(self.root) + dir_name + str(self.mode) + "_data_" + str(idx_temp) + ".pkl"
        fh = open(path, "rb")
        depickled_info = cPickle.load(fh)
        fh.close()
        parameters = th.as_tensor(depickled_info["theta"])
        output = th.as_tensor(depickled_info["output"])
        return output, parameters


class NPDataset(th.utils.data.Dataset):
    def __init__(self, root, mode, num_samples, device, samples_per_dir):
        self.root = root + "/" + str(mode) + "/"
        self.num_samples = int(num_samples)
        self.mode = mode
        self.device = device
        self.samples_per_dir = samples_per_dir
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        idx_temp = idx
        dir_name = self.mode + "_dir_" + str((idx_temp//self.samples_per_dir) + 1) + "/"
        path_output = str(self.root) + dir_name + str(self.mode) + "_data_output_" + str(idx_temp) + ".npy"
        path_thetas = str(self.root) + dir_name + str(self.mode) + "_data_theta_" + str(idx_temp) + ".npy"
        redraw_count = 0
        while not os.path.exists(path_output) or not os.path.getsize(path_output) > 0 or not os.path.exists(path_thetas) or not os.path.getsize(path_thetas) > 0:
            redraw_count += 1
            print(path_output)
            # If we actually have a file that doesn't exist, just grab a random different one.
            idx_temp = random.randint(1,self.num_samples)
            dir_name = self.mode + "_dir_" + str((idx_temp//self.samples_per_dir) + 1) + "/"
            path_output = str(self.root) + dir_name + str(self.mode) + "_data_output_" + str(idx_temp) + ".npy"
            path_thetas = str(self.root) + dir_name + str(self.mode) + "_data_theta_" + str(idx_temp) + ".npy"
            if redraw_count > 100:
                print("ERROR: Too many redraws.")
                break
        if redraw_count > 0:
            print("redrew :" + str(redraw_count))
        fh1 = open(path_output, "rb")
        fh2 = open(path_thetas, "rb")
        output = th.as_tensor(np.load(fh1))
        parameters = th.as_tensor(np.load(fh2))
        fh1.close()
        fh2.close()
        return output, parameters

class DenseStack(th.nn.Module):
    """
    Apply a sequence of dense layers.
    Args:
        num_nodes: Sequence of number of hidden nodes. The first number of nodes must match the
            input.
        activation: Activation function.
    """

    def __init__(self, num_nodes: typing.Iterable[int], activation: th.nn.Module):
        super().__init__()
        layers = []
        for i, j in zip(num_nodes, num_nodes[1:]):
            layers.extend([th.nn.Linear(i, j), activation])
        self.layers = th.nn.Sequential(*layers[:-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)

class MixtureDensityNetwork_multi_param(th.nn.Module):
    """
    Simple Beta mixture density network the network. Using two-dimensional beta distributions.
    Args:
        compressor: Module to compress data to summary statistics.
        expansion_nodes: Number of nodes in hidden layers to expand from statistics to mixture
            density parameters. The first number of nodes must match the number of statistics. The
            last number of nodes is the number of components of the mixture.
            This should be a list of ints.
    """
    def __init__(self, compressor: th.nn.Module, expansion_nodes, num_param,
                 activation: th.nn.Module) -> None:
        super().__init__()
        self.compressor = compressor
        # Build stacks for mixture weights and beta-distribution concentration parameters.
        self.logits = DenseStack(expansion_nodes, activation)
        self.log_a = DenseStack([*expansion_nodes[:-1], num_param * expansion_nodes[-1]], activation)
        self.log_b = DenseStack([*expansion_nodes[:-1], num_param * expansion_nodes[-1]], activation)
        self.num_param = num_param

    def forward(self, x: th.Tensor) -> th.distributions.Distribution:
        # Compress the data and estimate properties of the Gaussian copula mixture.
        y: th.Tensor = self.compressor(x)
        logits = self.logits(y)
        a = self.log_a(y).exp() # Require the exp() to enforce the domain of a and b as positives.
        b = self.log_b(y).exp()

        # This breaks a and b into a list of triples (or as many parameters as needed.)
        a = a.reshape((*a.shape[:-1], -1, self.num_param))
        b = b.reshape((*b.shape[:-1], -1, self.num_param))

        component_distribution = th.distributions.Independent(th.distributions.Gamma(a, b), 1)

        return th.distributions.MixtureSameFamily(
            th.distributions.Categorical(logits=logits),
            component_distribution,
        )
