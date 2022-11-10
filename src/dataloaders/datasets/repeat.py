import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, length, dim):
        assert length % 2 == 0
        self.length = length
        self.dim = dim
        self.n_samples = n_samples

        raw_inputs = torch.rand(n_samples, self.length // 2, self.dim)
        # inputs = torch.tensor(inputs)

        inputs = []
        labels = []
        for x in raw_inputs:
            label = []
            indices = list(range(length // 2)) * 2
            random.shuffle(indices)
            label = [0] * length
            pos = [0] * (length // 2)
            
            for i, idx in enumerate(indices):
                pos[idx] += 1
                if pos[idx] > 1:
                    label[i] = 1

            inputs.append(x[indices])
            labels.append(label)
        inputs = torch.stack(inputs, dim=0)
        labels = torch.tensor(labels)

        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return self.n_samples


class RepeatRegDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, length, dim):
        assert length % 2 == 0
        self.length = length
        self.dim = dim
        self.n_samples = n_samples

        raw_inputs = torch.rand(n_samples, self.length // 2, self.dim)
        # inputs = torch.tensor(inputs)

        inputs = []
        labels = []
        for x in raw_inputs:
            label = []
            indices = list(range(length // 2)) * 2
            random.shuffle(indices)
            label = [0.] * length
            pos = [-1] * (length // 2)
            
            for i, idx in enumerate(indices):
                if pos[idx] != -1:
                    label[i] = i - pos[idx]
                else:
                    pos[idx] = i

            inputs.append(x[indices])
            labels.append(label)
        inputs = torch.stack(inputs, dim=0)
        labels = torch.tensor(labels)

        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return self.n_samples