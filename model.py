import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dim, c, units):
        super(Net, self).__init__()

        self.units = [dim] + units + [c]
        self.layers = nn.ModuleList()
        for i in range(len(self.units)-1):
            self.layers.append(nn.Linear(self.units[i], self.units[i+1], bias=False))

    def forward(self, x):
        # flatten input, as we are only using fully connected layers
        x = x.view(x.shape[0], -1)

        # apply a ReLU activation to all but the last layer leading to the logits
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
