import torch
import torch.nn.functional as F


def forward(x, weights, use_sparse_mul):
    if use_sparse_mul:
        mm = torch.sparse.mm
    else:
        mm = torch.mm

    # flatten input, as we are only using fully connected layers
    x = x.view(x.shape[0], -1)

    # the two transposes are needed because <torch.sparse.mm> requires the first argument to be of type <SparseTensor>
    for i in range(len(weights)-1):
        x = F.relu(torch.t(mm(weights[i], torch.t(x))))
    x = torch.t(torch.mm(weights[-1], torch.t(x)))
    return x
