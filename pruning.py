import torch


def weight_prune(weights, k):
    """
    Prune <weights> for a given layer, removing <k> fraction of them.
    Note that this removes the smallest <k> fraction of any weights in the weight matrix, so that in most scenarios
    every neuron will have surviving connections. This has important effects in model performance,
    compared to the <unit_prune> approach (see <unit_prune> function and attached report).

    Special note: a heap data structure (using heapq library) was investigated in place of using sort,
    for identifying the smallest <k> fraction of weights. Asymptotically, a heap data structure is O(n log(nk)), k < 1,
    for this task, whereas sort is O(n log(n)), but sort still performed better in practice
    due to torch.sort() being optimized for torch tensors. However, in future if pytorch does support heap data
    structures, this may be done more efficiently.
    """

    # obtain the original shape of weights before flattening
    shape = weights.shape
    weights = torch.flatten(weights)

    # obtain indices corresponding to absolute weight values, sorted from low to high
    _, indices = torch.abs(weights).sort()
    # remove the smallest <k> fraction of weights and reshape
    weights[indices[0:int(k*indices.shape[0])]] = 0
    return weights.view(shape)


def unit_prune(weights, k):
    """
    Prune all weights corresponding to a unit in a given layer, for <k> fraction of the units.
    Note that this removes the smallest <k> fraction of units in the weight matrix, so that entire neurons are dropped.
    This has important effects in model performance, compared to the <weight_prune> approach
    (see <weight_prune> function and attached report).

    Special note: see special note in function <weight_prune>.
    """

    # obtain indices corresponding to l2-norm of unit weights, sorted from low to high
    _, indices = torch.sum(weights**2, dim=1).sort()
    # remove the smallest <k> fraction of units
    weights[indices[0:int(k*indices.shape[0])], :] = 0
    return weights
