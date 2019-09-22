import sys
import os
import time

import copy
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config import get_config
from model import Net
from pruning import weight_prune, unit_prune
from ops import forward
from plotting import plot_line, plot_hist
from utils import make_dirs, calculate_acc, get_average, append


def main(cfg):
    # setting up output directories, and writing to stdout
    make_dirs(cfg.stdout_dir, replace=False)
    if cfg.train:
        run_type = 'train'
    else:
        if 'weight' in cfg.prune_type.lower():
            run_type = 'weight-prune'
        else:
            run_type = 'unit-prune'
    sys.stdout = open('{}/stdout_{}_{}.txt'.format(cfg.stdout_dir, cfg.model_name, run_type), 'w')
    print(cfg)
    print('\n')
    sys.stdout.flush()

    # if train mode, replace the previous plot and ckpt directories; if in prune mode, use existing directories
    if cfg.plot:
        make_dirs(os.path.join(cfg.plot_dir, cfg.model_name), replace=cfg.train)
    if cfg.save_model:
        make_dirs(os.path.join(cfg.model_dir, cfg.model_name), replace=cfg.train)

    # set random seed
    if cfg.random_seed != 0:
        random_seed = cfg.random_seed
    else:
        random_seed = random.randint(1, 100000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # set device as cuda or cpu
    if cfg.use_gpu and torch.cuda.is_available():
        # reproducibility using cuda
        torch.cuda.manual_seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if cfg.use_gpu:
            print('gpu option was to <True>, but no cuda device was found')
            print('\n')

    # datasets and dataloaders
    # normalizing training and validation images to [0, 1] suffices for the purposes of our research objective
    # in training, <drop_last> minibatch in an epoch set to <True> for simplicity in tracking training performance
    dataset_train = MNIST(root='./data/mnist', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    dataset_val = MNIST(root='./data/mnist', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=100, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # automatically compute number of classes
    targets = np.asarray(dataset_train.targets)
    c = np.unique(targets).shape[0]

    # define model
    # weights initialized using Kaiming uniform (He initialization)
    # number of units per hidden layer is passed in as an argument
    net = Net(np.product(cfg.img_size), c, cfg.units).to(device)

    criterion = nn.CrossEntropyLoss()

    if cfg.train:
        # training mode

        if cfg.use_sgd:
            optimizer = optim.SGD(params=net.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.use_nesterov)
        else:
            optimizer = optim.Adam(params=net.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

        # tracking training and validation stats over epochs
        epochs = []
        train_loss_epochs, val_loss_epochs = [], []
        train_acc_epochs, val_acc_epochs = [], []

        # best model is defined as model with best performing validation loss
        best_loss = float('inf')
        for epoch in range(cfg.epochs):
            # tracking training and validation stats over a given epoch
            train_loss_epoch, val_loss_epoch = [], []
            train_acc_epoch, val_acc_epoch = [], []

            # training set
            for i, (x, y) in enumerate(dataloader_train):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = net(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                acc = calculate_acc(logits, y)

                append((train_loss_epoch, loss.item()), (train_acc_epoch, acc.item()))

            # validation set
            with torch.no_grad():
                for i, (x, y) in enumerate(dataloader_val):
                    x, y = x.to(device), y.to(device)

                    logits = net(x)
                    loss = criterion(logits, y)

                    acc = calculate_acc(logits, y)

                    append((val_loss_epoch, loss.item()), (val_acc_epoch, acc.item()))

            train_loss_epoch, val_loss_epoch = get_average(train_loss_epoch), get_average(val_loss_epoch)
            train_acc_epoch, val_acc_epoch = get_average(train_acc_epoch), get_average(val_acc_epoch)

            print('train_epoch{:0=3d}_loss{:.4f}_acc{:.4f}'.format(epoch+1, train_loss_epoch, train_acc_epoch))
            print('valid_epoch{:0=3d}_loss{:.4f}_acc{:.4f}'.format(epoch+1, val_loss_epoch, val_acc_epoch))
            print('\n')
            sys.stdout.flush()

            if cfg.plot:
                append((epochs, epoch+1), (train_loss_epochs, train_loss_epoch), (val_loss_epochs, val_loss_epoch),
                       (train_acc_epochs, train_acc_epoch), (val_acc_epochs, val_acc_epoch))

                plot_line(epochs, train_loss_epochs, val_loss_epochs, 'Epoch Number', 'Loss', cfg)
                plot_line(epochs, train_acc_epochs, val_acc_epochs, 'Epoch Number', 'Accuracy', cfg)

            if val_loss_epoch < best_loss:
                best_loss = val_loss_epoch
                print('New best model at epoch {:0=3d} with val_loss {:.4f}'.format(epoch+1, best_loss))
                print('\n')
                if cfg.save_model:
                    # save model when validation loss improves
                    save_name = '{}_net_epoch{:0=3d}_val_loss{:.4f}'.format(cfg.model_name, epoch+1, best_loss)
                    torch.save(net.state_dict(), os.path.join(cfg.model_dir, cfg.model_name, '{}.pth'.format(save_name)))
                    with open(os.path.join(cfg.model_dir, cfg.model_name, '{}.txt'.format(cfg.model_name)), 'w') as file:
                        file.write('{}.pth'.format(save_name))

    else:
        # pruning mode

        # checks on arguments passed in
        for k in cfg.sparsity:
            assert 0 <= k <= 1
        if cfg.use_sparse_mul:
            assert cfg.to_sparse

        # load model
        with open(os.path.join(cfg.model_dir, cfg.model_name, '{}.txt'.format(cfg.model_name)), 'r') as file:
            load_name = file.readline()
        net.load_state_dict(torch.load(os.path.join(cfg.model_dir, cfg.model_name, '{}'.format(load_name))))
        net.eval()

        # select pruning approach to use
        if 'weight' in cfg.prune_type.lower():
            prune = weight_prune
        else:
            prune = unit_prune

        sparsities = []
        val_loss_sparse, val_acc_sparse = [], []
        time_sparsities = []
        for k in cfg.sparsity:
            val_loss_k, val_acc_k = [], []
            time_k = []

            # copy network so that the sparsity changes are not additive for each k
            net_sparse = copy.deepcopy(net)

            pruned_weights = []
            # prune model, except for the last layer
            for (i, p) in enumerate(net_sparse.parameters()):
                if i < len(cfg.units):
                    original_weights = copy.deepcopy(p.data)
                    if cfg.plot:
                        # plot magnitude of original weights (for comparison to post-pruned weights)
                        plot_hist([torch.abs(original_weights.flatten()).cpu().numpy()], ['b'], cfg.prune_type, i+1, k, 'Non-Pruned Weight Magnitudes', 'Counts', cfg)
                    prune(p.data, k)
                    if cfg.plot:
                        # plot original magnitudes of pruned weights, and magnitudes of remaining weights, separately
                        pruned_weights_non_zero = torch.abs(original_weights.flatten()[p.data.flatten() != 0])
                        pruned_weights_zeroed = torch.abs(original_weights.flatten()[p.data.flatten() == 0])
                        plot_hist([pruned_weights_non_zero.cpu().numpy(), pruned_weights_zeroed.cpu().numpy()], ['g', 'r'], cfg.prune_type, i+1, k, 'Weight Magnitudes', 'Counts', cfg)
                        plot_hist([pruned_weights_non_zero.cpu().numpy()], ['k'], cfg.prune_type, i+1, k, 'Surviving Weight Magnitudes', 'Counts', cfg)
                if cfg.to_sparse and i < len(cfg.units):
                    pruned_weights.append(p.data.to_sparse())
                else:
                    pruned_weights.append(p.data)

            with torch.no_grad():
                for i, (x, y) in enumerate(dataloader_val):
                    x, y = x.to(device), y.to(device)

                    start = time.time()
                    logits = forward(x, pruned_weights, cfg.use_sparse_mul)
                    end = time.time()
                    loss = criterion(logits, y)

                    acc = calculate_acc(logits, y)

                    append((val_loss_k, loss.item()), (val_acc_k, acc.item()), (time_k, end-start))

            val_loss_k, val_acc_k, time_k = get_average(val_loss_k), get_average(val_acc_k), get_average(time_k)

            print('valid_{}_k{:.2f}_loss{:.4f}_acc{:.4f}'.format(run_type, k, val_loss_k, val_acc_k))
            print('valid_{}_k{:.2f}_time/minibatch{:.6f}'.format(run_type, k, time_k))
            print('\n')
            sys.stdout.flush()

            if cfg.plot:
                append((sparsities, k), (val_loss_sparse, val_loss_k), (val_acc_sparse, val_acc_k), (time_sparsities, time_k))

                plot_line(sparsities, [], val_loss_sparse, 'Sparsity {} Prune'.format(cfg.prune_type), 'Loss', cfg)
                plot_line(sparsities, [], val_acc_sparse, 'Sparsity {} Prune'.format(cfg.prune_type), 'Accuracy', cfg)
                plot_line(sparsities, [], time_sparsities, 'Sparsity {} Prune'.format(cfg.prune_type), 'Time', cfg)

            if cfg.save_model:
                torch.save(net_sparse.state_dict(), os.path.join(cfg.model_dir, cfg.model_name, '{}_sparse_net_{}_val_loss{:.4f}.pth'.format(cfg.model_name, run_type, val_loss_k)))


if __name__ == '__main__':
    cfg, unparsed = get_config()
    main(cfg)
