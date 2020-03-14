import argparse

parser = argparse.ArgumentParser(description='MNIST-SPARSE')


def parsebool(x):
    return x.lower() == 'true'


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


# neural network params
nn_arg = add_argument_group('Neural Network Params')
nn_arg.add_argument('--units', type=list, default=[1000, 1000, 500, 200],
                    help='number of units per fully connected layer; a modular approach'
                         'so that the number of units per layer (and hence number of layers) can be given as arguments')

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--img_size', type=tuple, default=(1, 28, 28),
                      help='size of input images')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--train', type=parsebool, default=True,
                       help='whether to train (<True>) or prune (<False>)')
train_arg.add_argument('--epochs', type=int, default=25,
                       help='number of epochs to train for')
train_arg.add_argument('--batch_size', type=int, default=128,
                       help='batch size during training')
train_arg.add_argument('--use_sgd', type=parsebool, default=False,
                       help='whether to use sgd or adam optimizer')
train_arg.add_argument('--lr', type=float, default=3e-4,
                       help='learning rate')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='momentum')
train_arg.add_argument('--use_nesterov', type=parsebool, default=True,
                       help='whether to use nesterov momentum')
train_arg.add_argument('--beta1', type=float, default=0.9,
                       help='beta1 in adam optimizer')
train_arg.add_argument('--beta2', type=float, default=0.999,
                       help='beta2 in adam optimizer')
train_arg.add_argument('--shuffle', type=parsebool, default=True,
                       help='whether to shuffle training data')
train_arg.add_argument('--num_workers', type=int, default=4,
                       help='number of workers/subprocesses to use in dataloader')

# sparse params
sparse = add_argument_group('Pruning Params')
sparse.add_argument('--sparsity', type=list,
                       default=[0., 0.10, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99, 1.],
                       help='fractional sparsities (k) to apply to the network')
sparse.add_argument('--prune_type', type=str, default='Weight',
                       help='use <Weight> or <Unit> pruning')
sparse.add_argument('--to_sparse', type=parsebool, default=True,
                       help='whether to use a coordinate list (COO) sparse representation of the weight matrices'
                            'using the pytorch to_sparse() module')
sparse.add_argument('--use_sparse_mul', type=parsebool, default=True,
                       help='whether to use torch.sparse.mm or the standard torch.mm matrix multiply module'
                            'note that to_sparse must be set to <True> if torch.sparse.mm is to be used')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--model_name', type=str, default='mnist-sparse',
                      help="descriptive model name")
misc_arg.add_argument('--stdout_dir', type=str, default='./stdout',
                      help="directory to log program stdout to")
misc_arg.add_argument('--model_dir', type=str, default='./ckpt',
                      help='directory in which to save model checkpoints')
misc_arg.add_argument('--use_gpu', type=parsebool, default=True,
                      help="whether to use a gpu, if available")
misc_arg.add_argument('--random_seed', type=int, default=2,
                      help='seed for reproducibility')
misc_arg.add_argument('--save_model', type=parsebool, default=True,
                      help='whether to save the model, if validation loss improves, at the end of each epoch')
misc_arg.add_argument('--plot', type=parsebool, default=True,
                      help='whether to plot performance metrics')
misc_arg.add_argument('--plot_dir', type=str, default='./plots',
                      help='directory in which to save plots')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
