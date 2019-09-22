# Neural Network Sparsification via Pruning #

**A deep learning repository for pruning neural networks.**

For any questions and comments, please send correspondences to *Daniel Eftekhari* at daniel.eftekhari@mail.utoronto.ca

**Requirements**:

Python 3.5 or higher (tested on Python 3.6).<br />
PyTorch 1.2.0.

Specific module and version requirements are listed in requirements.txt. After cloning the repository,<br />
cd to the repo directory, and enter `pip3 install -r requirements.txt`<br />
Note: You may need to run `sudo apt-get install python3-tk` afterwards to have matplotlib work correctly.

**File Descriptions**:

config.py -> complete list of command-line parsable arguments<br />
main.py -> includes training and pruning call logic<br />
model.py -> the neural network is constructed here<br />
ops.py -> a custom neural network forward function that optimizes matrix multiplications for sparse tensors<br />
plotting.py -> plotting utils<br />
pruning.py -> weight and unit pruning functions are defined here<br />
utils.py -> miscellaneous utils

**Usage**:

**Training**<br />
To train a neural network on the MNIST classification task, enter `python3 main.py`. See config.py for a detailed list of arguments which can be passed in,
including passing the neural network architecture as an argument (by default it is set to a ReLU activated neural network with 4 hidden layers).

**Pruning**<br />
To prune the neural network with weight-pruning, enter<br />
`python3 main.py --train=False --prune_type=Weight --use_gpu=False`<br />
To prune the neural network with unit-pruning, enter<br />
`python3 main.py --train=False --prune_type=Unit --use_gpu=False`<br />
Here, the --use_gpu option is set to False in order to reproduce the timing results.
