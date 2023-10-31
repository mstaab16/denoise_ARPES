import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from dlsia.core import helpers
from dlsia.core.train_scripts import train_regression, train_segmentation
from dlsia.core.networks import smsnet, baggins, tunet
from dlsia.test_data.two_d import build_test_data, torch_hdf5_loader
from dlsia.viz_tools import plots, draw_sparse_network
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from qlty import qlty2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from dataset import NpyDataset

Y=1024
X=1024
window=(256,256)
step=(256,256)
border=None
border_weight=0

train_data = NpyDataset(start_idx=0, end_idx=100, Y=Y, X=X, window=window, step=step, border=border, border_weight=border_weight)
validation_data = NpyDataset(start_idx=100, end_idx=150, Y=Y, X=X, window=window, step=step, border=border, border_weight=border_weight)
# qlty_obj = qlty2D.NCYXQuilt(Y=Y, X=X, window=window, step=step, border=border, border_weight=border_weight)


batch_size=16
num_workers=0

loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers}
train_loader = DataLoader(train_data, **loader_params)
loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers}
validation_loader = DataLoader(validation_data, **loader_params)

# SMSNet parameters
in_channels = 1
out_channels = 1 
num_layers = 20

# When alpha > 0, short-range skip connections are favoured
alpha = 0.75

# When gamma is 0, the degree of each node is chosen uniformly between 0 and max_k
# specifically, P(degree) \propto degree^-gamma
gamma = 0.75

# we can limit the maximum and minimum degree of our graph 
max_k = num_layers
min_k = 1

# features channel posibilities per edge
hidden_out_channels = [5] 

# possible dilation choices
dilation_choices = [1,2,3,4,5] 

# Here are some parameters that define how networks are drawn at random
# the layer probabilities dictionairy define connections
layer_probabilities={'LL_alpha': alpha,
                     'LL_gamma': gamma,
                     'LL_max_degree':max_k,
                     'LL_min_degree':min_k,
                     'IL': 0.1,
                     'LO': 0.1,
                     'IO': False}

# if desired, one can introduce scale changes (down and upsample)
# a not-so-thorough look indicates that this isn't really super beneficial
# in the model systems we looked at
sizing_settings = {'stride_base':2, #better keep this at 2
                   'min_power': 0,
                   'max_power': 0}

# defines the type of network we want to build

network_type = "Regression"

nets = [] 
n_networks = 5
epochs = 100     # Set number of epochs
criterion = nn.L1Loss()   # For segmenting 
learning_rate = 1e-3

for ii in range(n_networks):
    torch.cuda.empty_cache()
    print("Network %i"%(ii+1))
    smsnet_model = smsnet.random_SMS_network(in_channels=in_channels,
                                             out_channels=out_channels,
                                             in_shape=(32,32),
                                             out_shape=(32,32),
                                             sizing_settings=sizing_settings,
                                             layers=num_layers,
                                             dilation_choices=dilation_choices,
                                             hidden_out_channels=hidden_out_channels,
                                             layer_probabilities=layer_probabilities,
                                             network_type=network_type
                                            )
    
    # lets plot the network
    # net_plot,dil_plot,chan_plot = draw_sparse_network.draw_network(smsnet_model)
    # plt.show()
    print(f'Net {ii} created')

    nets.append(smsnet_model)
    
for net in nets:
    print("Start training")
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of refineable parameters: ", pytorch_total_params)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # Defined in loop, one per network    
    device = helpers.get_device()
    print(f'Using device: {device}')
    net = net.to(device)
    tmp = train_regression(net,
                           train_loader,
                           validation_loader,
                           epochs,
                           criterion,
                           optimizer,
                           device,
                           show=1,
                           use_amp=True)    
    smsnet_model = net.cpu()

for i, net in enumerate(nets[::-1]):
    net_name = f'nets\\10_31_2023\\net_{i}.pth'
    torch.save(net.state_dict(), net_name)
    # plots.plot_training_results_regression(tmp[1]).show()