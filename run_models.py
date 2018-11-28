import numpy as np
from parameters import *
import model
import sys, os
import pickle
import time


def try_model(save_fn, gpu_id = None):
    # Specify GPU Id as a string
    # Leave blank to use CPU
    try:
        model.main(save_fn, gpu_id)
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')


###############################################################################
###############################################################################
###############################################################################


mnist_updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'mnist',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 3906,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

fashion_mnist_updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'fashion-mnist',
    'n_train_batches'       : 3906,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

cifar_updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'n_tasks'               : 50,
    'task'                  : 'cifar',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 977,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }

imagenet_updates = {
    'layer_dims'            : [4096, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'imagenet',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 977*2,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }

# updates for multi-head network, Cifar and Imagenet only
multi_updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}
imagenet_multi_updates = {'layer_dims':[4096, 2000, 2000, 1000], 'multihead': True}

# updates for split networks
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10], 'multihead': False}
fashion_mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10], 'multihead': False}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5], 'multihead': False}
imagenet_split_updates = {'layer_dims':[4096, 3665, 3665, 10], 'multihead': False}


# training a network on 100 sequential MNIST permutations using synaptic intelligence 
# and context-dependent gating (XdG) 
def run_mnist_SI_model(gpu_id):
    print('MNIST - Synaptic Stabilization = SI - Gating = 80%')
    update_parameters(mnist_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
    update_parameters({'stabilization': 'EWC', 'omega_c': 0.035, 'omega_xi': 0.01})
    save_fn = 'mnist_EWC_partial.pkl'
    try_model(save_fn, gpu_id)

# training a network on 100 sequential Fashion-MNIST permutations using synaptic intelligence 
# and context-dependent gating (XdG) 
def run_fashion_mnist_SI_model(gpu_id):
    print('Fashion-MNIST - Synaptic Stabilization = SI - Gating = 80%')
    update_parameters(fashion_mnist_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
    update_parameters({'stabilization': 'pathint', 'omega_c': 0.035, 'omega_xi': 0.01})
    save_fn = 'fashion_mnist_SI_XdG.pkl'
    try_model(save_fn, gpu_id)

# training a network on 20 sequential CIFAR permutations using synaptic intelligence 
# and context-dependent gating (XdG) 
def run_cifar_SI_model(gpu_id):
    print('CIFAR - Synaptic Stabilization = SI - Gating = 80%')
    update_parameters(cifar_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 1.0})
    update_parameters({'stabilization': 'pathint', 'omega_c': 0.75, 'omega_xi': 0.01})
    update_parameters({'train_convolutional_layers': False})
    save_fn = 'cifar_SI_XdG_50.pkl'
    try_model(save_fn, gpu_id)

# training a network on 100 sequential Imagenet tasks using synaptic intelligence 
# and context-dependent gating (XdG) 
def run_imagenet_SI_model(gpu_id):
    print('ImageNet - Synaptic Stabilization = SI - Gating = 80%')
    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 1.0})
    update_parameters({'stabilization': 'pathint', 'omega_c': 0.75, 'omega_xi': 0.01})
    update_parameters({'train_convolutional_layers': True})
    save_fn = 'imagenet_SI_XdG.pkl'
    try_model(save_fn, gpu_id)


if __name__ == "__main__":
    '''
    mnist_start_time = time.time()
    run_mnist_SI_model("0")
    mnist_end_time = time.time()
    mnist_run_time = mnist_end_time - mnist_start_time
    print('EWC_XdG finished, took', mnist_run_time, 'seconds')
    '''
    '''
    fashoin_mnist_start_time = time.time()
    run_fashion_mnist_SI_model("0")
    fashoin_mnist_end_time = time.time()
    fashion_mnist_run_time = fashoin_mnist_end_time - fashoin_mnist_start_time
    print('SI_XdG finished, took', fashion_mnist_run_time, 'seconds')
    '''
    
    cifar_start_time = time.time()
    run_cifar_SI_model("0")
    cifar_end_time = time.time()
    cifar_run_time = cifar_end_time - cifar_start_time
    print('cifar_SI_XdG finished, took', cifar_run_time, 'seconds')
    
    
    '''
    imagenet_start_time = time.time()
    run_imagenet_SI_model("0")
    imagenet_end_time = time.time()
    imagenet_run_time = imagenet_end_time - imagenet_start_time
    print('imagenet_SI_model finished, took', imagenet_run_time, 'seconds')
    '''
