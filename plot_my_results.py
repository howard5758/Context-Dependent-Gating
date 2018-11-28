import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib


def plot_best_result(ax, data_dir, prefix, color = [0,0,1], split = 1, description = [], label=None, linestyle = '-'):

    # Get filenames
    name_and_data = []
    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix):
            x = pickle.load(open(data_dir + full_fn, 'rb'))
            name_and_data.append((full_fn, x['accuracy_full'][-1], x['par']['omega_c']))

    # Find number of c's and v's
    cids = []
    vids = []
    xids = []
    om_c = []
    for (f, _, oc) in name_and_data:

        if 'xi' in f:
            if f[-12] not in cids:
                cids.append(f[-12])
            if f[-8] not in xids:
                xids.append(f[-8])
            if f[-5] not in vids:
                vids.append(f[-5])
        else:
            if f[-9].isdigit():
                c = f[-9:-7]
            else:
                c = f[-8]
            if c not in cids:
                cids.append(c)
            if f[-5] not in vids:
                vids.append(f[-5])
            xids = [0]
        om_c.append(oc)


    accuracies = np.zeros((len(xids),len(cids)))
    count = np.zeros((len(xids),len(cids)))
    cids = sorted(cids)
    vids = sorted(vids)
    xids = sorted(xids)

    print(prefix, cids, vids, xids)

    for i, c_id, in enumerate(cids):
        for v_id in vids:
            for j, x_id in enumerate(xids):
                #text_c = 'omega'+str(c_id)
                text_c = 'omega'+str(c_id)
                text_v = '_v'+str(v_id)
                text_x = '_xi'+str(x_id)
                for full_fn in os.listdir(data_dir):
                    if full_fn.startswith(prefix) and 'xi' in full_fn and text_c in full_fn and text_v in full_fn and text_x in full_fn:
                        #print('c_id', c_id)
                        x = pickle.load(open(data_dir + full_fn, 'rb'))
                        accuracies[j,i] += x['accuracy_full'][-1]
                        count[j,i] += 1
                    elif full_fn.startswith(prefix) and not 'xi' in full_fn  and text_c in full_fn and text_v in full_fn:
                        #print('c_id', c_id)
                        x = pickle.load(open(data_dir + full_fn, 'rb'))
                        accuracies[j,i] += x['accuracy_full'][-1]
                        count[j,i] += 1

    accuracies /= (1e-16+count)
    accuracies = np.reshape(accuracies,(1,-1))
    print(prefix)
    print(accuracies)
    ind_best = np.argsort(accuracies)[-1]
    best_c = int(ind_best[-1]%len(cids))
    best_xi = ind_best[-1]//len(cids)
    task_accuracy = []


    for v_id in vids:
        #text_c = 'omega'+str(cids[best_c])
        text_c = 'omega'+str(cids[best_c])
        text_xi = 'xi'+str(xids[best_xi])
        text_v = '_v'+str(v_id)

        print(prefix, text_c, text_xi, text_v)

        for full_fn in os.listdir(data_dir):
            if full_fn.startswith(prefix)  and 'xi' in full_fn and text_c in full_fn and text_v in full_fn and text_x in full_fn:
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                task_accuracy.append(x['accuracy_full'])
                print(prefix,' ', full_fn, ' ', x['par']['stabilization'], ' omega C ', x['par']['omega_c'])
            elif full_fn.startswith(prefix)  and not 'xi' in full_fn and text_c in full_fn and text_v in full_fn:
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                task_accuracy.append(x['accuracy_full'])
                print(prefix,' ', full_fn, ' ', x['par']['stabilization'], x['par']['gating_type'], x['par']['multihead'], ' omega C ', x['par']['omega_c'], x['par']['omega_xi'])

    task_accuracy = np.mean(np.stack(task_accuracy),axis=0)


    if split > 1:
        task_accuracy = np.array(task_accuracy)
        task_accuracy = np.tile(np.reshape(task_accuracy,(-1,1)),(1,split))
        task_accuracy = np.reshape(task_accuracy,(1,-1))[0,:]

    if not description == []:
        print(description , ' ACC after 10 trials = ', task_accuracy[9],  ' after 30 trials = ', task_accuracy[29],  \
            ' after 100 trials = ', task_accuracy[99])

    ax.plot(np.arange(1, np.shape(task_accuracy)[0]+1), task_accuracy, color = color, linestyle = linestyle, label=label)

    return task_accuracy[[9,-1]]

def plot_mnist():
    print('Ploting mnist results')
    savedir = './/savedir//mnist//'

    ax1 = plt.subplots(1, 4, figsize=(4, 2.5))
    accuracy = {}

    ylim_min = 0.6

    accuracy['SI_XdG'] = plot_best_result(ax1, savedir, 'mnist_SI_XdG', label='SI + XdG', linestyle = '-')
    accuracy['SI_partial'] = plot_best_result(ax1, savedir, 'mnist_SI_partial', label='SI + partial', linestyle = '-')
    accuracy['EWC_XdG'] = plot_best_result(ax1, savedir, 'mnist_EWC_XdG', label='EWC + XdG', linestyle = '--')
    accuracy['EWC_partial'] = plot_best_result(ax1, savedir, 'mnist_EWC_partial', label='EWC + partial', linestyle = '--')

    ax1.legend(ncol=4, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [ylim_min,1], [0,100], [])

    plt.tight_layout()
    plt.savefig(savedir + 'mnist_results.png', format='png')
    plt.show()


def plot_fashion_mnist():
    print('Ploting Fashion-mnist results')
    savedir_fashion = './/savedir//fashion-mnist//'
    savedir_mnist = './/savedir//mnist//'
    savedir_cifar = './/savedir//cifar//'

    ax1 = plt.subplots(1, 3, figsize=(4, 2.5))
    accuracy = {}

    ylim_min = 0.6

    accuracy['fashion_mnist_SI_XdG'] = plot_best_result(ax1, savedir_fashion, 'fashion_mnist_SI_XdG', label='Fashion-MNIST', linestyle = '-')
    accuracy['mnist_SI_XdG'] = plot_best_result(ax1, savedir_mnist, 'mnist_SI_XdG', label='MNIST', linestyle = '-')
    accuracy['cifar_SI_XdG'] = plot_best_result(ax1, savedir_cifar, 'cifar_SI_XdG', label='CIFAR', linestyle = '-')

    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [ylim_min,1], [0,100], [])

    plt.tight_layout()
    plt.savefig(savedir_fashion + 'fashion_mnist_results.png', format='png')
    plt.show()

def plot_cifar():
    print('Ploting CIFAR results')
    savedir = './/savedir//cifar//'

    ax1 = plt.subplots(1, 4, figsize=(4, 2.5))
    accuracy = {}

    ylim_min = 0.6

    accuracy['SI_XdG'] = plot_best_result(ax1, savedir, 'cifar_SI_XdG', label='SI + XdG', linestyle = '-')
    accuracy['SI_partial'] = plot_best_result(ax1, savedir, 'cifar_SI_partial', label='SI + partial', linestyle = '-')
    accuracy['EWC_XdG'] = plot_best_result(ax1, savedir, 'cifar_EWC_XdG', label='EWC + XdG', linestyle = '--')
    accuracy['EWC_partial'] = plot_best_result(ax1, savedir, 'cifar_EWC_partial', label='EWC + partial', linestyle = '--')

    ax1.legend(ncol=4, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [ylim_min,1], [0,100], [])

    plt.tight_layout()
    plt.savefig(savedir + 'cifar_results.png', format='png')
    plt.show()



print('Generating plots')
#plot_mnist()
#plot_cifar()
plot_fashion_mnist()
    