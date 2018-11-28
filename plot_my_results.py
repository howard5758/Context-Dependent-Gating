import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib


def get_accuracy(data_dir, prefix, color = [0,0,1], split = 1, description = [], label=None, linestyle = '-'):
   # list of accuracies

    task_accuracy = []

    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix):
            print('Found file', full_fn)
            x = pickle.load(open(data_dir + full_fn, 'rb'))
            task_accuracy.append(x['accuracy_full'])

    accuracies = np.zeros(len(task_accuracy[0]))
    task = np.zeros(len(task_accuracy[0]))


    for i in range(0, len(task_accuracy[0])):
        task[i] = i
        accuracies[i] = task_accuracy[0][i]

    return accuracies



def plot_mnist():

    print('Ploting mnist results')
    savedir = 'savedir/mnist/'


    all_accuracies = np.zeros((4, 100))


    SI_partial = all_accuracies[0, :] = get_accuracy(savedir, 'mnist_SI_partial')
    SI_XdG = all_accuracies[1, :] = get_accuracy(savedir, 'mnist_SI_XdG')
    EWC_partial = all_accuracies[2, :] = get_accuracy(savedir, 'mnist_EWC_partial')
    EWC_XdG = all_accuracies[3, :] = get_accuracy(savedir, 'mnist_EWC_XdG')



    all_labels = ['SI+partial', 'SI+XdG', 'EWC+partial', 'EWC+XdG']  

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)


    for i in range(0, 4):
        ax1.plot(all_accuracies[i, :], label=i)

    

    colormap = ['b', 'g', 'r', 'c']
    for i, j in enumerate(ax1.lines):
        j.set_color(colormap[i])
        j.set_label(all_labels[i])

    ax1.legend(loc=1)
    ax1.set_title('Accuracies of four combinations on MNIST dataset')
    plt.xlabel('Number of tasks')
    plt.ylabel('Accuracy')
    plt.savefig(savedir + 'mnist_results.png', format='png')
    plt.show()


def plot_fashion_mnist():

    print('Ploting fashion mnist results')
    savedir_fashion = 'savedir/fashion_mnist/'
    savedir_mnist = 'savedir/mnist/'
    savedir_cifar = 'savedir/cifar/'


    all_accuracies = np.zeros((4, 100))

    fashion_mnist = all_accuracies[0, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
    mnist = all_accuracies[1, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
    cifar = all_accuracies[2, :] = get_accuracy(savedir_cifar, 'cifar_SI_XdG')

    all_labels = ['Fashion MNIST', 'MNIST', 'CIFAR']  

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for i in range(0, 4):
        ax1.plot(all_accuracies[i, :], label=i)

    colormap = ['b', 'g', 'r']
    for i, j in enumerate(ax1.lines):
        j.set_color(colormap[i])
        j.set_label(all_labels[i])

    ax1.legend(loc=1)
    ax1.set_title('Accuracies of SI-XdG combinations on Fashion-MNIST, MNIST, CIFAR dataset')
    plt.xlabel('Number of tasks')
    plt.ylabel('Accuracy')
    plt.savefig(savedir_fashion + 'fashion_mnist_results.png', format='png')
    plt.show()


def plot_cifar():

    print('Ploting cifar results')
    savedir = 'savedir/cifar/'

    all_accuracies = np.zeros((4, 20))

    SI_partial = all_accuracies[0, :] = get_accuracy(savedir, 'cifar_SI_partial')
    SI_XdG = all_accuracies[1, :] = get_accuracy(savedir, 'cifar_SI_XdG')
    EWC_partial = all_accuracies[2, :] = get_accuracy(savedir, 'cifar_EWC_partial')
    EWC_XdG = all_accuracies[3, :] = get_accuracy(savedir, 'cifar_EWC_XdG')

    all_labels = ['SI+partial', 'SI+XdG', 'EWC+partial', 'EWC+XdG']
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for i in range(0, 4):
        ax1.plot(all_accuracies[i, :], label=i)

    colormap = ['b', 'g', 'r', 'c']
    for i, j in enumerate(ax1.lines):
        j.set_color(colormap[i])
        j.set_label(all_labels[i])

    ax1.legend(loc=1)
    ax1.set_title('Accuracies of four combinations on CIFAR dataset')
    plt.xticks([0, 5, 10, 15, 20])
    plt.xlabel('Number of tasks')
    plt.ylabel('Accuracy')
    plt.savefig(savedir + 'cifar_results.png', format='png')
    plt.show()




print('Generating plots')
#plot_mnist()
#plot_cifar()
plot_fashion_mnist()

    