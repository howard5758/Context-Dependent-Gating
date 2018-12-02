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

    return task_accuracy[0]



def plot_mnist():

    print('Ploting mnist results')
    savedir = 'savedir/mnist/'
    png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/'


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
    plt.savefig(png_OutPath + 'mnist_results.png', format='png')
    plt.show()


def plot_fashion_mnist():

    print('Ploting fashion mnist results')
    savedir_fashion = 'savedir/fashion_mnist/'
    savedir_mnist = 'savedir/mnist/'
    savedir_cifar = 'savedir/cifar/'

    png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/'

    all_accuracies = np.zeros((4, 100))

    fashion_mnist = all_accuracies[0, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
    mnist = all_accuracies[1, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
    cifar = all_accuracies[2, 0:50] = get_accuracy(savedir_cifar, 'cifar_SI_XdG_50')

    all_labels = ['Fashion MNIST', 'MNIST', 'CIFAR']  

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    # all plotting first 50 tasks
    for i in range(0, 3):
    	ax1.plot(all_accuracies[i, 0:50], label=i)
        	

    # CIFAR only has 20 tasks
    #ax1.plot(all_accuracies[2, 0:20], label=2)

    colormap = ['b', 'g', 'r']
    for i, j in enumerate(ax1.lines):
        j.set_color(colormap[i])
        j.set_label(all_labels[i])

    ax1.legend(loc=1)
    ax1.set_title('Accuracies of SI-XdG combinations on Fashion-MNIST, MNIST, CIFAR dataset')
    plt.xlabel('Number of tasks')
    plt.ylabel('Accuracy')
    plt.savefig(png_OutPath + 'fashion_mnist_results.png', format='png')
    plt.show()


def plot_mix(type):

    if type == 'MixTraining':
        
        print('Ploting MNIST and Fashion-MNIST MixTraining results')
        
        savedir_mnist = 'savedir/mnist/'
        savedir_fashion = 'savedir/fashion_mnist/'
        savedir_mix = 'savedir/mix/' + type + '/'
        #savedir_cifar = 'savedir/cifar/'

        png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/MixTraining/'

        all_accuracies = np.zeros((20, 100))

        # 7 results in total
        mnist = all_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        fashion_mnist = all_accuracies[1, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
        mix_10_100_0 = all_accuracies[2, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_100_0')
        mix_10_100_50 = all_accuracies[3, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_100_50')
        mix_10_80_20 = all_accuracies[4, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_80_20')
        mix_10_60_40 = all_accuracies[5, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_60_40')
        mix_10_40_60 = all_accuracies[6, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_40_60')
        #cifar = all_accuracies[3, 0:50] = get_accuracy(savedir_cifar, 'cifar_SI_XdG_50')

        all_labels = ['MNIST', 
                      'Fashion MNIST', 
                      '10 100% MNIST, 0% Fashion-MNIST + 90 0% MNIST, 90% Fashion-MNIST', 
                      '10 100% MNIST, 0% Fashion-MNIST + 90 50% MNIST, 50% Fashion-MNIST',
                      '10 80% MNIST, 20% Fashion-MNIST + 90 20% MNIST, 80% Fashion-MNIST', 
                      '10 60% MNIST, 40% Fashion-MNIST + 90 40% MNIST, 60% Fashion-MNIST',
                      '10 40% MNIST, 60% Fashion-MNIST + 90 60% MNIST, 40% Fashion-MNIST',]  

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        # all plotting first 50 tasks
        for i in range(0, 7):
            ax1.plot(all_accuracies[i, :], label=i)
                
        # CIFAR only has 50 tasks
        #ax1.plot(all_accuracies[3, 0:50], label=3)

        colormap = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        for i, j in enumerate(ax1.lines):
            j.set_color(colormap[i])
            j.set_label(all_labels[i])

        ax1.legend(loc='lower left', prop={'size': 6})
        ax1.set_title('Accuracies of SI-XdG on mix of MNIST, Fashion-MNIST dataset')
        plt.xlabel('Number of tasks')
        plt.ylabel('Accuracy')
        plt.savefig(png_OutPath + 'MixTraing.png', format='png')
        plt.show()

    elif type == 'ReviewTraining':
        print('Ploting MNIST and Fashion-MNIST ReviewTraining results')
        
        savedir_mnist = 'savedir/mnist/'
        savedir_fashion = 'savedir/fashion_mnist/'
        savedir_mix = 'savedir/mix/' + type + '/'
        #savedir_cifar = 'savedir/cifar/'

        png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/ReviewTraining/'

        all_accuracies = np.zeros((20, 100))

        # 7 results in total
        mnist = all_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        fashion_mnist = all_accuracies[1, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
        review_10_0 = all_accuracies[2, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_0')
        review_10_10 = all_accuracies[3, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_10') 
        review_10_20 = all_accuracies[4, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_20') 
        #cifar = all_accuracies[3, 0:50] = get_accuracy(savedir_cifar, 'cifar_SI_XdG_50')

        all_labels = ['MNIST', 
                      'Fashion MNIST', 
                      'No review',
                      'Review every 10 tasks',
                      'Review every 20 tasks']  

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        for i in range(0, 5):
            ax1.plot(all_accuracies[i, :], label=i)
                
        # CIFAR only has 50 tasks
        #ax1.plot(all_accuracies[3, 0:50], label=3)

        colormap = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        for i, j in enumerate(ax1.lines):
            j.set_color(colormap[i])
            j.set_label(all_labels[i])

        ax1.legend(loc='lower left', prop={'size': 6})
        ax1.set_title('Accuracies of SI-XdG on MNIST dataset with review')
        plt.xlabel('Number of tasks')
        plt.ylabel('Accuracy')
        plt.savefig(png_OutPath + 'ReviewTraing.png', format='png')
        plt.show()

    else:
        print("Unrecognized task type, please use MixTraining or ReviewTraining")

def plot_cifar():

    print('Ploting cifar results')
    savedir = 'savedir/cifar/'

    png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/'

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

    colormap = ['blue', 'green', 'red', 'cyan']
    for i, j in enumerate(ax1.lines):
        j.set_color(colormap[i])
        j.set_label(all_labels[i])

    ax1.legend(loc=1)
    ax1.set_title('Accuracies of four combinations on CIFAR dataset')
    plt.xticks([0, 5, 10, 15, 20])
    plt.xlabel('Number of tasks')
    plt.ylabel('Accuracy')
    plt.savefig(png_OutPath + 'cifar_results.png', format='png')
    plt.show()




print('Generating plots')
#plot_mnist()
#plot_cifar()
#plot_fashion_mnist()
plot_mix('ReviewTraining')

    