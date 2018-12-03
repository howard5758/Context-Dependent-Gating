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


# helper function that actually makes the plot
def make_plots(withCIFAR, accuracies, all_labels, outputPath, graph_name, graph_title, showGraph):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # CIFAR only has 50 tasks
    if withCIFAR:
        for i in range(0, 2):
            ax.plot(accuracies[i, :], label=i)
        ax.plot(accuracies[2, 0:50], label=2)
    else:
        for i in range(0, accuracies.size):
            ax.plot(accuracies[i, :], label=i)

    # for now 7 seven basic colors are provided
    colormap = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    
    for i, j in enumerate(ax.lines):
        j.set_color(colormap[i])
        j.set_label(all_labels[i])

    ax.legend(loc='lower left', prop={'size': 6})
    ax.set_title(graph_title)
    plt.xlabel('Number of tasks')
    plt.ylabel('Accuracy')
    plt.savefig(outputPath + graph_name, format='png')
    if showGraph:
        plt.show()



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


def plot_mix(type):
    savedir_mnist = 'savedir/mnist/'
    savedir_fashion = 'savedir/fashion_mnist/'
    savedir_cifar = 'savedir/cifar/'
#********************************************************************************************************************************************
#*******************************                                                                      ***************************************
#******************************* General Graph 1 - Comparisons between MNIST, Fashion-MNIST and CIFAR ***************************************
#*******************************                                                                      ***************************************
#********************************************************************************************************************************************
    if type == 'General':
        print('Ploting MNIST, Fashion-MNIST and CIFAR Training results')
        general_png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/'

        general_accuracies = np.zeros((3, 100))
        mnist = general_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        fashion_mnist = general_accuracies[1, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
        cifar = general_accuracies[2, 0:50] = get_accuracy(savedir_cifar, 'cifar_SI_XdG_50')

        general_labels = ['MNIST',
                         'Fashion-MNIST',
                         'CIFAR']

        CIFARinGeneral = True
        general_name = 'Fashion-MNIST.png'
        general_title = 'Accuracies over 100 tasks with MNIST, Fashion-MNIST and CIFAR datasets'
        showGeneralGraph = False

        make_plots(CIFARinGeneral, general_accuracies, general_labels, general_png_OutPath, general_name, general_title, showGeneralGraph)


#********************************************************************************************************************************************
#********************************************************                     ***************************************************************
#******************************************************** Mix Training Graphs ***************************************************************
#********************************************************                     ***************************************************************
#********************************************************************************************************************************************
    elif type == 'MixTraining':
        print('Ploting MNIST and Fashion-MNIST MixTraining results')
        savedir_mix = 'savedir/mix/' + type + '/'
        mix_png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/MixTraining/'

        #********************************************** Graph 1 - the influence of mixtraining **********************************************
        mix1_accuracies = np.zeros((20, 100))

        mnist = mix1_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        fashion_mnist = mix1_accuracies[1, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
        mix_10_100_0 = mix1_accuracies[2, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_100_0')
        mix_10_100_50 = mix1_accuracies[3, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_100_50')
        # We need to a couple more experiments in this category
        
        mix1_labels = ['Pure MNIST', 
                      'Pure Fashion MNIST', 
                      'First 10 tasks trained with 100% MNIST, 0% Fashion-MNIST, later 90 tasks trained with 0% MNIST, 100% Fashion-MNIST', 
                      'First 10 tasks trained with 100% MNIST, 0% Fashion-MNIST, later 90 tasks trained with 50% MNIST, 50% Fashion-MNIST']

        CIFAR_Mix1 = False
        mix1_name = 'MixTraining_1.png'
        mix1_title = 'Accuracies over 100 tasks with both mixed and unmixed datasets'
        showMixGraph1 = False

        make_plots(CIFAR_Mix1, mix1_accuracies, mix1_labels, mix_png_OutPath, mix1_name, mix1_title, showMixGraph1)

        #***************************** Graph 2 - comparisons between mix data in all training tasks ***************************************

        mix2_accuracies = np.zeros((20, 100))

        # 6 results in total
        mnist = mix2_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        fashion_mnist = mix2_accuracies[1, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
        mix_10_100_0 = mix2_accuracies[2, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_100_0')
        mix_10_80_20 = mix2_accuracies[3, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_80_20')
        mix_10_60_40 = mix2_accuracies[4, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_60_40')
        mix_10_40_60 = mix2_accuracies[5, :] = get_accuracy(savedir_mix, 'mix_SI_XdG_10_40_60')
        # We need to run more experiments here

        all_labels = ['Pure MNIST', 
                      'Pure Fashion MNIST', 
                      'First 10 tasks trained with 100% MNIST, 0% Fashion-MNIST, later 90 tasks trained with 0% MNIST, 100% Fashion-MNIST', 
                      'First 10 tasks trained with 80% MNIST, 20% Fashion-MNIST, later 90 tasks trained with 20% MNIST, 80% Fashion-MNIST',
                      'First 10 tasks trained with 60% MNIST, 40% Fashion-MNIST, later 90 tasks trained with 40% MNIST, 60% Fashion-MNIST' 
                      'First 10 tasks trained with 40% MNIST, 60% Fashion-MNIST, later 90 tasks trained with 60% MNIST, 40% Fashion-MNIST'] 

        CIFAR_Mix2 = False
        mix2_name = 'MixTraining_2.png'
        mix2_title = 'Accuracies over 100 tasks with different kinds of mixed datasets'
        showMixGraph2 = False

        make_plots(CIFAR_Mix2, mix2_accuracies, mix2_labels, mix_png_OutPath, mix2_name, mix2_title, showMixGraph2) 

        

#********************************************************************************************************************************************
#***************************************************                        *****************************************************************
#*************************************************** Review Training Graphs *****************************************************************
#***************************************************                        *****************************************************************
#********************************************************************************************************************************************
    elif type == 'ReviewTraining':
        print('Ploting MNIST and Fashion-MNIST ReviewTraining results')
        savedir_mix = 'savedir/mix/' + type + '/'
        review_png_OutPath = '/Users/zhuokaizhao/Desktop/UChicago/Autumn2018/CMSC_35200_Deep_Learning_System/Project/Plots/ReviewTraining/'

        #***************************** Graph 1 - the relationships between different review frequence ***************************************
        review1_accuracies = np.zeros((7, 100))
        
        mnist = review1_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        fashion_mnist = review1_accuracies[1, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
        review_10_0 = review1_accuracies[1, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_0')
        review_10_2 = review1_accuracies[2, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_2')
        review_10_5 = review1_accuracies[5, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_5')
        review_10_10 = review1_accuracies[6, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_10') 
        review_10_20 = review1_accuracies[7, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_20')

        review1_labels = ['SI+XdG - Pure MNIST',
                          'SI+XdG - Pure Fashion-MNIST',
                          'SI+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, no review',
                          'SI+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, review MNIST every 2 tasks',
                          'SI+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, review MNIST every 5 tasks',
                          'SI+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, review MNIST every 10 tasks',
                          'SI+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, review MNIST every 20 tasks',]

        CIFAR_Review1 = False
        review1_name = 'ReviewTraining_1.png'
        review1_title = 'Accuracies over 100 tasks with different review frequencies'
        showReviewGraph1 = False

        make_plots(CIFAR_Review1, review1_accuracies, review1_labels, review_png_OutPath, review1_name, review1_title, showReviewGraph1)


        #***************************** Graph 2 - the relationships between different numbers of first tasks ********************************
        review2_accuracies = np.zeros((7, 100))
        
        mnist = review2_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        fashion_mnist = review2_accuracies[1, :] = get_accuracy(savedir_fashion, 'fashion_mnist_SI_XdG')
        review_10_2 = review2_accuracies[2, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_2')
        review_20_2 = review2_accuracies[3, :] = get_accuracy(savedir_mix, 'review_SI_XdG_20_2')
        review_40_2 = review2_accuracies[4, :] = get_accuracy(savedir_mix, 'review_SI_XdG_40_2')

        review2_labels = ['SI+XdG - Pure MNIST',
                          'SI+XdG - Pure Fashion-MNIST',
                          'SI+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, review MNIST every 2 tasks',
                          'SI+XdG - First 20 task train with MNIST, later train with Fashion-MNIST, review MNIST every 2 tasks',
                          'SI+XdG - First 40 task train with MNIST, later train with Fashion-MNIST, review MNIST every 2 tasks',]

        CIFAR_Review2 = False
        review2_name = 'ReviewTraining_3_DiffNumFirstTasks.png'
        review2_title = 'Accuracies over 100 tasks with different initial tasks that have been trained solely with MNIST'
        showReviewGraph2 = False

        make_plots(CIFAR_Review2, review2_accuracies, review2_labels, review_png_OutPath, review2_name, review2_title, showReviewGraph2)
        
        
        #***************************** Graph 3 - the comparisons between different methods (SI vs EWC) ***********************************
        review3_accuracies = np.zeros((3, 100))
        
        mnist = review3_accuracies[0, :] = get_accuracy(savedir_mnist, 'mnist_SI_XdG')
        review_10_2 = review3_accuracies[1, :] = get_accuracy(savedir_mix, 'review_SI_XdG_10_2')
        review_10_2_EWC = review3_accuracies[2, :] = get_accuracy(savedir_mix, 'review_EWC_XdG_10_2')
        
        review3_labels = ['SI+XdG - Pure MNIST',
                          'SI+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, review MNIST every 2 tasks',
                          'EWC+XdG - First 10 task train with MNIST, later train with Fashion-MNIST, review MNIST every 2 tasks']  

        CIFAR_Review3 = False
        review3_name = 'ReviewTraining_4_SIvsEWC.png'
        review3_title = 'Accuracies over 100 tasks with different initial tasks that have been trained solely with MNIST'
        showReviewGraph3 = False

        make_plots(CIFAR_Review3, review3_accuracies, review3_labels, review_png_OutPath, review3_name, review3_title, showReviewGraph3)
        

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
plot_mix('General')
plot_mix('MixTraining')
plot_mix('ReviewTraining')

    