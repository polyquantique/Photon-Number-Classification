import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from os import listdir
from .files import open_object
import warnings

from .clustering.kernelDensity import kernel_density

warnings.filterwarnings("ignore")

def load_run_results(file_name, 
                     bw = (-5, -2, 20), 
                     clustering = False , 
                     print_network = False,
                     print_train = False):
    """
    Load a `run` folder to plot the dimensionality reduction output, the losses, an input compared 
    to the autoencoder output and cluster labelling using kernel density estimation . 

    Parameters
    ----------
    file_name : str
        Name of the file or path to file inside the `Autoencoder Log` folder.
    bw : tuple or numpy.array
        If bw is a tuple, it represents the parameters inside np.logspace(\*bw).
        Otherwise, an array can be used, this represents an array containing all 
        the possible bandwidth used in the kernel density estimation.
    Returns
    -------
    None
    """
    path = f"{file_name}"
    
    with plt.style.context("seaborn-v0_8"):
        for index_fold, fold in enumerate(listdir(path)):

            if any([print_network, print_train]):
                log = open_object(f"{path}/{fold}/log.bin")

            if print_network:
                print(log['network'])

            if print_train:
                print(log['train'])

            fig, axs = plt.subplots(3,1,figsize=(10,10))

            results = open_object(f"{path}/{fold}/results.bin")
            results_encode = np.array(results['encode'])
            mean_ = np.mean(results_encode)
            std_ = np.std(results_encode)

            if clustering:
                kd = kernel_density((results_encode-mean_)/std_, bw)
                clusters = kd.clusters_low
            else:
                clusters = [results_encode]
            
            bins = np.linspace(min(results_encode), max(results_encode), 10_000).flatten()

            for index_cluster, cluster in enumerate(clusters):
                axs[0].hist(cluster.flatten() , bins, alpha = 0.5, label=f"{index_cluster}",histtype='step', fill=True)
            axs[0].set_xlabel("feature")
            axs[0].set_ylabel("counts")
            axs[0].legend(ncol=3)
                
            axs[2].plot(results['input'][1],label=f"Autoencoder input {index_fold}")
            axs[2].plot(results['decode'][1],label=f"Autoencoder output {index_fold}")
            axs[2].set_ylabel("Normalized voltage")
            axs[2].set_xlabel("element")
            axs[2].legend()

            loss = open_object(f"{path}/{fold}/loss.bin")

            axs[1].plot(loss['train_loss'],label=f"Train {index_fold}")
            axs[1].plot(loss['validation_loss'],label=f"Validation {index_fold}")
            axs[1].hlines(loss['test_loss'], 0, len(loss['validation_loss'])-1, linestyles='dashed',label = f"Test {index_fold}")
            axs[1].legend()
            axs[1].set_ylabel("loss")
            axs[1].set_xlabel("epoch")
            plt.show()

def load_run_results_2D(file_name, 
                     bw = (-5, -2, 20), 
                     clustering = False , 
                     print_network = False,
                     print_train = False):
    """
    Load a `run` folder to plot the dimensionality reduction output, the losses, an input compared 
    to the autoencoder output and cluster labelling using kernel density estimation . 

    Parameters
    ----------
    file_name : str
        Name of the file or path to file inside the `Autoencoder Log` folder.
    bw : tuple or numpy.array
        If bw is a tuple, it represents the parameters inside np.logspace(\*bw).
        Otherwise, an array can be used, this represents an array containing all 
        the possible bandwidth used in the kernel density estimation.
    Returns
    -------
    None
    """
    path = f"{file_name}"
    
    with plt.style.context("seaborn-v0_8"):
        for index_fold, fold in enumerate(listdir(path)):

            if any([print_network, print_train]):
                log = open_object(f"{path}/{fold}/log.bin")

            if print_network:
                print(log['network'])

            if print_train:
                print(log['train'])

            fig, axs = plt.subplots(3,1,figsize=(10,10))

            results = open_object(f"{path}/{fold}/results.bin")
            results_encode = np.array(results['encode'])
            mean_ = np.mean(results_encode)
            std_ = np.std(results_encode)

            if clustering:
                kd = kernel_density((results_encode-mean_)/std_, bw)
                clusters = kd.clusters_low
            else:
                clusters = [results_encode]
            
            #bins = np.linspace(min(results_encode), max(results_encode), 10_000).flatten()

            for index_cluster, cluster in enumerate(clusters):
                axs[0].scatter(cluster[:,0], cluster[:,1] , alpha = 0.1, label=f"{index_cluster}", s=1)
            axs[0].set_xlabel("feature")
            axs[0].set_ylabel("counts")
            axs[0].legend(ncol=3)
                
            axs[2].plot(results['input'][1],label=f"Autoencoder input {index_fold}")
            axs[2].plot(results['decode'][1],label=f"Autoencoder output {index_fold}")
            axs[2].set_ylabel("Normalized voltage")
            axs[2].set_xlabel("element")
            axs[2].legend()

            loss = open_object(f"{path}/{fold}/loss.bin")

            axs[1].plot(loss['train_loss'],label=f"Train {index_fold}")
            axs[1].plot(loss['validation_loss'],label=f"Validation {index_fold}")
            axs[1].hlines(loss['test_loss'], 0, len(loss['validation_loss'])-1, linestyles='dashed',label = f"Test {index_fold}")
            axs[1].legend()
            axs[1].set_ylabel("loss")
            axs[1].set_xlabel("epoch")
            plt.show()



def load_sweep_results(file_name, parameters):
    """
    Parameters
    ----------
    - file_name : str
            - Name of the file or path to file inside the `Autoencoder Log` folder.
    
    Returns
    -------
    - None

    """
    path = f"AutoencoderLog/{file_name}"

    files = listdir(path)
    l = len(files)
    parameter1 = np.empty(l)
    parameter2 = np.empty(l)
    loss_sweep = np.empty(l)
    min_loss = 1

    for index, sweep in enumerate(sorted(files)):
        loss_cum = 0
        fold_list = sorted(listdir(f"{path}/{sweep}"))
        fold_len = len(fold_list)

        for fold in fold_list:
            loss = open_object(f"{path}/{sweep}/{fold}/loss.bin")
            loss_cum += min(loss['test_loss'])#['test_loss'][0]

        config_file = open_object(f"{path}/{sweep}/{fold}/log.bin")
        
        parameter1[index] = config_file['train'][parameters[0]]
        parameter2[index] = config_file['train'][parameters[1]]
        loss_sweep[index] = (loss_cum / fold_len)

        if loss_sweep[-1] < min_loss:   
            min_loss = loss_sweep[-1]
            min_parameter1 = parameter1[-1]
            min_parameter2 = parameter2[-1]
            min_file = sweep

    
    x = np.unique(parameter1)
    y = np.unique(parameter2)
    #X,Y = np.meshgrid(x,y)
    
    print(f"min : {min_loss}")
    print(f"file : {min_file}")
    print(f"{parameters[0]} : ", min_parameter1)
    print(f"{parameters[1]} : ", min_parameter2)
    """
    if len(loss_sweep) < len(y)*len(x):
        loss_sweep.extend(np.zeros(len(y)*len(x)-len(loss_sweep)))
    print(loss_sweep)  

    Z= np.array(loss_sweep).reshape(len(x),len(y))#.np.rot90(np.array(loss_sweep).reshape(len(y),len(x)))
    

    #plt.pcolormesh(X,Y,Z)
    plt.imshow(Z, norm=colors.LogNorm())#, interpolation="bilinear")
    plt.xlabel(parameters[0])
    plt.ylabel(parameters[1])
    plt.xticks(np.arange(len(x)), labels=x)
    plt.yticks(np.arange(len(y)), labels=y)
    plt.colorbar()
    plt.show()
    """

    if len(loss_sweep) < len(y)*len(x):
        loss_sweep = np.append(loss_sweep, np.zeros(len(y)*len(x)-len(loss_sweep)))
    print(loss_sweep)  

    Z = np.array(loss_sweep).reshape(len(x), len(y)).T  # Transpose added here

    #plt.pcolormesh(X,Y,Z) , norm=colors.LogNorm()
    plt.imshow(Z)#, extent=[np.min(parameter1), np.max(parameter1), np.min(parameter2), np.max(parameter2)])  # Added extent to show correct axis scales
    plt.xlabel(parameters[0])
    plt.ylabel(parameters[1])
    plt.xticks(np.arange(len(x)), labels=x)
    plt.yticks(np.arange(len(y)), labels=y)
    plt.colorbar()
    plt.show()

