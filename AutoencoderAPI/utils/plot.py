import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from .files import open_object
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-pastel')

# Class specific import
def plot_hist(X, xlabel=None, ylabel=None, bins=700):

    plt.figure()
    plt.hist(X, bins=bins)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


# Class specific import
from .metrics import silhouette_kmean

def load_run_results(file_name):
    """
    # load_run_results

    load_run_results(file_name)

    Load a 'run' folder to plot the dimensionality reduction output, the losses, an input compared 
    to the autoencoder output and the Silhouette score of K-means clustering for multiple cluster numbers. 

    Parameters
    ----------
    - file_name : str
            - Name of the file or path to file inside the `Autoencoder Log` folder.
    
    Returns
    -------
    - None
    """
    path = f"Autoencoder Log/{file_name}"
    
    for index_fold, fold in enumerate(listdir(path)):

        fig, axs = plt.subplots(2,2,figsize=(15,10))

        results = open_object(f"{path}/{fold}/results.bin")

        scores, optimal_cluster, optimal_score, clusters = silhouette_kmean(results['encode'], 40)
        print(f"Optimal number of clusters : {optimal_cluster}")
        
        bins = np.linspace(min(results['encode']), max(results['encode']), 1_000).reshape(-1)

        for index_cluster, cluster in enumerate(clusters):
            axs[0,0].hist(cluster , bins, alpha = 0.5, label=f"{index_cluster}")
        axs[0,0].set_xlabel("feature")
        axs[0,0].set_ylabel("counts")
        axs[0,0].legend(ncol=3)
            
        axs[1,0].plot(range(2, len(scores)+2), scores, label="Approx Silhouette")
        axs[1,0].hlines(optimal_score, 2, len(scores)+2, linestyles='dashed', label="Final Silhouette")
        axs[1,0].set_ylabel("Clustering score")
        axs[1,0].set_xlabel("Number of cluster")
        axs[1,0].legend()

        axs[1,1].plot(results['decode'][1],label=f"Autoencoder output {index_fold}")
        axs[1,1].plot(results['input'][1],label=f"Autoencoder input {index_fold}")
        axs[1,1].set_ylabel("Normalized voltage")
        axs[1,1].set_xlabel("element")
        axs[1,1].legend()

        loss = open_object(f"{path}/{fold}/loss.bin")

        axs[0,1].plot(loss['train_loss'],label=f"Train {index_fold}")
        axs[0,1].plot(loss['validation_loss'],label=f"Validation {index_fold}")
        axs[0,1].hlines(loss['test_loss'], 0, len(loss['validation_loss'])-1, linestyles='dashed',label = f"Test {index_fold}")
        axs[0,1].legend()
        axs[0,1].set_ylabel("loss")
        axs[0,1].set_xlabel("epoch")

    config_file = open_object(f"{path}/{fold}/log.bin")
    print("Activation list : ", config_file['network']['activation_list'])
    print("Layer list : ", config_file['network']['layer_list'])
    print("Optimal Silhouette score : ", optimal_score)


# Class specific import

def load_sweep_results(file_name, parameters : tuple):
    """
    # load_sweep_results

    load_sweep_results(file_name, parameters)


    Parameters
    ----------
    - file_name : str
            - Name of the file or path to file inside the `Autoencoder Log` folder.
    
    Returns
    -------
    - None

    """
    path = f"Autoencoder Log/{file_name}"

    parameter1 = []
    parameter2 = []
    loss_sweep = []
    min_loss = 0#1
    
    for sweep in sorted(listdir(path)):
        loss_cum = 0
        fold_list = sorted(listdir(f"{path}/{sweep}"))
        fold_len = len(fold_list)

        for index, fold in enumerate(fold_list):
            loss = open_object(f"{path}/{sweep}/{fold}/loss.bin")
            loss_cum += loss['Silhouette']#['test_loss'][0]

        config_file = open_object(f"{path}/{sweep}/{fold}/log.bin")
        
        parameter1.append(config_file['train'][parameters[0]])
        parameter2.append(config_file['train'][parameters[1]])
        loss_sweep.append(loss_cum / fold_len)

        if loss_sweep[-1] > min_loss:   #< min_loss:
            min_loss = loss_sweep[-1]
            min_parameter1 = parameter1[-1]
            min_parameter2 = parameter2[-1]

    
    x=np.unique(parameter1)
    y=np.unique(parameter2)
    X,Y = np.meshgrid(x,y)
    print("min : ", min_loss)
    print(f"{parameters[0]} : ", min_parameter1)
    print(f"{parameters[1]} : ", min_parameter2)

    Z= np.array(loss_sweep).reshape(len(y),len(x))#np.rot90(np.array(loss_sweep).reshape(len(y),len(x)))
    plt.xticks(np.arange(len(x)), labels=x)
    plt.yticks(np.arange(len(y)), labels=y)

    #plt.pcolormesh(X,Y,Z)
    plt.imshow(Z)#, norm=colors.LogNorm(), interpolation="bilinear")
    plt.xlabel(parameters[0])
    plt.ylabel(parameters[1])
    plt.colorbar()
    plt.show()