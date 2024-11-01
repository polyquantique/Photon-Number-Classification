from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
from adjustText import adjust_text
import numpy as np
import os

STYLE = "seaborn-v0_8"


def plot_traces(X_high : np.array):
    """

    Plot signals.

    Parameters
    ----------
    X_high : ndarray

    Returns
    -------
    None
    
    """
    with plt.style.context(STYLE):
        plt.figure(figsize=(6,3), dpi=100)
        plt.plot(X_high[::10].T, alpha = 0.01)
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Voltage (a.u.)')
        plt.show()


def norm(X : np.array):
    """

    Normalize an array

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    norm : ndarray
    
    """
    return (X - X.min()) / (X.max() - X.min())


def save_results(gm, 
                 name_method : str = 'Max', 
                 path : str = r'src/Results'):
    
    if not os.path.isdir(path):
        try:
            os.mkdir(f'{path}/Confidence')
            os.mkdir(f'{path}/Mean Clusters')
            os.mkdir(f'{path}/Trustworthiness Euclidian')
            os.mkdir(f'{path}/g2')
        except OSError as error:
            print(error)
    
    if not gm.confidence_1D is None:
        np.save(f'{path}/Confidence/{name_method}.npy', gm.confidence_1D) 
    if not gm.confidence_2D is None:
        np.save(f'{path}/Confidence/{name_method}.npy', gm.confidence_2D)

    if not gm.cluster_means is None:
        np.save(f'{path}/Mean Clusters/{name_method}.npy', gm.cluster_means) 

    if not gm.trustworthiness_eucl is None:
        np.save(f'{path}/Trustworthiness Euclidian/{name_method}.npy', gm.trustworthiness_eucl) 
    
    if not gm.g2 is None:
        np.save(f'{path}/g2/{name_method}_g2.npy', gm.g2) 
        np.save(f'{path}/g2/{name_method}_db.npy', gm.unique_db)


def get_means(name_method : str = 'Max', 
              path : str = r'src/Results TES'):
    try:
        means = np.load(f'{path}/Mean Clusters/{name_method}.npy') 
    except:
        means = None
    return means



def plot_results(config,
                 pad = 0.5,
                 xlim = (0,30),
                 ylim = (0.7,1.01),
                 yscale = 'linear',
                 path_results = r'src/Results TES (Uniform)/'):

    texts = []
    objects = []
    cmap = mpl.colormaps['tab20b']
    colors = cmap(np.linspace(0, 1, len(config.keys())))

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize = (6,3))
        for method, color in zip(config, colors):

            method_dict = config[method]
            i, f = method_dict['n_photons']
            result = np.load(f'{path_results}/Confidence/{method}', allow_pickle=True)
            confidence = result[:f]
            photon_number = np.arange(i,f+i)

            objects.append(ax.plot(photon_number, confidence, 
                                    c = color, 
                                    alpha=1, 
                                    linewidth = 1,
                                    linestyle = method_dict['line']))

            texts.append(ax.text(photon_number[-1]+pad, 
                                    confidence[-1], 
                                    method_dict['Name'], 
                                    color = color, 
                                    fontsize=10, 
                                    weight="bold", 
                                    va = "center"))
            
        adjust_text(texts = texts,
                    ax = ax, 
                    expand=(1.05, 1.2),
                    only_move = {"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    ensure_inside_axes = False) 
        plt.ylabel('Confidence')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.yscale(yscale)
        plt.xlabel('Photon number')
        plt.savefig(f'{path_results}Confidence.pdf', format='pdf', bbox_inches='tight')
        plt.show()




def plot_trust(config,
               path_results = r'src/Results TES (Uniform)/'):

    trust1 = []
    trust2 = []
    trust3 = []
    name_method = []

    for index_method, method in enumerate(config):

            method_dict = config[method]
            result = np.load(f'{path_results}/Trustworthiness Euclidian/{method}')
            print(method)
            trust1.append(result[-1])
            trust2.append(result[-2])
            trust3.append(result[-3])
            name_method.append(method_dict['Name'])

    y_pos = np.arange(4*len(trust1))

    with plt.style.context("seaborn-v0_8"):
        plt.figure(figsize = (6,10))
        ax = plt.gca()
        hbars1 = ax.barh(y_pos[0::4], trust1, align='center', alpha=0.7)
        hbars2 = ax.barh(y_pos[1::4], trust2, align='center', alpha=0.7)
        hbars3 = ax.barh(y_pos[2::4], trust3, align='center', alpha=0.7)
        ax.set_yticks(y_pos[1::4], labels=name_method)
        ax.bar_label(hbars1, fmt='%.3f', padding=3)
        ax.bar_label(hbars2, fmt='%.3f', padding=3)
        ax.bar_label(hbars3, fmt='%.3f', padding=3)
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.set_xlabel('Trustworthiness Euclidean')
        ax.set_xlim(0.89,1.01)
        plt.savefig(f'{path_results}Trustworthiness.pdf', format='pdf', bbox_inches='tight')
        plt.show()