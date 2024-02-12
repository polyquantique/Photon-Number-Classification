import matplotlib.pyplot as plt

def plot_1D(X_l):
    
    with plt.style.context("seaborn-v0_8"):
        plt.figure(figsize=(10,5))
        plt.hist(X_l, bins=5000)
        plt.xlabel('Feature Space')
        plt.ylabel('Counts')
        plt.show()


def plot_traces(X):

    with plt.style.context("seaborn-v0_8"):
        plt.figure(figsize=(10,4), dpi=200)
        [plt.plot(x, alpha = 1) for x in X[::20]]#0.01
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Voltage (a.u.)')
        #plt.xticks([])
        #plt.yticks([])
        plt.show()