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
        plt.figure()
        [plt.plot(x, alpha = 0.01) for x in X[::20]]
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Voltage (a.u.)')
        #plt.xticks([])
        #plt.yticks([])
        plt.show()