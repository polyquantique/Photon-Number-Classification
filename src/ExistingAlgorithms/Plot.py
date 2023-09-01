import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

def plot_1D(X_l):

    plt.figure(figsize=(10,5))
    plt.hist(X_l, bins=5000)
    plt.xlabel('Feature Space')
    plt.ylabel('Counts')
    plt.show()


def plot_traces(X):

    plt.figure()
    [plt.plot(x, alpha = 0.01) for x in X[::20]]
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Voltage (a.u.)')
    plt.show()