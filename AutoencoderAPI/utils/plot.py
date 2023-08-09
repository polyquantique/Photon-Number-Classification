import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def plot_hist(X, xlabel=None, ylabel=None, bins=700):

    plt.figure()
    plt.hist(X, bins=bins)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()