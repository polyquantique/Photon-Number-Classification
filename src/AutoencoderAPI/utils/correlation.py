import numpy as np

def second_order(N):
    """
    Second order correlation considering an array of photon number.

    Parameters
    ----------
    N : numpy.array
        Array containing all photon numbers
  
    Returns
    -------
    g2 : float
        Second order correlation of the photon source.
    """
    mean = np.mean(N)
    variance = np.var(N)

    return variance/mean**2 + 1 - 1/mean