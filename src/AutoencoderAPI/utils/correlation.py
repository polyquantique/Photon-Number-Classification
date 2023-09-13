import numpy as np

def second_order(N):
    
    mean = np.mean(N)
    variance = np.var(N)

    return variance/mean**2 + 1 - 1/mean#np.mean(num) / np.mean(denum)