import numpy as np

def second_order(N):
    
    len_ = len(N)
    num = np.array([N[t] * N[t+1] for t in range(len_-1)])
    denum = np.array([N[t] * N[t] for t in range(len_-1)])

    return np.mean(num) / np.mean(denum)