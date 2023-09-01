import numpy as np


def dimension_reduction(X, function, **param):
    """
    # reduction(X,function,**param)

    
    
    -X (array-like, sparse matrix) : Inputs of the function that needs to be computed.
    -function (function) : Function that is executed.
    -param (**kwargs) : Multiple keyword arguments to define the parameters of `function`.
    
    """
    method = function(**param)
    X_i = X[::10]

    try:
        trained = method.fit(X_i)
        X_l = trained.transform(X[::10])
        X_r = trained.inverse_transform(X_l)
        X_i = X[::10]
    except Exception as ex:
        print(ex)
        X_l = method.fit_transform(X_i)
        X_r = np.array([None])


    return X_i, X_r, X_l