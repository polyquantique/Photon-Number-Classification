import numpy as np
import os


def dimension_reduction(X_train, X_test, path, function, **param):
    """
    # reduction(X,function,**param)

    
    
    -X (array-like, sparse matrix) : Inputs of the function that needs to be computed.
    -function (function) : Function that is executed.
    -param (**kwargs) : Multiple keyword arguments to define the parameters of `function`.
    
    """
    method = function(**param)
    X_i = X_test
    file_name = f"{path}/{function}_{param}.npy"

    if os.path.isfile(file_name):
        X_l = np.load(file_name)
    else:
        try:
            trained = method.fit(X_train)
            X_l = trained.transform(X_test)
            #X_r = trained.inverse_transform(X_l)
            X_i = X_test
        except Exception as ex:
            print(ex)
            X_l = method.fit_transform(X_i)
            #X_r = np.array([None])

        X_l = (X_l - np.min(X_l)) / (np.max(X_l) - np.min(X_l))
        np.save(file_name, X_l)

    return X_i, X_l #X_r, 