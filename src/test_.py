import numpy as np
from os import listdir
# pytest -k Test_Autoencoders -q --verbose 
from AutoencoderAPI.autoencoderGaussianMixture import autoencoder_gaussianMixture
from AutoencoderAPI.autoencoderBayesianGaussianMixture import autoencoder_bayesianGaussianMixture

class Test_Autoencoders:
    """
    Autoencoder gaussian mixture initialized using kernel density estimation.
    """
    def test_autoencoder_gaussianMixture(self):

        signal_size = 250
        path_test = r'C:\Users\dalbe\Documents\Msc Engineering Physics\Single-Photon-Detection\src\Datasets\NIST (250) test/'
        X_test = np.concatenate([np.fromfile(f"{path_test}{fileName}",dtype=np.float16).reshape(-1,signal_size) for fileName in listdir(path_test)])
        X_test = X_test.astype("double")
        
        model_path = r"C:\Users\dalbe\Documents\Msc Engineering Physics\Single-Photon-Detection\src\Test\test NIST (250)\fold 0"
        ADG = autoencoder_gaussianMixture(model_path)
        ADG.fit(np.copy(X_test),
                plot_cluster=False,
                plot_traces=False,
                plot_traces_average=False,
                bw_cst=[0.008],#[0.008],
                filter_input=True,
                filter_threshold = 0.0004,
                flip=True)
        
        assert ADG.network is not None
        assert ADG.config_load is not None   
        assert ADG.predict is not None
        assert ADG.labels is not None   
        

         
    
    def test_autoencoder_bayesianGaussianMixture(self):

        signal_size = 250
        path_test = r'C:\Users\dalbe\Documents\Msc Engineering Physics\Single-Photon-Detection\src\Datasets\NIST (250) test/'
        X_test = np.concatenate([np.fromfile(f"{path_test}{fileName}",dtype=np.float16).reshape(-1,signal_size) for fileName in listdir(path_test)])
        X_test = X_test.astype("double")
        
        model_path = r"C:\Users\dalbe\Documents\Msc Engineering Physics\Single-Photon-Detection\src\Test\test NIST (250)\fold 0"
        ABG = autoencoder_bayesianGaussianMixture(model_path)
        ABG.fit(np.copy(X_test),  
                plot_cluster = False, 
                plot_traces = False,
                plot_traces_average = False, 
                cluster_max = 30, 
                flip = True,
                filter_input = True,
                filter_threshold = 0.0005)
        
        assert ABG.network is not None
        assert ABG.config_load is not None   
        assert ABG.predict is not None
        assert ABG.labels is not None  

