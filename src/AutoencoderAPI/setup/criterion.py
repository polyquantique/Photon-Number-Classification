from torch import nn
from random import sample

import warnings
from .custom_loss import pytorch_kmeans_silhouette_loss, sklearn_kernelDensity_loss, sklearn_kmeans_silhouette_loss, triplet_MSE

class build_criterion:
    """
    Definition of the criterion used in the training process.
    Multiple loss function are evaluated and each require specific inputs.
    The initialization takes care of the different cases to remove the selection 
    process of the training loops.

    Parameters
    ----------
    config : dict
        Dictionary containing the experiment parameters. 

    Returns
    -------
    None
    """
    def __init__(self, config):
        
        def generic(output_, input_, X, network, list_):
            return self.criterion(output_, input_)

        def triplet(output_, input_, X, negative, param2):
            return self.criterion(input_, output_, negative)

        def custom_with_sample(output_, input_, X, network, list_):
            X_sub = X[sample(list_, 10_000)]
            crit = self.criterion
            return crit.forward(output_, input_, network, X_sub)
        
        def custom_without_sample(output_, input_, X, negative, param2):
            crit = self.criterion
            return crit.forward(input_, output_, negative, param2)

        criterion_dict = {
            "CrossEntropy"       : (nn.CrossEntropyLoss() , generic),
            "L1Loss"             : (nn.L1Loss() , generic),
            "MSELoss"            : (nn.MSELoss() , generic),
            #"NLLLoss"            : (nn.NLLLoss() , generic),
            "HingeEmbeddingLoss" : (nn.HingeEmbeddingLoss() , generic),
            #"MarginRankingLoss"  : (nn.MarginRankingLoss() , generic),
            "KLDivLoss"          : (nn.KLDivLoss() , generic),
            "TripletMarginLoss"  : (nn.TripletMarginLoss() , triplet),
            "TripletMSE"         : (triplet_MSE() , custom_without_sample),
            "pytorch_kmeans_silhouette_loss"  : (pytorch_kmeans_silhouette_loss() , custom_with_sample),
            "sklearn_kernelDensity_loss"      : (sklearn_kernelDensity_loss() , custom_with_sample),
            "sklearn_kmeans_silhouette_loss"  : (sklearn_kmeans_silhouette_loss() , custom_with_sample)
        }

        try:
            self.criterion, crit_type = criterion_dict[config['train']['criterion']]
        except Exception as ex:
            self.criterion = criterion_dict["MSELoss"]
            warnings.warn("No criterion was defined int the configuration dict (was set MSELoss)")

        self.lossFunction = crit_type
        

    def forward(self, output_, input_, X, param1, param2):
        """
        Use the initialized criterion to output the loss of a specific experiment.

        Parameters
        ----------
        output_ : torch.tensor
            Neural network output (reconstruction for autoencoder).
        input_ : torch.tensor
            Neural network input.
        X : torch.tensor
            Dataset 
        param1 : Any
            Depend on criterion type
        param2 : Any
            Depend on criterion type
        
        Returns
        -------
        loss : torch.tensor
            Loss score 
        """   
        loss = self.lossFunction(output_, input_, X, param1, param2)

        return loss