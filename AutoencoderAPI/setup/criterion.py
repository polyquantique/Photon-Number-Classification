from torch import nn
from random import choice, sample
from .custom_loss import pytorch_kmeans_silhouette_loss, sklearn_kernelDensity_loss, sklearn_kmeans_silhouette_loss


class build_criterion:

    def __init__(self, config):
        """
        # criterion

        criterion(config)

        Definition of the criterion used in the training process.
        Multiple loss function are evaluated and each require specific inputs.
        The initialization takes care of the different cases to remove the selection 
        process of the training loops.

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                See the `main` class for more details on the config dictionary

        Returns
        -------
        - None
        """
        def generic(output_, input_, X, network, list_):
            return self.criterion(output_, input_)

        def triplet(output_, input_, X, network, list_):
            negative = X[choice(list_)].view(1,-1)
            return self.criterion(input_, output_, negative)

        def custom(output_, input_, X, network, list_):
            X_sub = X[sample(list_, 10_000)]
            crit = self.criterion
            return crit.forward(output_, input_, network, X_sub)

        criterion_dict = {
            "CrossEntropy"       : (nn.CrossEntropyLoss() , generic),
            "L1Loss"             : (nn.L1Loss() , generic),
            "MSELoss"            : (nn.MSELoss() , generic),
            "NLLLoss"            : (nn.NLLLoss() , generic),
            "HingeEmbeddingLoss" : (nn.HingeEmbeddingLoss() , generic),
            "MarginRankingLoss"  : (nn.MarginRankingLoss() , generic),
            "KLDivLoss"          : (nn.KLDivLoss() , generic),
            "TripletMarginLoss"  : (nn.TripletMarginLoss() , triplet),
            "pytorch_kmeans_silhouette_loss"  : (pytorch_kmeans_silhouette_loss() , custom),
            "sklearn_kernelDensity_loss"      : (sklearn_kernelDensity_loss() , custom),
            "sklearn_kmeans_silhouette_loss"  : (sklearn_kmeans_silhouette_loss() , custom)
        }

        try:
            self.criterion, crit_type = criterion_dict[config['train']['criterion']]
        except Exception as ex:
            criterion = criterion_dict["MSELoss"]
            warnings.warn("No criterion was defined int the configuration dict (was set MSELoss)")

        self.lossFunction = crit_type
        

    def forward(self, output_, input_, X, network, list_):
        """
        # forward

        forward(output_, input_, X=None, network=None, list_=None)

        Use the initialized criterion to output the loss of a specific experiment.

        Parameters
        ----------
        - output_ : 
        - input_ :
        - X :
        - network :
        - list_ :

        Returns
        -------
        - loss : 
        """   
        loss = self.lossFunction(output_, input_, X, network, list_)

        return loss