import torch
import random


def validation(alpha, 
               network, 
               X, 
               criterion, 
               cluster_means,
               self,
               cluster_label, 
               store=False):
    """
    Validation or testing.
    This action consists of a forward pass of the network using the desired samples.
    In this step the intermediate results can be stored in a `results` dictionary.
    The results consists of the input, the encoder output and the decoder output.


    Parameters
    ----------
    alpha : float 
        Weight coefficient of the triplet + MSE loss function 
    network : Pytorch sequential : 
        Autoencoder neural network that is trained to reproduce its input signal.
    X : torch.tensor
        Input samples used to validate or test the autoencoder.
    criterion : Pytorch criterion
        Criterion used for training.
    cluster_label : torch.tensor
        Cluster label of every element in X.
    store : bool
        If `True` the intermediate results are stored in the `results` dictionary.
    
    Returns
    -------
    store = `True` : 
        Average loss : float
            Average loss of the training process (loss of one epoch).
        results : dict
            Dictionary containing the intermediate results of the process 
            (input, encoder output and decoder output)
    store = `False` : 
        Average loss : float
            Average loss of the training process (loss of one epoch).
    """
    arr = torch.arange(cluster_means.size(0))
    negative_low = torch.tensor([torch.roll(arr, random.choice([-1,1]))[index] for index in cluster_label])
        #positive_low = torch.tensor([cluster_means[index] for index in cluster_label])

    positive_low = cluster_means[cluster_label].to(self.device)
    negative_low = cluster_means[negative_low].to(self.device)
    cumu_loss = 0
    _ = None

    if store:
        results = {'encode' : [],
                    'input'  :  [],
                    'decode' : []
                    }

    network.eval()
    with torch.no_grad():
        for index, (positive_low_, negative_low_, input_high) in enumerate(zip(positive_low, negative_low, X)):

            if store:
                output_low = network(input_high, encoding=True)
                output_high = network(output_low, decoding =True)

                results['encode'].append(output_low.cpu().clone().view(-1).numpy())

                if index < 2:
                    results['input'].append(input_high.cpu().clone().view(-1).numpy())
                    results['decode'].append(output_high.cpu().clone().view(-1).numpy())

            else:
                output_low = network(input_high, encoding=True)
                output_high = network(output_low, decoding =True)
            
            #current_label = cluster_label[index]
            #negative_index = torch.where(cluster_label != current_label)[0]
            #rand_index = torch.randint(negative_index.size(0), (1,))
            #negative = X[negative_index[rand_index]]
            loss = criterion.forward(output_high, input_high, output_low, (positive_low_.view(1,-1) , negative_low_.view(1,-1)), alpha)
            cumu_loss += loss.item()

    if store:
        return cumu_loss / len(X), results
            
    return cumu_loss / len(X)

        