import torch


def validation(alpha, network, X, criterion, cluster_label, store=False):
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
    cumu_loss = 0
    _ = None

    if store:
        results = {'encode' : [],
                    'input'  :  [],
                    'decode' : []
                    }

    network.eval()
    with torch.no_grad():
        for index, data in enumerate(X):

            if store:
                encode = network(data, encoding=True)
                decode = network(encode, decoding =True)

                save_encode = torch.clone(encode).numpy()
                results['encode'].append(save_encode[0,0])

                if index < 2:
                    results['input'].append(data.clone().view(-1).numpy())
                    results['decode'].append(decode.clone().view(-1).numpy())

            else:
                decode = network(data)
            
            current_label = cluster_label[index]
            negative_index = torch.where(cluster_label != current_label)[0]
            rand_index = torch.randint(negative_index.size(0), (1,))
            negative = X[negative_index[rand_index]]

            loss = criterion.forward(decode, data, _, negative.view(1,-1), alpha)
            cumu_loss += loss.item()

    if store:
        return cumu_loss / len(X), results
            
    return cumu_loss / len(X)

        