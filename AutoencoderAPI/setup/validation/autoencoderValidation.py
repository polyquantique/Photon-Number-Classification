import torch


def validation_test(config, network, X, criterion, store=False):
    """
    # validation_test

    validation_test(config, network, X, criterion, store=False)

    Validation or testing of the network.
    This action consists of a forward pass of the network using the desired samples.
    In this step the intermediate results can be stored in a `results` dictionary.
    The results consists of the input, the encoder output and the decoder output.


    Parameters
    ----------
    - config : dict
            - Dictionary containing the experiment parameters. 
                See the `autoencoder` class for more details on the config dictionary.
    - network : Pytorch sequential : 
            - Autoencoder neural network that is trained to reproduce its input signal.
    - X : torch.tensor
            - Input samples used to validate or test the autoencoder.
    - criterion : Pytorch criterion
            - Criterion used for training.
    - store : bool
            - If `True` the intermediate results are stored in the `results` dictionary.
    
    Returns
    -------
    - store = `True` : 
        - Average loss : float
            - Average loss of the training process (loss of one epoch).
        - results : dict
            - Dictionary containing the intermediate results of the process 
                (input, encoder output and decoder output)
    - store = `False` : 
        - Average loss : float
            - Average loss of the training process (loss of one epoch).
    """
    cumu_loss = 0
    list_ = range(X.size(0))

    if store:
        results = {'encode' : [],
                    'input'  : [],
                    'decode' : []
        }

    network.eval()
    with torch.no_grad():
        for index, input_ in enumerate(X):

            if store:
                encode = network(input_, encoding=True)
                decode = network(encode, decoding =True)

                save_encode = torch.clone(encode).numpy()
                results['encode'].append(save_encode[0])

                if index < 2:
                    results['input'].append(input_.clone().numpy()[0])
                    results['decode'].append(decode.clone().numpy()[0])

            else:
                decode = network(input_)
                
            loss = criterion.forward(decode, input_, X, network, list_)

            cumu_loss += loss.item()

    if store:
        return cumu_loss / len(X), results
    
    return cumu_loss, len(X)