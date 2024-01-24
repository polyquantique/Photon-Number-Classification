from tqdm.notebook import tqdm

def train(network, X, optimizer, criterion):
    """
    Training process executed for every epoch. The actions consists of setting the gradients to zero, 
    making predictions for the batch, computing the loss and its gradient and updating the weights and biases.

    Parameters
    ----------
    network : Pytorch sequential : 
        Autoencoder neural network that is trained to reproduce its input signal.
    X : torch.tensor
        Input samples used to train the autoencoder.
    optimizer : Pytorch optimizer
        Optimizer used for training.
    criterion : Pytorch criterion
        Criterion used for training.
    
    Returns
    -------
    Average loss : float
        Average loss of the training process (loss of one epoch).
    """
    cumu_loss = 0
    _ = None
    network.train()
    for input_ in tqdm(X):
        # Zero gradient
        optimizer.zero_grad()
        # Forward
        output_ = network(input_)
        # Criterion
        loss = criterion.forward(output_, input_, _, _, _)
        # Backward
        loss.backward()
        optimizer.step()
        # Loss
        cumu_loss += loss.item()

    return cumu_loss / len(X)