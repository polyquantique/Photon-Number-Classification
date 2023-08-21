import torch


def train(config, network, X_train, optimizer, criterion, cluster_label):
        """
        # train_epoch

        train_epoch(config, network, X_train, optimizer, criterion)

        Training process executed for every epoch. The actions consists of setting the gradients to zero, 
        making predictions for the batch, computing the loss and its gradient and updating the weights and biases.

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary
        - network : Pytorch sequential : 
                - Autoencoder neural network that is trained to reproduce its input signal.
        - X_train : torch.tensor
                - Input samples used to train the autoencoder.
        - optimizer : Pytorch optimizer
                - Optimizer used for training.
        - criterion : Pytorch criterion
                - Criterion used for training.
        
        Returns
        -------
        - Average loss : float
                - Average loss of the training process (loss of one epoch).
        """
        cumu_loss = 0
        _ = None
        network.train()
        for index, input_ in enumerate(X_train):#tqdm(enumerate(X_train),total=X_train.size(0) , desc='Train Triplet'):
            
            # Zero gradient
            optimizer.zero_grad()
            # Forward
            output_ = network(input_)
            # Criterion
            current_label = cluster_label[index]
            negative_index = torch.where(cluster_label != current_label)[0]
            rand_index = torch.randint(negative_index.size(0), (1,))
            negative = X_train[negative_index[rand_index]]
            loss = criterion.forward(output_, input_, _, negative.view(1,-1), config['train']['alpha'])
            # Backward
            loss.backward()
            optimizer.step()
            # Loss
            cumu_loss += loss.item()

        return cumu_loss / len(X_train)