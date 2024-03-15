import torch
from tqdm.notebook import tqdm
import random


def train(config, 
          network, 
          X, 
          optimizer, 
          criterion, 
          cluster_label, 
          cluster_means, 
          self):
        """
        Training process executed for every epoch. The actions consists of setting the gradients to zero, 
        making predictions for the batch, computing the loss and its gradient and updating the weights and biases.

        The triplet loss negative is defined using a random values from another random cluster.

        Parameters
        ----------
        config : dict
                Dictionary containing the experiment parameters. 
        network : Pytorch sequential : 
                Autoencoder neural network that is trained to reproduce its input signal.
        X : torch.tensor
                Input samples used to train the autoencoder.
        optimizer : Pytorch optimizer
                Optimizer used for training.
        criterion : Pytorch criterion
                Criterion used for training.
        cluster_label : torch.tensor
                Cluster labels for every element in X.

        Returns
        -------
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
        network.train()
        #for index, input_ in tqdm(enumerate(X), total=X.size(0)):#tqdm(enumerate(X_train),total=X_train.size(0) , desc='Train Triplet'):
        for positive_low_, negative_low_, input_high in tqdm(zip(positive_low, negative_low, X), total=X.size(0)):

                # Zero gradient
                optimizer.zero_grad()
                # Forward
                output_low = network(input_high, encoding=True)
                output_high = network(output_low, decoding =True)
                # Criterion
                #current_label = cluster_label[index]
                
                #negative_index = torch.where(cluster_label != current_label)[0]
                #rand_index = torch.randint(negative_index.size(0), (1,))
                #negative = X[negative_index[rand_index]]

                loss = criterion.forward(output_high, input_high, output_low, (positive_low_.view(1,-1) , negative_low_.view(1,-1)), config['train']['alpha'])
                # Backward
                loss.backward()
                optimizer.step()
                # Loss
                cumu_loss += loss.item()

        return cumu_loss / len(X)
