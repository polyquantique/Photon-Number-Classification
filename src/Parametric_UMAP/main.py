import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import torch.nn.functional as F

from .data import UMAPDataset, MatchDataset
from .modules import get_umap_graph, umap_loss
from .model import default_encoder, default_decoder

import json
import numpy as np
from scipy.optimize import curve_fit
from umap import UMAP

"""
From https://github.com/elyxlz/umap_pytorch/tree/main
"""

def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta = 1.0,
        min_dist=0.1,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        match_nonparametric_umap=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta # weight for reconstruction loss
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):

        if not self.match_nonparametric_umap:
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
            encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0], negative_sample_rate=5)
            self.log("umap_loss", encoder_loss, prog_bar=True)

            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss

        else:
            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)
            self.log("encoder_loss", encoder_loss, prog_bar=True)
            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

class PUMAP():
    def __init__(
        self,
        encoder=None,
        decoder=None,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        random_state=None,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        num_workers=1,
        num_gpus=1,
        match_nonparametric_umap=False,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.dim_input = None
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.match_nonparametric_umap = match_nonparametric_umap
        
    def fit(self, X):

        

        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1, 
            max_epochs=self.epochs
            )
        
        self.dim_input = X.shape[1]


        if self.encoder is None:
            self.encoder = default_encoder(self.dim_input, self.n_components)  
        else: 
            self.encoder
        
        if self.decoder is None or isinstance(self.decoder, nn.Module):
            self.decoder = self.decoder
        elif self.decoder == True:
            self.decoder = default_decoder(self.dim_input, self.n_components)
            

        if not self.match_nonparametric_umap:
            self.model = Model(
                lr = self.lr, 
                encoder = self.encoder, 
                decoder = self.decoder, 
                beta = self.beta,
                min_dist = self.min_dist, 
                reconstruction_loss=self.reconstruction_loss
                )
            graph = get_umap_graph(
                X = X, 
                n_neighbors = self.n_neighbors, 
                metric=self.metric, 
                random_state=self.random_state
                )
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
                )

        else:
            print("Fitting Non parametric Umap")
            non_parametric_umap = UMAP(
                n_neighbors = self.n_neighbors, 
                min_dist = self.min_dist, 
                metric = self.metric, 
                n_components = self.n_components, 
                random_state = self.random_state, 
                verbose = True
                )
            non_parametric_embeddings = non_parametric_umap.fit_transform(torch.flatten(X, 1, -1).numpy())
            self.model = Model(
                lr = self.lr, 
                encoder = self.encoder, 
                decoder = self.decoder, 
                beta = self.beta, 
                reconstruction_loss = self.reconstruction_loss, 
                match_nonparametric_umap = self.match_nonparametric_umap
                )
            print("Training NN to match embeddings")
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(
                    dataset = MatchDataset(X, non_parametric_embeddings), 
                    batch_size = self.batch_size, 
                    num_workers =   self.num_workers)
            )
        
    @torch.no_grad()
    def transform(self, X):
        print(f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})")
        return self.model.encoder(X).detach().cpu().numpy()
    
    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()


        
def load_pumap(path):

    return Model.load_from_checkpoint(path)

if __name__== "__main__":
    pass