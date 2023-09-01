
# SINGLE PHOTON DETECTION

Comparison of different algorithms for classification of transition edge sensor traces for photon number state detection.
With the development of a variety of techniques in the field of A.I. the goal is to quantify the advantages of modern classification techniques 
in the context of photon detection. 

## EXPERIMENTS

The different algorithms are compared in a single notebook available in :

`src/Metrics.ipynb`

The following methods are evaluated :

- Maximum Value
- Area
- Principal Component Analysis (PCA)
- Kernel Principal Component Analysis (K-PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Non-Negative Matrix Factorization (NMF)
- Isomap
- Autoencoder


## AUTOENCODER

The different algorithms are compared in a single notebook available in :

`src/AutoencoderAPI.ipynb`

An autoencoder is a type of neural network trained to reproduce the signal it receives as input. Half the network (encoder) can be used to asign an arbitrary number of parameters to an input signal. This way, the encoder acts as a dimensionality reduction process.

<p align="center">
    <img src="doc/Assets README/BowTie to HalfTie.png"/>
</p>

Neural networks allow for a wide variety of architectures and therefore their evaluation is less straighforward. The `AutoencoderAPI` was designed to explore a variety of algorithms and create a framework to compare their performance.


## TODO

- Pytorch dataset classs structure for batching and true random.
- Random sweep for NN architecture based on current file structure.

## ACKNOWLEDGMENTS



## REFERENCES

[1] Y. Zhang, Q. Shang, and G. Zhang, ‘pyDRMetrics - A Python toolkit for dimensionality reduction quality assessment’, Heliyon, vol. 7, no. 2, p. e06199, Feb. 2021, doi: 10.1016/j.heliyon.2021.e06199.


[2] R. Ran, T. Gao, and B. Fang, ‘Transformer-based dimensionality reduction’. arXiv, Oct. 15, 2022. Accessed: Jul. 17, 2023. [Online]. Available: http://arxiv.org/abs/2210.08288


[3] ‘MultiheadAttention — PyTorch 2.0 documentation’. https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html (accessed Sep. 01, 2023).



