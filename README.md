[![a](https://img.shields.io/static/v1?label=arXiv&message=2411.05737&color=active&style=flat-square)](https://arxiv.org/abs/2411.05737)


# Photon Number Classification

Comparison of different algorithms for the classification of [transition edge sensor](https://en.wikipedia.org/wiki/Transition-edge_sensor) signals.
With the development of a variety of techniques in the field of machine learning the goal is to quantify the advantages of modern classification techniques in the context of photon detection. 

## Files

### Confidence

- [`Methods_Uniform.ipynb`](Methods_Uniform.ipynb)
- [`Methods_Geometric.ipynb`](Methods_Geometric.ipynb)
- [`Methods_Large.ipynb`](Methods_Large.ipynb)
- [`Methods_Noise.ipynb`](Methods_Noise.ipynb)

### Training

- [`Train_PtSNE.ipynb`](Train_PtSNE.ipynb)
- [`Train_PUMAP.ipynb`](Train_PUMAP.ipynb)

### Figures

- [`Effect_Gaussians.ipynb`](Effect_Gaussians.ipynb)
- [`Figures_General.ipynb`](Figures_General.ipynb)
- [`Figures_Methods.ipynb`](Figures_Methods.ipynb)
- [`Find_Distributions.ipynb`](Find_Distributions.ipynb)


## Experiments

The different algorithms are compared in a single notebook available in :

`uniform_TES.ipynb`

The following methods are evaluated :

- [Maximum Value](Figures_Methods.ipynb)
- [Area](Figures_Methods.ipynb)
- [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Kernel Principal Component Analysis (K-PCA)](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
- [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
- [Uniform Manifold Approximation and Projection (UMAP)](https://umap-learn.readthedocs.io/en/latest/)
- [Non-Negative Matrix Factorization (NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
- [Isomap](https://en.wikipedia.org/wiki/Isomap)
- [Parametric t-SNE](https://proceedings.mlr.press/v5/maaten09a.html)
- [Parametric UMAP](https://umap-learn.readthedocs.io/en/latest/)


## TODO

- Include Sphinx documentation

## Acknowledgements

N.D.-C. and N.Q. acknowledge support from the Ministère de l'Économie et de l'Innovation du Québec, the Natural Sciences and Engineering Research Council Canada, Photonique Quantique Québec, and thank S. Montes-Valencia, J. Martinez-Cifuentes and A. Boon for valuable discussions. We also thank Z. Levine and S. Glancy for their careful feedback on our manuscript.






