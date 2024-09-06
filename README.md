
# PHOTON NUMBER CLASSIFICATION

Comparison of different algorithms for the classification of transition edge sensor signals.
With the development of a variety of techniques in the field of A.I. the goal is to quantify the advantages of modern classification techniques 
in the context of photon detection. 

## EXPERIMENTS

The different algorithms are compared in a single notebook available in :

`uniform_TES.ipynb`

The following methods are evaluated :

- Maximum Value
- Area
- Principal Component Analysis (PCA)
- Kernel Principal Component Analysis (K-PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Non-Negative Matrix Factorization (NMF)
- Isomap
- Parametric t-SNE
- Parametric UMAP


## TODO

- Include Sphinx documentation

## ACKNOWLEDGMENTS

We thank the Ministère de l'Économie et de l’Innovation du Québec and the Natural Sciences and Engineering Research Council of Canada for their financial support.

We acknowledge the help of NIST and Guillaume Thekkadath who provided the datasets used in this work.

## REFERENCES

[1] T. Gerrits, B. Calkins, N. Tomlin, A. E. Lita, A. Migdall, R. Mirin, and S. W. Nam, “Extending single-photon optimized superconducting transition edge sensors beyond the single-photon counting regime,” Optics Express, vol. 20, no. 21, pp. 23 798–23 810, Oct. 2012.


[2] G. S. Thekkadath, “Preparing and characterizing quantum states of light using photon-number-resolving detectors.”


[3] Y. Ichinohe et al., ‘Application of Deep Learning to the Evaluation of Goodness in the Waveform Processing of Transition-Edge Sensor Calorimeters’, J Low Temp Phys, vol. 209, no. 5, pp. 1008–1016, Dec. 2022, doi: 10.1007/s10909-022-02719-7.




