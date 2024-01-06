
# PHOTON NUMBER CLASSIFICATION

Comparison of different algorithms for classification of transition edge sensor signals for photon number state detection.
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
- Test GPU functionalities of pytorch
- Modify Log structure to reduce the storing and change it to computing when loading results.
- Implementation of confidence measure 
    - [1] T. Schapeler et al., ‘How well can superconducting nanowire single-photon detectors resolve photon number?’, arXiv.org. Accessed: Nov. 21, 2023. [Online]. Available: https://arxiv.org/abs/2310.12471v2
    - [2] P. C. Humphreys, B. J. Metcalf, T. Gerrits, T. Hiemstra, A. E. Lita, J. Nunn, S. W. Nam, A. Datta, W. S. Kolthammer, and I. A. Walmsley, Tomography of photon-number resolving continuous-output detectors, New Journal of Physics 17, 103044 (2015).
- Implementation of Voigt profiles and Cross talk between photon-number clusters
    - [1] G. Sauer, M. Kolarczik, R. Gomez, J. Conrad, and F. Steinlechner, ‘Resolving Photon Numbers Using Ultra-High-Resolution Timing of a Single Low-Jitter Superconducting Nanowire Detector’. arXiv, Oct. 19, 2023. doi: 10.48550/arXiv.2310.12472.

- Prediction histogram/trace/mean traces plot
- Flexible dataset import
    - Non normalized files
    - Batching
    - Various file types
- Automatic method comparison
- Unbalanced dataset option


## ACKNOWLEDGMENTS

We thank the Ministère de l'Économie et de l’Innovation du Québec and the Natural Sciences and Engineering Research Council of Canada for their financial support.

We acknowledge the help of NIST and Guillaume Thekkadath who provided the datasets used in this work.

## REFERENCES

[1] T. Gerrits, B. Calkins, N. Tomlin, A. E. Lita, A. Migdall, R. Mirin, and S. W. Nam, “Extending single-photon optimized superconducting transition edge sensors beyond the single-photon counting regime,” Optics Express, vol. 20, no. 21, pp. 23 798–23 810, Oct. 2012.


[2] G. S. Thekkadath, “Preparing and characterizing quantum states of light using photon-number-resolving detectors.”


[3] Y. Ichinohe et al., ‘Application of Deep Learning to the Evaluation of Goodness in the Waveform Processing of Transition-Edge Sensor Calorimeters’, J Low Temp Phys, vol. 209, no. 5, pp. 1008–1016, Dec. 2022, doi: 10.1007/s10909-022-02719-7.




