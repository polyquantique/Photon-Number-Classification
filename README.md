# Single Photon Detection

## Table of Contents  
* [Principal component analysis](#principal-component-analysis)
* [Dimensionality reduction](#dimensionality-reduction)  
* [Neural network](#autoencoder)
* [Futur tasks](#todo)  

## Principal component analysis 
`PCA.ipynb`

Analysis of Guillaume Thekkadath's TES dataset using principal component analysis (PCA). 

## Dimensionality reduction 
`DimensionalityReduction.ipynb`

A qualitative exploration of dimensionality reduction techniques for single photon detection. The goal is to identify methods that can create easily distinguishable clusters. Clustering techniques can afterward be applied to label signals to photon numbers. 

## Autoencoder
`NeuralNetwork.ipynb`

Neural network trained to reproduce the signal it receives as input. Half the network (encoder) can be used to asign an arbitrary number of parameters to an input signal. This way, the encoder acts as a dimensionality reduction process.

<p align="center">
    <img src="Assets/BowTie to HalfTie.png"/>
</p>

## TODO

- Dimensionality reduction metric.
    - Parameter optimization for dimensionality reduction techniques.
