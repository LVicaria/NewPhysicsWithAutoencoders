
# Autoencoders in Discovering New Physics

This repository contains code and documentation that supplement my dissertation titled **Autoencoders in Discovering New Physics**. 
The objective of this project is to experiment with autoencoders and investigate their potential applications in identifying new patterns in physics data.

## Files in this Repository

- **Autoencoder.py**: This file contains the implementation of the autoencoder model used in the project. The model is designed to learn efficient codings of input data and can be used to uncover latent structures that may indicate new physical phenomena.
  
- **DataLoader1.py**: This file handles the data loading and preprocessing required for training the autoencoder. It includes utilities for managing datasets and preparing them for input into the model.

- **Autoencoders_in_Discovering_New_Physics.pdf**: This is my dissertation, which presents an in-depth exploration of autoencoders in anomaly detection, specifically within the field of particle physics.

## Professor's Contributions

The following utility functions were provided by my professor for use in this project:
- `pad_image`: A function used to pad images to a desired size.
- `normalize`: A function to normalize input data before feeding it to the autoencoder.

## How to Run

1. Clone this repository.
2. Install the required libraries (ensure you have the necessary machine learning libraries such as TensorFlow or PyTorch installed).
3. Run the **Autoencoder.py** file to initiate model training on your dataset.
4. Use **DataLoader1.py** to load and preprocess your data before feeding it to the autoencoder.

## Dissertation Overview

The dissertation explores the use of autoencoders in anomaly detection in particle physics, with a specific focus on the identification of deviations from established physical models. It includes both theoretical and practical aspects, culminating in a hands-on demonstration using quantum chromodynamics and top jet data.
