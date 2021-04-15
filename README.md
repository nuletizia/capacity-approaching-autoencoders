# capacity-approaching-autoencoders
This repository contains the official TensorFlow implementation of the following paper:

-- Capacity-driven Autoencoders for Communications --

If you used the repository for your experiments, please cite the paper.

<img src="https://github.com/nuletizia/capacity-approaching-autoencoders/blob/master/teaser.png" width=700>

The paper deals with autoencoders that are trained by jointly maximizing the mutual information between the transmitted and received symbols and minimizing the classical cross-entropy loss function. Capacity is approached at low SNRs while further investigations are needed for large code-length, especially due to numerical issues with the estimator. 

A minimal example of using a pre-trained model is given in Capacity-Approaching_Autoencoders.py. When executed, the script loads a pre-trained autoencoder model, with an AWGN channel, from the folder "Models_AE". If you want to train your own model, please delete the models inside that folder.

Test the model
> python Capacity-Approaching_Autoencoders.py

Train the model

> python Capacity-Approaching_Autoencoders.py --train True
