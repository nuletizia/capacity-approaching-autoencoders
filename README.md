# capacity-approaching-autoencoders
This repository contains the official TensorFlow implementation of the following paper:

-- N. A. Letizia, A. M. Tonello, "Capacity-driven Autoencoders for Communications," IEEE Open Journal of Communications Society, 2021 --
-- Preprint: https://arxiv.org/abs/2009.05273 --

If you use the repository for your experiments, please cite the paper.

<img src="https://github.com/nuletizia/capacity-approaching-autoencoders/blob/master/teaser.png" width=700>

The paper deals with autoencoders that are trained by jointly maximizing the mutual information between the transmitted and received data symbols and by minimizing the classical cross-entropy loss function. 

A minimal example of using a pre-trained model is given in Capacity-Approaching_Autoencoders.py. When executed, the script loads a pre-trained autoencoder model, with an AWGN channel, from the folder "Models_AE". If you want to train your own model, please delete the models inside that folder.

Test the model
> python Capacity-Approaching_Autoencoders.py

Train the model

> python Capacity-Approaching_Autoencoders.py --train True
