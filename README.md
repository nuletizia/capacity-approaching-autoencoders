# Capacity-approaching-autoencoders
This repository contains the official TensorFlow implementation of the following paper:


Capacity-driven Autoencoders for Communications : https://ieeexplore.ieee.org/document/9449919


If you use the repository for your experiments, please cite the paper.

<img src="https://github.com/nuletizia/capacity-approaching-autoencoders/blob/master/teaser.png" width=700>

The paper deals with autoencoders that are trained by jointly maximizing the mutual information between the transmitted and received data symbols and by minimizing the classical cross-entropy loss function. 

A minimal example of using a pre-trained model is given in Capacity-Approaching_AE.py. When executed, the script loads a pre-trained autoencoder model, with an AWGN channel, from the folder "Models_AE". If you want to train your own model, please delete the models inside that folder.

Test the model
> python Capacity-Approaching_AE.py

Train the model
> python Capacity-Approaching_AE.py --train True


# gammaDIME (August 2021)
To change the type of mutual information estimator (MINE is used as default) to gammaDIME (paper available soon), please use the following command:
> python Capacity-Approaching_AE.py --train True --MI_type gammaDIME
