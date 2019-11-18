# Variational Autoencoder for Voxel Data
This repository contains a sample implementation of a VAE, processing voxel data. 
The architecture  is described [here](https://www.researchgate.net/publication/306187553_Generative_and_Discriminative_Voxel_Modeling_with_Convolutional_Neural_Networks).
In order to train the model you need to generate specific training data which is described [here](https://github.com/galberding/occ_variational_autoencoder).

Adapt the ```train_path``` and ```test_path``` to the corresponding data. 
Start the training with:
```
python genNet.py
```
The results, train and test score as well as the reconstruction of the voxel data can be monitored with tensorboard.
You can do this, the ```OUT_FILE``` can be adapted.
Navigate to the directory and start tensorboard:
```
tensorboard –logdir .
```
