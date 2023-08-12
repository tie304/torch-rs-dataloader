# Torch-rs Dataloader

Currently the pytorch rust bindings lack support for data loading and augmenting.
This is a fun project to write a dataloader in rust to work with the torch bindings in order to train models. 
It uses rusts image library to do transforms but only has a few defined in the repo.

There is both a augmentaion pipline and a transform pipeline. The augumentaion pipline runs first and generates N new images based 
on how many functions you've added to the pipeline. The transform pipeline transforms both the original image and augmented images before going into the network. 

This repo uses the cats VS dogs dataset as an example. 


## Setting up

1. Download liptorch 2.0 https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
2. Download cats and dogs dataset https://www.kaggle.com/c/dogs-vs-cats
3. Orginize into two directories train and test.
4. Change paths in `run.sh` and run

